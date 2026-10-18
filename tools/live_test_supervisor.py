"""Operator-side acceptance recovery; never imported by the playing Agent."""
from dataclasses import replace
import json
from pathlib import Path
import time

from majsoul.calibration.frames import write_image
from majsoul.live.collector import TOGGLE_LABELS, TogglePanel, MatchCollector, CollectError
from majsoul.live.bridge import build_domain_state, infer_phase
from majsoul.state.decision import decision_observation

# Stop burning the turn as early as possible, then restore all remaining options.
TAKEOVER_TARGETS=('auto_discard','auto_win','no_call','auto_kita','auto_sort')


class AcceptanceStopped(RuntimeError):
    pass


class AcceptanceGuard:
    """Stop this test on the first failed action or stalled visible turn."""

    def __init__(self, maximum_stall_seconds=8.0, *, local_fallback=False):
        self.maximum_stall_seconds = maximum_stall_seconds
        self.local_fallback = local_fallback
        self.actionable_since = None
        self.failure = None

    def observe(self, event, fields, now):
        if self.failure:
            return
        if self.local_fallback and event in ('verification_unknown', 'verification_failed',
                'preflight_abort', 'decision_expired', 'decision_failed', 'safe_pause'):
            # The independent local watchdog owns the first recovery attempt.
            # Original failure events remain in the audit log.
            return
        if event in ('emergency_failed', 'emergency_unavailable', 'verification_unknown', 'verification_failed', 'preflight_abort',
                     'decision_expired', 'decision_failed', 'safe_pause', 'fatal_error', 'interrupted'):
            self.failure = {'event': event, 'details': fields, 'at': now}
        elif event in ('emergency_verified', 'emergency_resolved', 'verified', 'window_resolved', 'confirm_appeared',
                       'table_ready', 'agent_runtime_reset_done'):
            self.actionable_since = None
        elif event == 'decision_gate':
            clock = fields.get('countdown') or {}
            if clock.get('status') == 'complete' or (self.local_fallback and
                    fields.get('phase') in ('discard', 'response', 'riichi_select')):
                if self.actionable_since is None:
                    self.actionable_since = now
            else:
                self.actionable_since = None
        self.check(now)

    def check(self, now):
        if (not self.failure and self.actionable_since is not None and
                now - self.actionable_since >= self.maximum_stall_seconds):
            self.failure = {'event': 'test_action_stalled', 'at': now,
                            'since': self.actionable_since}
        if self.failure:
            raise AcceptanceStopped(self.failure['event'])


class AgentRecovery:
    """Single-threaded operator handoff, with the client protecting every hand.

    Recovery never executes a model action. It re-enters the ordinary Agent
    only after verified OFF settings and a new legal current observation.
    """
    OFF_TARGETS = ('auto_win', 'no_call', 'auto_kita', 'auto_discard')

    def __init__(self, config, directory, *, maximum_same_failures=3):
        if maximum_same_failures < 1:
            raise ValueError('maximum_same_failures must be positive')
        self.config, self.directory = config, Path(directory)
        self.maximum_same_failures = maximum_same_failures
        self.active = False
        self.failures = []
        self.counts = {}
        self.setup_round = None
        self.retry_at = 0
        self.last_status = None
        self.resume_at = 0

    def start(self, loop, failure):
        self.active = True  # Disable the test observer before logging or toggling.
        self.failures.append(failure)
        details = failure.get('details') or {}
        key = (failure['event'], details.get('reason') or details.get('error') or
               tuple(sorted({i.split(':', 1)[0] for i in details.get('issues', ())})))
        self.counts[key] = self.counts.get(key, 0) + 1
        loop.log('control_transferred', owner='client', reason=failure,
                 frame_index=loop._frame_index, same_failure_count=self.counts[key])
        # Auto-discard is enabled first; rebuilding is done under protection.
        self.setup_round = None
        self.retry_at = 0
        self.resume_at = 0
        self._enable(loop)
        loop.reset_runtime('failure recovery', restart_policy=True)
        if self.counts[key] >= self.maximum_same_failures:
            loop.log('recovery_exhausted', reason='repeated_same_failure',
                     failure=failure, count=self.counts[key])
            loop._notify_session('Live-machine test stopped: the same or a similar cause has already failed '
                + str(self.counts[key]) + ' times. The client continues takeover until the end of the match. Diagnosis: '
                + json.dumps(failure, ensure_ascii=False))
            raise AcceptanceStopped('repeated_same_failure')

    def _enable(self, loop):
        report = enable_client_takeover(loop.window, self.config,
            self.directory / f'failure-{len(self.failures):02d}' /
            f'round-{loop.session_ui.lifecycle.round_index:03d}')
        loop.log('client_takeover_status', **report)
        self.retry_at = loop.now() + self.config.toggle_retry_seconds
        if report['verified']:
            self.setup_round = loop.session_ui.lifecycle.round_index
        return report['verified']

    def _wait(self, loop, reason, **fields):
        status = (reason, str(fields))
        if status != self.last_status:
            self.last_status = status
            loop.log('recovery_waiting', reason=reason, **fields)
        return False

    def step(self, loop, frame, index):
        if loop._session_frame(frame):
            return loop.session_ui.lifecycle.complete
        current_round = loop.session_ui.lifecycle.round_index
        if self.setup_round != current_round:
            if loop.now() >= self.retry_at:
                self._enable(loop)
            return False  # Always capture again after drawer input.
        roi = loop._resolve_roi(frame)
        tracked = loop._track(frame.image, roi, loop._active_slots,
                              frame.captured_at, index, frame)
        if tracked is None:
            return self._wait(loop, 'perception_failed')
        observation, _, _ = tracked
        if loop.now() < self.resume_at:
            return False
        current = decision_observation(observation, loop.tracker.self_slot)
        issues = loop.tracker.validator.validate_observation(current,
            allow_ui_occlusion=True, allow_public_uncertainty=True)
        if issues or loop._frame_unknowns:
            loop._frame_unknowns = []
            return self._wait(loop, 'table_not_recovered', issues=list(issues))
        own = observation.player(loop.tracker.self_slot)
        if (observation.action_buttons or observation.drawn_index is not None or
                len(observation.hand) != 13 - 3 * len(own.melds)):
            return self._wait(loop, 'waiting_for_opponent_turn')
        # Disable auto-discard last, so other setup work cannot burn our turn.
        loop.log('agent_resume_preflight', frame_index=index, round_index=current_round)
        try:
            panel = TogglePanel(loop.window, replace(self.config, toggle_targets=self.OFF_TARGETS),
                                sleep=loop.sleep)
            missing = panel.ensure(enabled=False)
            if missing:
                raise CollectError('Client auto options still enabled: ' + ', '.join(missing))
            fresh, fresh_index = loop._capture()
            if fresh.window != frame.window or loop.session_ui.classify(fresh)[:2] != (False, None):
                raise CollectError('Window or lifecycle changed during handoff')
            fresh_tracked = loop._track(fresh.image, loop._resolve_roi(fresh), loop._active_slots,
                                       fresh.captured_at, fresh_index, fresh)
            if fresh_tracked is None or loop._frame_unknowns:
                raise CollectError('Fresh input unavailable after disabling auto options')
            raw, visible, _ = fresh_tracked
            current = decision_observation(raw, loop.tracker.self_slot)
            state = loop.tracker.decision_snapshot(current)
            phase = infer_phase(current, state, pending_riichi=False)
            if phase not in ('waiting', 'discard', 'response'):
                raise CollectError('No valid listening or action state after handoff')
            if phase != 'waiting':
                if not current.countdown or current.countdown.get('status') != 'complete':
                    raise CollectError('Current action deadline is unrecognized after handoff')
                domain = build_domain_state(state, current, visible, loop._buttons(),
                                            pending_riichi=False, window=fresh.window)
                if not loop.engine.legal_actions(domain):
                    raise CollectError('No legal current action after handoff')
            if not 0 <= loop.now() - fresh.captured_at <= loop.config.decision.max_decision_age:
                raise CollectError('Handoff observation expired')
        except (KeyboardInterrupt, SystemExit):
            raise  # The runner finally block keeps the current match protected.
        except Exception as exc:
            loop.log('agent_resume_rejected', reason=f'{type(exc).__name__}: {exc}')
            self.setup_round = None
            self.retry_at = 0
            self._enable(loop)
            self.resume_at = loop.now() + self.config.toggle_retry_seconds
            return False
        loop.decision_budget.reset()
        loop._actionable_since = None
        self.active = False
        self.last_status = None
        loop.log('control_transferred', owner='agent', reason='current_state_and_auto_off_verified',
                 frame_index=fresh_index, phase=phase, round_index=current_round)
        # This frame is not reused for a click. Ordinary fresh preflight applies.
        return False


def enable_client_takeover(window, config, directory: Path):
    """After Agent shutdown, enable and verify all five using the collector UI."""
    directory.mkdir(parents=True, exist_ok=True)
    config = replace(config, toggle_targets=TAKEOVER_TARGETS)
    snapshots = []

    def record(frame, label):
        path = directory / f'{len(snapshots):02d}-{label}.png'
        write_image(path, frame.image)
        snapshots.append(str(path))
        return path

    report = {'targets': list(TOGGLE_LABELS), 'verified': False, 'attempts': [],
              'snapshots': snapshots}
    try:
        panel = TogglePanel(window, config, recorder=record)
        for _ in range(config.toggle_max_attempts):
            # Each ensure reobserves stable OFF before a click; never blind toggling.
            try:
                missing = panel.ensure(enabled=True)
            except Exception as exc:
                # Drawer animation/focus may reject the first opening. Reobserve
                # immediately within the existing bound instead of abandoning
                # the urgent setup until the collector's later retry timer.
                report['attempts'].append({'error': f'{type(exc).__name__}: {exc}'})
                continue
            report['attempts'].append({'missing': missing})
            if not missing:
                report['verified'] = True
                break
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        (directory / 'takeover.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def finish_with_client_takeover(window, config, live, directory: Path, *, maximum_seconds=1800,
                                sleep=time.sleep, now=time.monotonic, notifier=None,
                                failure_report=None):
    """Keep collector lifecycle active through this match, without starting another.

    UI recovery stays independent of CNN/state/Agent decisions. The collector
    already re-verifies settings after each hand's result disappears.
    """
    config = replace(config, dataset_root=directory.parent, toggle_targets=TAKEOVER_TARGETS)
    deadline = now() + maximum_seconds

    class CurrentMatchTakeover(MatchCollector):
        already_at_lobby = False

        def _capture(self):
            if now() >= deadline:
                raise CollectError('Current-match takeover time limit reached')
            return super()._capture()

        def _enter_match(self):
            # Intentionally no MenuNavigator: only finish the existing match.
            frame = self._capture()
            self._resolve_roi(frame)
            self.already_at_lobby = self._menu_score(frame) >= self.config.menu_threshold
            if failure_report is not None:
                # Use the protective collector's queue: the stopped Agent's
                # intentionally short shutdown must not truncate this report.
                self._notify('Agent stop diagnosis: ' + json.dumps(failure_report, ensure_ascii=False))

        def _play_match(self):
            if self.already_at_lobby:
                self.sleep(self.config.table_ready_seconds)
                frame = self._capture()
                if self._menu_score(frame) < self.config.menu_threshold:
                    raise CollectError('Lobby changed during takeover preflight')
                self._save_frame(frame, 'takeover_lobby_verified')
                return
            return super()._play_match()

        def _track(self, frame, roi_eff):
            # This is protective client control, not semantic dataset acceptance.
            return None

        def _notify(self, message, image=None, *, metadata=None):
            return super()._notify('Test-failure takeover (client plays on its behalf, not model acceptance): ' + message,
                                   image, metadata=metadata)

    report = {'mode': 'client_failure_takeover', 'agent_acceptance': False,
              'maximum_seconds': maximum_seconds, 'targets': list(TOGGLE_LABELS)}
    collector = None
    try:
        collector = CurrentMatchTakeover(window, config, live, max_matches=1,
                                        session=directory.name, sleep=sleep, now=now,
                                        notifier=notifier)
        report['exit_code'] = collector.run()
        report['already_at_lobby'] = collector.already_at_lobby
        report['returned_lobby'] = report['exit_code'] == 0
    except Exception as exc:
        report.update(exit_code=1, returned_lobby=False, error=f'{type(exc).__name__}: {exc}')
    finally:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'takeover-session.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
