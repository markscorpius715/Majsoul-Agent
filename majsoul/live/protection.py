"""Independent process supervision. Only one child at a time may send input.

The supervisor does not classify tiles, call a policy, or wait for an action
posterior. A failed/stalled Agent is fenced by terminating and joining its
process before the client protector is started. A takeover is temporary: an
observe-only rejoin Agent resynchronizes, and at an opponent turn the protector
verifies the client auto options OFF and is fenced before that Agent is granted
input. Fatal reasons or exhausted recovery budgets latch protection until the
current match reaches the lobby; it never starts another match.
"""
from dataclasses import dataclass
import json
import math
import multiprocessing as mp
from pathlib import Path
from queue import Empty, Full
import time

from .operation_clock import OperationClock
from .processes import (CHECKPOINT, FORWARDED_EVENTS, WindowLease, _drain, _emit,
                        _input_window, _run_agent, read_checkpoint, stop_input_owner)


@dataclass(frozen=True)
class ProtectionConfig:
    enabled: bool = False
    poll_seconds: float = .1
    observer_seconds: float = .2
    heartbeat_seconds: float = 2.0
    observation_seconds: float = 2.0
    startup_seconds: float = 30.0
    recognition_seconds: float = 30.0
    maximum_operation_seconds: float = 8.0
    shortest_turn_seconds: float = 5.0
    takeover_margin_seconds: float = 2.5
    quantization_seconds: float = 1.0
    stop_grace_seconds: float = .1
    worker_retry_seconds: float = .5
    uncertain_seconds: float = 1.0
    resolution_grace_seconds: float = .3
    modal_manifest: Path | None = None
    # Temporary takeover and verified handback.
    recovery_enabled: bool = True
    warm_rejoin: bool = True
    rejoin_delay_seconds: float = 1.0
    rejoin_backoff_max_seconds: float = 20.0
    rejoin_startup_seconds: float = 90.0
    rejoin_stall_seconds: float = 10.0
    handback_stable_frames: int = 3
    handback_stable_seconds: float = 1.5
    handback_ready_max_age: float = 1.0
    handback_release_seconds: float = 15.0
    handbacks_per_hand: int = 2
    handbacks_per_match: int = 8
    same_reason_limit: int = 4
    soft_failure_limit: int = 4
    soft_failure_dedupe_seconds: float = 1.0
    prematch_retries: int = 3
    flap_window_seconds: float = 20.0

    BOOLEANS = ('enabled', 'recovery_enabled', 'warm_rejoin')
    COUNTS = ('handback_stable_frames', 'handbacks_per_hand', 'handbacks_per_match',
              'same_reason_limit', 'soft_failure_limit', 'prematch_retries')

    @classmethod
    def from_dict(cls, data, root):
        values = dict(data or {})
        if set(values) - set(cls.__dataclass_fields__):
            raise ValueError('protection contains unsupported configuration')
        if values.get('modal_manifest'):
            path = Path(values['modal_manifest'])
            values['modal_manifest'] = path if path.is_absolute() else root/path
        obj = cls(**values)
        for key in cls.BOOLEANS:
            if type(getattr(obj, key)) is not bool:
                raise ValueError(f'protection.{key} must be boolean')
        for key in cls.COUNTS:
            value = getattr(obj, key)
            if type(value) is not int or not (0 if key == 'prematch_retries' else 1) <= value <= 100:
                raise ValueError('Invalid protection '+key)
        for key, value in vars(obj).items():
            if key not in cls.BOOLEANS + cls.COUNTS + ('modal_manifest',) and (
                    type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 120):
                raise ValueError('Invalid protection '+key)
        if obj.takeover_margin_seconds + obj.quantization_seconds >= obj.shortest_turn_seconds:
            raise ValueError('Protection must leave time before the shortest turn expires')
        return obj


# No-input outcomes the Agent recovers from itself. The independent operation
# deadline still covers a missed turn; only a repeated run escalates.
SOFT_FAILURES = frozenset(('verification_unknown', 'preflight_abort', 'decision_expired',
    'decision_failed', 'safe_pause', 'emergency_unavailable'))
FAILURES = frozenset(('verification_failed', 'fatal_error', 'emergency_failed',
    'agent_stopped', 'worker_exception'))
COMPLETIONS = frozenset(('verified', 'window_resolved', 'emergency_verified', 'emergency_resolved'))
HANDBACK_EVENTS = frozenset(('agent_ready_for_handback', 'input_granted'))
# Reasons that never hand input back to an Agent during this match.
FATAL_KINDS = frozenset(('operator_stop', 'operator_interrupt', 'operator_stop_before_start'))
# Restored only after the auto options are verified OFF; auto-sort stays ON and
# auto-discard is disabled last so setup work cannot consume an own turn.
HANDBACK_OFF_TARGETS = ('auto_win', 'no_call', 'auto_kita', 'auto_discard')


def recovery_class(reason, *, seen_match, lobby_verified):
    """'fatal' latches protection, 'prematch' restarts the Agent from the lobby,
    and anything else is a temporary takeover followed by a verified handback."""
    kind = (reason or {}).get('kind')
    if kind in FATAL_KINDS:
        return 'fatal'
    if kind == 'observation_error' and 'input_desktop_blocked' in str(reason.get('error', '')):
        return 'fatal'  # A confirmed lock screen is never handled automatically.
    if lobby_verified and not seen_match:
        return 'prematch'
    return 'recoverable'


class ProtectionMonitor:
    """Clock-driven operation budget, independent of the Agent's frame loop."""
    def __init__(self, config, now, *, matching_seconds=None, require_agent_recognition=False):
        self.config = config
        self.require_agent_recognition=require_agent_recognition
        self.agent_recognized=not require_agent_recognition
        self.recognition_since=None
        self.matching_seconds=config.startup_seconds if matching_seconds is None else matching_seconds
        if (type(self.matching_seconds) not in (int,float) or
                not math.isfinite(self.matching_seconds) or self.matching_seconds<=0):
            raise ValueError('Matching timeout must be finite and positive')
        self.matching_since=None
        self.started = self.last_progress = self.last_observation = now
        self.ready = False
        self.pre_match_since = now
        self.seen_match = False
        self.operation_since = self.operation_due = None
        self.operation_origin = None
        self.last_sample = None
        self.reason = None
        self.completed_at = float('-inf')
        self.table_seen = False
        self.lobby_since = None
        self.lobby_verified = False
        self.waiting_since = self.uncertain_since = None
        self.uncertain_changed_at = self.uncertain_hand_count = None
        self.last_input = float('-inf')
        self.last_stage = None
        self.clock = OperationClock.from_protection(config)
        self.idle_at = None
        self.carried_reading = None
        self.soft_failures = 0
        self.soft_last = float('-inf')
        self.hand_epoch = 0
        self.last_confirm = False

    def rearm(self, now):
        """A verified handback starts a fresh Agent budget. Samples captured
        before ``now`` can neither reopen nor extend an operation window."""
        self.reason = None
        self.agent_recognized = self.seen_match = self.ready = True
        self.operation_since = self.operation_due = self.operation_origin = None
        self.waiting_since = self.uncertain_since = None
        self.clock.reset()
        self.idle_at = now
        self.uncertain_changed_at = self.uncertain_hand_count = None
        self.completed_at = now
        self.last_progress = max(self.last_progress, now)
        self.last_input = float('-inf')
        self.soft_failures = 0
        self.soft_last = float('-inf')

    def restart_prematch(self, now):
        """A replacement Agent gets the original bounded menu/matching waits."""
        self.reason = None
        self.ready = False
        self.agent_recognized = not self.require_agent_recognition
        self.pre_match_since = self.last_progress = now
        self.matching_since = self.recognition_since = None
        self.soft_failures = 0

    def trip(self, reason):
        if self.reason is None:
            self.reason = reason
        return self.reason

    def event(self, event, fields, now):
        self.last_progress = max(self.last_progress, now)
        if event == 'heartbeat' and fields.get('stage'):
            self.last_stage = fields['stage']
        if event == 'worker_ready':
            if not self.ready:
                # Loading the models and navigating/matching have separate
                # bounded waits; initialization must not consume the latter.
                self.pre_match_since = max(self.pre_match_since, now)
            self.ready = True
        if event=='table_wait_started' and self.matching_since is None:
            self.matching_since=now
            self.recognition_since=now
        if event=='agent_game_started' and not self.agent_recognized:
            self.agent_recognized=True
            self.seen_match=True
            self._close_operation(now)
            self.waiting_since=self.uncertain_since=None
            self.completed_at=now
        if event=='heartbeat' and fields.get('stage')=='input_end' and fields.get('input_completed'):
            self.last_input=now
        if event in COMPLETIONS:
            self.completed_at = now
            # A verification can arrive after the observer already saw the
            # next window (e.g. the discard after a nuki replacement draw).
            # A reading captured after the input belongs to that window and
            # is never later than its real deadline, so it carries over.
            reading = self.clock.last_reading
            self._close_operation(now)
            if reading is not None and reading[0] > self.last_input:
                self.carried_reading = reading
            self.waiting_since = None
            self.soft_failures = 0
        if event in SOFT_FAILURES:
            # Several log lines may describe one missed window; count it once.
            if now-self.soft_last >= self.config.soft_failure_dedupe_seconds:
                self.soft_failures += 1
                self.soft_last = now
            if self.soft_failures >= self.config.soft_failure_limit:
                self.trip({'kind': 'repeated_soft_failures', 'event': event,
                           'count': self.soft_failures, 'details': fields})
        # A successful stop in the lobby is still checked by the observer.
        if event in FAILURES and not fields.get('match_complete', False):
            self.trip({'kind': 'agent_failure', 'event': event, 'details': fields})

    def observe(self, sample, now):
        captured = sample['captured_at']
        if not math.isfinite(captured) or captured>now+.01 or now-captured>self.config.observation_seconds:
            self.lobby_verified=False;self.lobby_since=None
            self.trip({'kind':'stale_or_invalid_observer_frame','captured_at':captured})
            return
        # Never use delayed/out-of-order IPC observations to postpone protection.
        if captured <= self.last_observation and self.last_sample is not None:
            return
        if sample.get('desktop_status',{}).get('status')=='unknown':
            # A focus transition may temporarily have no queryable foreground.
            # It authorizes no input and renews neither observation freshness nor
            # any operation deadline; the existing observer timeout still trips.
            self.last_sample=sample
            self.lobby_verified=False;self.lobby_since=None
            return
        if captured-self.last_observation > self.config.observation_seconds:
            self.lobby_verified = False
            self.lobby_since = None
        self.last_observation = captured
        self.last_sample = sample
        if sample.get('error'):
            self.lobby_verified=False;self.lobby_since=None;self.table_seen=False
            self.trip({'kind':'observation_error','error':sample['error']})
            return
        confirm = bool(sample.get('confirm'))
        if confirm and not self.last_confirm:
            self.hand_epoch += 1  # Result overlays separate hands for recovery budgets.
        self.last_confirm = confirm
        self.table_seen = bool(sample.get('table'))
        self.seen_match |= bool(sample.get('table') or sample.get('confirm') or sample.get('modal'))
        if sample.get('lobby'):
            if self.lobby_since is None:
                self.lobby_since = captured
            self.lobby_verified = captured-self.lobby_since >= 1.0
            self._close_operation(captured)
            self.uncertain_since = self.waiting_since = None
            return
        self.lobby_since = None
        self.lobby_verified = False
        if sample.get('modal'):
            self.trip({'kind':'modal','name':sample['modal']})
            return
        if sample.get('confirm'):
            self._close_operation(captured)
            self.uncertain_since = self.waiting_since = None
            return
        if captured <= self.completed_at:
            return
        if not self.agent_recognized:
            # Before Agent recognition, table artwork, dealt faces and unknown
            # clock fragments are diagnostics, not a live-operation gate.
            return
        if sample.get('uncertain') and self.seen_match:
            count=sample.get('hand_count')
            if self.uncertain_since is None:
                self.uncertain_since=self.uncertain_changed_at=captured
            elif (type(count) is int and type(self.uncertain_hand_count) is int
                  and count != self.uncertain_hand_count):
                self.uncertain_changed_at=captured
            self.uncertain_hand_count=count
        else:self.uncertain_since=None
        if sample.get('actionable'):
            self.waiting_since=None
            if self.operation_since is None:
                self.operation_since = captured
                self.operation_origin = dict(sample)
                # An observer gap can hide the real start; the base-time
                # fallback counts from the last proof the window was closed.
                carried, self.carried_reading = self.carried_reading, None
                # The carried reading was the first sight of this window, so the
                # maximum operation time also counts from it.
                self.clock.start(captured if carried is None else min(captured, carried[0]), self.idle_at)
                if carried is not None:
                    self.clock.observe(*carried)
            # The clock is read once and counted down locally: later readings
            # can only shorten the deadline, and unreadable frames are ignored.
            self.clock.observe(captured, sample.get('total_seconds'))
            self.operation_due = self.clock.due
        elif sample.get('waiting'):
            self.carried_reading = None
            if self.operation_due is None:
                self.idle_at = captured
            else:
                # An unverified turn ending can mean a client timeout. Never
                # reset and silently forget it at the next five-second window.
                if self.operation_since<=self.last_input<captured and captured<self.operation_due:
                    # Early waiting after sent input proves the operation ended,
                    # not that the model's requested tile was played correctly.
                    self._close_operation(captured)
                    self.waiting_since=None
                elif self.waiting_since is None:self.waiting_since=captured

    def _close_operation(self, at):
        self.operation_since = self.operation_due = None
        self.clock.reset()
        self.idle_at = at
        self.carried_reading = None

    def check(self, now):
        if now-self.last_observation > self.config.observation_seconds:
            # A latched protection reason must not preserve an old lobby
            # certificate while the independent observer is unavailable.
            self.lobby_verified = False
            self.lobby_since = None
        if self.reason:
            return self.reason
        if not self.agent_recognized:
            if self.recognition_since is not None:
                if now-self.recognition_since>=self.config.recognition_seconds:
                    return self.trip({'kind':'agent_recognition_timeout','since':self.recognition_since,
                                      'limit':self.config.recognition_seconds})
            elif now-self.pre_match_since>=self.config.startup_seconds:
                return self.trip({'kind':'pre_match_timeout','since':self.pre_match_since})
            if now-self.last_observation>self.config.observation_seconds:
                return self.trip({'kind':'observer_stalled','last_observation':self.last_observation})
            return None
        if self.operation_due is not None and now >= self.operation_due:
            return self.trip({'kind':'operation_deadline','since':self.operation_since,'due':self.operation_due,
                              'clock':self.clock.to_dict(),'operation_origin':self.operation_origin})
        if self.waiting_since is not None and now-self.waiting_since>=self.config.resolution_grace_seconds:
            return self.trip({'kind':'unverified_operation_ended','since':self.operation_since})
        if self.uncertain_since is not None:
            # Dealing/sorting changes the local face count while the table UI
            # is already visible. Allow that observed animation to finish,
            # but cap the entire uncertainty interval before even a shortest
            # turn could expire after quantization and client takeover.
            cap=(self.config.shortest_turn_seconds-self.config.quantization_seconds
                 -self.config.takeover_margin_seconds)
            if (now-self.uncertain_changed_at>=self.config.uncertain_seconds or
                    now-self.uncertain_since>=cap):
                return self.trip({'kind':'critical_local_ui_uncertain','since':self.uncertain_since,
                                  'last_count_change':self.uncertain_changed_at})
        since=self.pre_match_since if self.matching_since is None else self.matching_since
        limit=self.config.startup_seconds if self.matching_since is None else self.matching_seconds
        if not self.seen_match and now-since>=limit:
            return self.trip({'kind':'pre_match_timeout','since':since})
        if now-self.last_observation > self.config.observation_seconds:
            return self.trip({'kind':'observer_stalled','last_observation':self.last_observation})
        limit = self.config.heartbeat_seconds if self.ready or self.table_seen else self.config.startup_seconds
        if now-self.last_progress > limit:
            return self.trip({'kind':'agent_stalled','last_progress':self.last_progress,
                              'limit':limit,'last_stage':self.last_stage})
        return None


def _observer(config, hwnd, output, stop, evidence_directory=None):
    # No CNN, global Tracker, OCR, policy, notifications or game input here.
    from .capture import LiveWindow
    from .modals import input_desktop_status
    from .protection_ui import ProtectionUI
    window = LiveWindow(hwnd)
    window.ignore_stop_key = True
    reader=ProtectionUI(config)
    if evidence_directory is not None:evidence_directory.mkdir(parents=True,exist_ok=True)
    previous=None
    sequence=0
    while not stop.is_set():
        try:
            desktop=input_desktop_status()
            if desktop['status']!='available':
                sample={'captured_at':time.monotonic(),'desktop_status':desktop}
                if desktop['status']=='blocked':sample['error']='input_desktop_blocked'
                if evidence_directory is not None:
                    with (evidence_directory/'samples.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(sample,ensure_ascii=False)+'\n')
                _emit(output,kind='sample',sample=sample)
                stop.wait(config.protection.observer_seconds)
                continue
            frame = window.capture()
            sample=reader.read(frame)
            state=tuple((key,str(value)) for key,value in sample.items()
                        if key not in ('captured_at','total_seconds'))
            if evidence_directory is not None and state!=previous:
                from ..calibration.frames import write_image
                sequence+=1
                path=evidence_directory/f'{sequence:06d}.jpg'
                write_image(path,frame.image)
                sample['evidence']=str(path)
                with (evidence_directory/'samples.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(sample,ensure_ascii=False)+'\n')
            previous=state
            _emit(output,kind='sample',sample=sample)
        except (Exception,KeyboardInterrupt) as exc:
            _emit(output,kind='sample',sample={'captured_at':time.monotonic(),'error':f'{type(exc).__name__}: {exc}'})
        stop.wait(config.protection.observer_seconds)


def _agent(config, hwnd, session, skip_menu, once, output,active=None):
    _run_agent(config,hwnd,session,output,active,skip_menu=skip_menu,once=once)


def _rejoin(config, hwnd, session, output, active, granted, activation, ready, stable_frames):
    """Warm, observe-only Agent which joins the current match mid-hand.

    It reports ``agent_ready_for_handback`` from consecutive validated opponent
    turn snapshots. Its window refuses every input until ``granted`` is set."""
    _run_agent(config,hwnd,session,output,active,granted=granted,activation=activation,
               ready=ready,skip_menu=True,join=True,input_gate=lambda:bool(granted.value),
               handback_frames=lambda:max(1,int(stable_frames.value)))


class HandbackReleased(Exception):
    """The protector verified every handback option OFF and sends no more input."""


def _protect(config, hwnd, directory, output, release_mouse=False,active=None,activation=None,ready=None,
             release=None):
    """Reuse the collector's verified toggles and round/result lifecycle."""
    from dataclasses import replace
    from .collector import CollectConfig, MatchCollector, TogglePanel, CollectError
    from .modals import ModalReader, input_desktop_available
    from ..window import _user32, WindowError
    modals = ModalReader(config.protection.modal_manifest)
    window = _input_window(hwnd,output,active,ignore_stop_key=True)
    class Protector(MatchCollector):
        modal_attempts = None
        handback_reader = None
        def _table_hook(self, frame):
            if release is None or not release.is_set():
                return False
            if self.handback_reader is None:
                from .protection_ui import ProtectionUI
                self.handback_reader = ProtectionUI(config)
            sample = self.handback_reader.read(frame)
            if not (sample['table'] and sample['waiting']) or sample['modal'] or sample['confirm']:
                return False  # Only a visible opponent turn is a safe handback point.
            self.log('handback_toggle_off_started')
            try:
                panel = TogglePanel(self.window, replace(self.config, toggle_targets=HANDBACK_OFF_TARGETS),
                                    capture=self._capture, sleep=self.sleep, recorder=self._save_frame)
                missing = panel.ensure(enabled=False)
                if missing:
                    raise CollectError('auto options still enabled: '+', '.join(missing))
                fresh = self._capture()
                if self.ui.confirm(fresh.image) or not self.ui.table_visible(fresh.image):
                    raise CollectError('round lifecycle changed during handback')
            except (CollectError, WindowError) as exc:
                release.clear()
                self.log('handback_rejected', error=str(exc))
                return True  # Re-verify every protection option before continuing.
            self.log('handback_released', missing=[])
            raise HandbackReleased()
        def log(self,event,**fields):
            _emit(output,kind='event',event=event,fields=fields)
            return super().log(event,**fields)
        def _capture(self):
            while True:
                _emit(output,kind='event',event='heartbeat',fields={'stage':'protect_capture'})
                if not input_desktop_available():
                    raise RuntimeError('input_desktop_locked_or_unavailable')
                frame = super()._capture()
                modal = modals.read(frame.image)
                if not modal:
                    return frame
                self.log('modal_observed',name=modal)
                if not modals.recoverable(modal):
                    raise RuntimeError('external_modal_requires_user:'+modal)
                # Match only reviewed modal text AND its button, reobserve
                # before input, and never click an obscured result underneath.
                fresh = window.capture()
                if fresh.window != frame.window or modals.read(fresh.image) != modal:
                    continue
                if self.modal_attempts is None:self.modal_attempts={}
                count=self.modal_attempts.get(modal,0)
                if count>=3:
                    raise RuntimeError('reviewed_modal_did_not_close:'+modal)
                point = modals.point(modal,fresh.image.shape[1::-1])
                window.click(point,expected=fresh)
                self.modal_attempts[modal]=count+1
                self.log('modal_confirm_clicked',name=modal)
                self.sleep(.2)
        def _enter_match(self):
            frame=self._capture()
            self.already_lobby=self._menu_score(frame)>=self.config.menu_threshold
        def _play_match(self):
            if not self.already_lobby:
                return super()._play_match()
        def _track(self,*args):
            return None
        def _setup_toggles(self,panel):
            # Prove auto-discard first. Do not spend the short operation window
            # setting convenience options while its safety switch is unknown.
            urgent=TogglePanel(self.window,replace(self.config,toggle_targets=('auto_discard',)),
                capture=self._capture,sleep=self.sleep,recorder=self._save_frame)
            missing=urgent.ensure(enabled=True)
            if missing:
                self.log('protect_auto_discard_unverified',missing=missing)
                raise CollectError('Urgent auto-discard was not verified')
            self.log('protect_auto_discard_verified')
            return super()._setup_toggles(panel)
    collect=replace(CollectConfig.load(config.lifecycle_ui),dataset_root=directory.parent,
        toggle_targets=('auto_discard','auto_win','no_call','auto_kita','auto_sort'),
        table_ready_seconds=.1)
    try:
        p=Protector(window,collect,config,max_matches=1,session=directory.name)
        _emit(output,kind='event',event='worker_ready',fields={})
        if ready is not None:ready.set()
        if activation is not None:activation.wait()
        if getattr(release_mouse,'value',release_mouse) and input_desktop_available():
            _user32().mouse_event(0x0004,0,0,0,0)  # LEFTUP after a killed input owner.
        code=p.run()
        _emit(output,kind='event',event='worker_exit',fields={'exit_code':code})
    except HandbackReleased:
        # Hold the window without input until the supervisor fences this
        # process; only then may the rejoined Agent be granted input.
        while True:
            _emit(output,kind='event',event='heartbeat',fields={'stage':'released_idle'})
            time.sleep(.1)
    except BaseException as exc:
        _emit(output,kind='event',event='worker_exception',fields={'error':f'{type(exc).__name__}: {exc}'})


def _standby(config,hwnd,directory,output,release_mouse,active,activation,ready,target,release=None):
    if target is _protect:
        target(config,hwnd,directory,output,release_mouse,active,activation,ready,release)
    else:
        # Fault-injection targets also obey the same explicit activation gate.
        ready.set();activation.wait()
        target(config,hwnd,directory,output,bool(release_mouse.value),active,release)


class RecoveryLedger:
    """Budgets and metrics for temporary takeovers. It never sends input."""
    def __init__(self, config):
        self.config = config
        self.trips = []
        self.handbacks = []
        self.same_reason = {}
        self.per_hand = {}
        self.flap_streak = 0
        self.prematch_restarts = 0
        self.latched = None
        self.last_handback_at = None
        self.client_used = False
        self.agent_table_seconds = self.client_table_seconds = 0.0

    @staticmethod
    def key(reason):
        return f"{reason.get('kind')}:{reason.get('event') or ''}"

    def record_trip(self, reason, now, klass):
        # A trip soon after a handback is a flap: back off and demand a longer
        # stable resynchronization before the next handback.
        if (self.last_handback_at is not None and
                now-self.last_handback_at <= self.config.flap_window_seconds):
            self.flap_streak += 1
        else:
            self.flap_streak = 0
        key = self.key(reason)
        self.same_reason[key] = self.same_reason.get(key, 0)+1
        self.trips.append({'at': round(now, 3), 'kind': reason.get('kind'), 'event': reason.get('event'),
                           'class': klass, 'key': key, 'flap_streak': self.flap_streak})

    def exhausted(self):
        if not self.config.recovery_enabled:
            return 'recovery_disabled'
        if len(self.handbacks) >= self.config.handbacks_per_match:
            return 'match_handback_budget_exhausted'
        if max(self.same_reason.values(), default=0) >= self.config.same_reason_limit:
            return 'repeated_same_reason'
        return None

    def delay(self):
        return min(self.config.rejoin_delay_seconds*2**self.flap_streak,
                   self.config.rejoin_backoff_max_seconds)

    def stable_frames(self):
        return min(100, self.config.handback_stable_frames+2*self.flap_streak)

    def hand_allows(self, hand):
        return self.per_hand.get(hand, 0) < self.config.handbacks_per_hand

    def record_handback(self, now, hand):
        trip = self.trips[-1] if self.trips else None
        self.per_hand[hand] = self.per_hand.get(hand, 0)+1
        self.last_handback_at = now
        self.handbacks.append({'at': round(now, 3), 'hand': hand, 'after': trip['key'] if trip else None,
                               'recovery_seconds': round(now-trip['at'], 3) if trip else None})
        return self.handbacks[-1]

    def account(self, mode, seconds, on_table):
        if on_table and seconds > 0:
            if mode == 'agent':
                self.agent_table_seconds += seconds
            else:
                self.client_table_seconds += seconds

    def report(self):
        times = [h['recovery_seconds'] for h in self.handbacks if h['recovery_seconds'] is not None]
        total = self.agent_table_seconds+self.client_table_seconds
        return {'trips': self.trips, 'handbacks': self.handbacks, 'latched': self.latched,
                'prematch_restarts': self.prematch_restarts,
                'mean_recovery_seconds': round(sum(times)/len(times), 3) if times else None,
                'max_recovery_seconds': max(times) if times else None,
                'agent_table_seconds': round(self.agent_table_seconds, 3),
                'client_table_seconds': round(self.client_table_seconds, 3),
                'agent_control_ratio': round(self.agent_table_seconds/total, 4) if total else None}


def run_protected(config, hwnd, *, skip_menu=False, once=False, session=None, notifier=None, _targets=None):
    """Public execute entry point, one current match, protect through lobby."""
    from dataclasses import replace
    from .notify import FeishuNotifier
    from .notifications import NotificationQueue
    from .menu import MenuConfig
    name=session or ('protected-'+time.strftime('%Y%m%d-%H%M%S')+f'-{time.time_ns()%1000000:06d}')
    if Path(name).name!=name or name in ('.','..') or '/' in name or '\\' in name:
        raise ValueError('session must be a single directory name')
    root=config.artifacts_root/name
    if root.exists() and any(root.iterdir()):
        raise ValueError('Session directory is not empty')
    if not config.lifecycle_ui:
        raise ValueError('Independent protection requires lifecycle_ui')
    root.mkdir(parents=True,exist_ok=True)
    targets=_targets or {}
    pc=config.protection
    agent_config=replace(config,artifacts_root=root)
    ctx=mp.get_context('spawn')
    observer_q=ctx.Queue(maxsize=32)
    events=ctx.Queue(maxsize=128)
    stop=ctx.Event()
    release=ctx.Event()  # Supervisor asks the current protector to hand back.
    observer=worker=None
    standby=None  # Warm protector: {'process','queue','release_mouse','active','activation','ready'}
    rejoin=None   # Observe-only Agent bundle; see spawn_rejoin.
    input_active=ctx.Value('b',False,lock=False)
    monitor=ProtectionMonitor(pc,time.monotonic(),
                              matching_seconds=MenuConfig.load(config.menu).table_timeout_seconds,
                              require_agent_recognition=True)
    ledger=RecoveryLedger(pc)
    mode='agent';attempt=0;worker_progress=time.monotonic();retry_at=0
    in_input=False;release_mouse=False;parent_stopping=False
    blocked_modal=None;modal_attempts={};last_protection_error=None
    owner_since=time.monotonic();agent_respawn_at=None;agent_spawns=0
    releasing_since=None;next_handback_at=0.0;rejoin_at=None;rejoin_count=0
    standby_count=0;standby_retry_at=0.0;last_reason=None
    flags={'interrupt':False}
    def log(event,**fields):
        row=dict(event=event,monotonic=time.monotonic(),wall_time=time.time(),owner=mode,**fields)
        line=json.dumps(row,ensure_ascii=False,default=str)
        with (root/'protection.jsonl').open('a',encoding='utf-8') as f:f.write(line+'\n')
        print(line,flush=True)
    queue=NotificationQueue(notifier or FeishuNotifier(config.notify),root/'protection-notifications',
        capacity=config.notify.queue_capacity,logger=log,max_attempts=config.notify.max_attempts,
        retry_seconds=config.notify.retry_seconds)
    def notify(text):
        queue.submit('notify_text','[majsoul-protection] '+text)
    def spawn_observer():
        nonlocal observer_q
        observer_q=ctx.Queue(maxsize=32)
        target=targets.get('observer',_observer)
        args=(config,hwnd,observer_q,stop)
        if target is _observer:args+= (root/'observer',)
        p=ctx.Process(target=target,args=args,name='majsoul-ui-observer',daemon=True)
        p.start();return p
    def spawn_standby(directory_name):
        bundle=dict(queue=ctx.Queue(maxsize=128),release_mouse=ctx.Value('b',False,lock=False),
                    active=ctx.Value('b',False,lock=False),activation=ctx.Event(),ready=ctx.Event())
        bundle['process']=ctx.Process(target=targets.get('standby',_standby),
            args=(config,hwnd,root/directory_name,bundle['queue'],bundle['release_mouse'],bundle['active'],
                  bundle['activation'],bundle['ready'],targets.get('client',_protect),release),
            name='majsoul-protection-standby',daemon=True)
        bundle['process'].start()
        return bundle
    def spawn_worker():
        nonlocal worker_progress,attempt,events,input_active,owner_since,agent_spawns
        worker_progress=owner_since=time.monotonic();attempt+=1
        # Each input owner has a distinct channel. A terminated child cannot
        # corrupt or inject late events into its replacement's channel.
        events=ctx.Queue(maxsize=128)
        input_active=ctx.Value('b',False,lock=False)
        kwargs={}
        if mode=='agent':
            agent_spawns+=1
            name='agent' if agent_spawns==1 else f'agent-{agent_spawns:03d}'
            target,args=_agent,(agent_config,hwnd,name,skip_menu,once,events,input_active)
        else:
            target,args=_protect,(config,hwnd,root/f'client-{attempt:03d}',events,release_mouse,input_active)
            kwargs={'release':release}
        target=targets.get(mode,target)
        p=ctx.Process(target=target,args=args,kwargs=kwargs,name='majsoul-'+mode,daemon=True)
        p.start()
        if mode=='agent':
            # The child cannot send a heartbeat before it exists. Count its
            # liveness from launch, retaining the independent operation and
            # pre-match deadlines established during observer/standby setup.
            monitor.event('input_owner_started',{},time.monotonic())
        log('input_owner_started',pid=p.pid,attempt=attempt)
        return p
    def activate_standby():
        nonlocal worker,standby,events,input_active,worker_progress,owner_since,attempt
        # Warm standby is granted input ONLY after the old process's exit
        # acknowledgment and terminal record.
        standby['release_mouse'].value=release_mouse
        worker,events,input_active=standby['process'],standby['queue'],standby['active']
        worker_progress=owner_since=time.monotonic();attempt+=1
        standby['activation'].set();standby=None
        log('input_owner_started',pid=worker.pid,attempt=attempt,prewarmed=True)
    def ensure_standby(now):
        nonlocal standby,standby_retry_at,standby_count
        if standby is not None and standby['process'].is_alive() or now<standby_retry_at:
            return
        if standby is not None:
            # A dead warm protector is replaced; it is not a reason to stop a
            # healthy Agent. Until it is warm, a takeover starts a cold one.
            log('protection_standby_exited',exit_code=standby['process'].exitcode)
            stop_input_owner(standby['process'],pc.stop_grace_seconds)
        standby_count+=1
        standby=spawn_standby(f'client-standby-{standby_count:03d}')
        standby_retry_at=now+max(pc.worker_retry_seconds,1.0)
        log('protection_standby_started',pid=standby['process'].pid)
    def spawn_rejoin(activate):
        nonlocal rejoin_count
        rejoin_count+=1
        now=time.monotonic()
        bundle=dict(queue=ctx.Queue(maxsize=128),active=ctx.Value('b',False,lock=False),
            granted=ctx.Value('b',False,lock=False),activation=ctx.Event(),ready=ctx.Event(),
            stable=ctx.Value('i',ledger.stable_frames(),lock=False),session=f'rejoin-{rejoin_count:03d}',
            spawned_at=now,activated_at=None,progress=now,loaded=False,ready_at=None,failed=None)
        bundle['process']=ctx.Process(target=targets.get('rejoin',_rejoin),
            args=(agent_config,hwnd,bundle['session'],bundle['queue'],bundle['active'],bundle['granted'],
                  bundle['activation'],bundle['ready'],bundle['stable']),name='majsoul-rejoin',daemon=True)
        bundle['process'].start()
        log('rejoin_spawned',pid=bundle['process'].pid,session=bundle['session'],warm=not activate)
        if activate:activate_rejoin(bundle)
        return bundle
    def activate_rejoin(bundle):
        if bundle['activated_at'] is not None:return
        bundle['stable'].value=ledger.stable_frames()
        bundle['activated_at']=bundle['progress']=time.monotonic()
        bundle['ready_at']=None
        bundle['activation'].set()
        log('rejoin_activated',session=bundle['session'],stable_frames=bundle['stable'].value)
    def retire_rejoin(reason):
        nonlocal rejoin,rejoin_at,releasing_since
        if rejoin is None:return
        stop_input_owner(rejoin['process'],pc.stop_grace_seconds)
        log('rejoin_retired',session=rejoin['session'],reason=reason)
        rejoin=None
        if releasing_since is not None:
            release.clear();releasing_since=None
        rejoin_at=time.monotonic()+ledger.delay()
    def drain_rejoin(now):
        if rejoin is None:return
        for item in _drain(rejoin['queue'],128):
            event,fields=item.get('event'),item.get('fields',{})
            at=min(now,item.get('emitted_at',now))
            rejoin['progress']=max(rejoin['progress'],at)
            if event=='worker_ready':
                rejoin['loaded']=True
            elif event=='agent_ready_for_handback' and rejoin['activated_at'] is not None:
                rejoin['ready_at']=at
            elif event in ('worker_exit','worker_exception','agent_stopped','fatal_error'):
                rejoin['failed']=event
            if event not in ('heartbeat','agent_ready_for_handback'):
                log('rejoin_event',session=rejoin['session'],source_event=event,details=fields)
    def check_rejoin(now):
        if rejoin is None:return
        if not rejoin['process'].is_alive():
            return retire_rejoin('process_exited')
        if rejoin['failed']:
            return retire_rejoin(rejoin['failed'])
        if not rejoin['loaded']:
            if now-rejoin['spawned_at']>pc.rejoin_startup_seconds:
                retire_rejoin('startup_timeout')
        elif rejoin['activated_at'] is not None and now-rejoin['progress']>pc.rejoin_stall_seconds:
            retire_rejoin('stalled')
    def rejoin_healthy(now):
        return (rejoin is not None and rejoin['process'].is_alive() and not rejoin['failed'] and
                rejoin['activated_at'] is not None and rejoin['loaded'] and
                now-rejoin['progress']<=pc.heartbeat_seconds)
    def latch(reason_text):
        nonlocal releasing_since
        if ledger.latched is None:
            ledger.latched=reason_text
            log('recovery_latched',reason=reason_text)
        release.clear();releasing_since=None
        retire_rejoin('latched')
    def hold_or_latch(reason,klass,now):
        """Client owns input; decide whether a later handback is permitted."""
        nonlocal next_handback_at,rejoin_at
        ledger.client_used=True
        exhausted=('fatal:'+str(reason.get('kind')) if klass=='fatal' else
                   'prematch_retries_exhausted' if klass=='prematch' else ledger.exhausted())
        if exhausted:
            latch(exhausted)
            return False
        next_handback_at=now+ledger.delay()
        if rejoin is not None:activate_rejoin(rejoin)
        else:rejoin_at=now
        return True
    def takeover(reason,now):
        nonlocal worker,mode,standby,attempt,retry_at,release_mouse,agent_respawn_at,last_reason
        klass=recovery_class(reason,seen_match=monitor.seen_match,lobby_verified=monitor.lobby_verified)
        ledger.record_trip(reason,now,klass)
        last_reason=reason
        log('protection_requested',reason=reason,last_sample=monitor.last_sample,recovery_class=klass)
        # Fencing is unconditional: no model return or late click can race the
        # client toggle operation after this join.
        release_mouse=in_input
        stop_input_owner(worker,pc.stop_grace_seconds)
        release_mouse=release_mouse or bool(input_active.value)
        log('input_owner_exited',pid=worker.pid,exit_code=worker.exitcode)
        record={'reason':reason,'pid':worker.pid,'exit_code':worker.exitcode,'external_fence':True,
                'acknowledged_at':time.monotonic(),'recovery_class':klass}
        (root/'agent-termination.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        with (root/'agent-terminations.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
        worker=None
        if klass=='prematch' and pc.recovery_enabled and ledger.prematch_restarts<pc.prematch_retries:
            # No match exists yet; a replacement Agent restarts from the lobby.
            ledger.prematch_restarts+=1
            agent_respawn_at=now+min(pc.rejoin_delay_seconds*2**(ledger.prematch_restarts-1),
                                     pc.rejoin_backoff_max_seconds)
            log('agent_restart_scheduled',attempt=ledger.prematch_restarts,at=agent_respawn_at)
            notify('Agent failed before entering the hand, currently still in the lobby; will restart the Agent (attempt '
                   +str(ledger.prematch_restarts)+'), reason: '+json.dumps(reason,ensure_ascii=False))
            return
        mode='client';retry_at=now
        if monitor.lobby_verified:
            # The old owner failed in the lobby. Fence it and record failure
            # without granting unnecessary input.
            pass
        elif standby is not None and standby['process'].is_alive():
            activate_standby()
        elif standby is not None:
            stop_input_owner(standby['process'],pc.stop_grace_seconds)
            standby=None;attempt=max(attempt,2)
        if hold_or_latch(reason,klass,now):
            notify('Agent paused, temporary client protection (takeover attempt '+str(len(ledger.trips))+'), reason: '
                   +json.dumps(reason,ensure_ascii=False)
                   +'. The observational Agent is resynchronizing, and will hand back after verifying on the opponent\'s turn that the auto option is off.')
        else:
            notify('Agent has stopped and switched to continuous client protection, reason: '+json.dumps(reason,ensure_ascii=False)
                   +'; will no longer hand back: '+str(ledger.latched)+'. This match will not start a new round; verify the auto option hand by hand until the lobby.')
    def grant(now):
        nonlocal worker,events,input_active,mode,rejoin,owner_since,attempt,worker_progress
        nonlocal releasing_since,blocked_modal,modal_attempts,rejoin_at
        # The protector has verified every handback option OFF and is idle.
        stop_input_owner(worker,pc.stop_grace_seconds)
        log('input_owner_exited',pid=worker.pid,exit_code=worker.exitcode,reason='handback')
        release.clear();releasing_since=None
        bundle,rejoin=rejoin,None
        worker,events,input_active=bundle['process'],bundle['queue'],bundle['active']
        mode='agent';attempt+=1
        worker_progress=owner_since=time.monotonic()
        monitor.rearm(owner_since)
        entry=ledger.record_handback(owner_since,monitor.hand_epoch)
        bundle['granted'].value=True
        blocked_modal=None;modal_attempts={}
        # Let the new owner settle before loading another warm model copy.
        rejoin_at=owner_since+pc.rejoin_stall_seconds
        log('input_owner_started',pid=worker.pid,attempt=attempt,handback=True,session=bundle['session'])
        log('handback_complete',**entry)
        notify('Agent has taken over again on the opponent\'s turn (attempt '+str(len(ledger.handbacks))+', about '
               +str(entry['recovery_seconds'])+' seconds since pause), the auto option has been verified off.')
    completed=False
    import signal
    def interrupt(*_):
        flags['interrupt']=True
        monitor.trip({'kind':'operator_interrupt'})
    with WindowLease(config.artifacts_root/'control-leases',hwnd):
        previous_sigint=signal.signal(signal.SIGINT,interrupt)
        try:
            observer=spawn_observer()
            # Observer must be healthy before any menu/game input is permitted.
            observer_ready=False
            while not observer_ready:
                try:
                    item=observer_q.get(timeout=.1)
                    sample=item['sample']
                    if sample.get('error'):raise RuntimeError(sample['error'])
                    monitor.observe(sample,time.monotonic())
                    observer_ready=True
                except Empty:
                    if not observer.is_alive() or time.monotonic()-monitor.started>pc.startup_seconds:
                        raise RuntimeError('Independent UI observer did not become ready; no input allowed')
            if monitor.reason or (root/'STOP').exists():
                monitor.trip({'kind':'operator_stop_before_start'})
                mode='client';ledger.client_used=True
                latch('operator_stop_before_start')
                notify('a stop request or obstruction was found before startup, Agent matching was not started; switching to only handling protection for the current match: '
                       +json.dumps(monitor.reason,ensure_ascii=False))
            standby=spawn_standby('client-002')
            warm_started=time.monotonic();warm_generation=0
            while not standby['ready'].wait(.1):
                now=time.monotonic()
                for item in _drain(observer_q,32):
                    monitor.observe(item['sample'],now)
                monitor.check(now)
                if monitor.lobby_verified and monitor.seen_match:
                    completed=True;mode='client'
                    log('protected_match_complete',agent_acceptance=False,initialization_incomplete=True)
                    return 1
                if (root/'STOP').exists():monitor.trip({'kind':'operator_stop_before_start'})
                limit=(pc.heartbeat_seconds if monitor.agent_recognized else pc.startup_seconds)
                if not standby['process'].is_alive() or now-warm_started>=limit:
                    notify('protection-side warm-up not complete, Agent input is still forbidden; restarting the protection side, auto option ON not yet confirmed.')
                    stop_input_owner(standby['process'],pc.stop_grace_seconds)
                    warm_generation+=1;warm_started=now
                    standby=spawn_standby(f'client-warm-{warm_generation:03d}')
                    # Nothing is lost by retrying in the lobby; a visible match
                    # or repeated failure still protects without the Agent.
                    if monitor.seen_match or warm_generation>pc.prematch_retries:
                        monitor.trip({'kind':'protection_initialization_failed'})
                if now-monitor.last_observation>pc.observation_seconds:
                    stop_input_owner(observer,pc.stop_grace_seconds)
                    observer=spawn_observer();monitor.last_observation=now
            log('protection_standby_ready',pid=standby['process'].pid)
            # Readiness can arrive after the last timed wait. Recheck the
            # original operation budget before ever granting Agent input.
            now=time.monotonic()
            for item in _drain(observer_q,32):
                monitor.observe(item['sample'],now)
            monitor.check(now)
            if mode=='agent' and monitor.reason:
                reason=monitor.reason
                klass=recovery_class(reason,seen_match=monitor.seen_match,lobby_verified=monitor.lobby_verified)
                ledger.record_trip(reason,now,klass);last_reason=reason
                if klass=='prematch' and pc.recovery_enabled and ledger.prematch_restarts<pc.prematch_retries:
                    ledger.prematch_restarts+=1
                    log('agent_restart_scheduled',attempt=ledger.prematch_restarts,at=now,reason=reason)
                    monitor.restart_prematch(now)
                else:
                    mode='client'
                    hold_or_latch(reason,klass,now)
            if mode=='agent':
                worker=spawn_worker()
            else:
                log('protection_requested',reason=monitor.reason,agent_started=False)
                notify('Agent not yet started, directly protecting the current match: '+json.dumps(monitor.reason,ensure_ascii=False))
                attempt=max(attempt,1)
                activate_standby()
            last_tick=time.monotonic()
            handback_released=False
            while True:
                now=time.monotonic()
                ledger.account(mode,now-last_tick,bool((monitor.last_sample or {}).get('table')))
                last_tick=now
                for item in _drain(events,128):
                    event,fields=item.get('event'),item.get('fields',{})
                    at=min(now,item.get('emitted_at',now))
                    worker_progress=max(worker_progress,at)
                    if event=='heartbeat':
                        if fields.get('stage')=='input_begin':in_input=True
                        elif fields.get('stage')=='input_end':in_input=False
                        if fields.get('stage') in ('input_begin','input_end'):
                            # Recorded so a run's monitor can be replayed offline.
                            log('agent_input',stage=fields['stage'],at=at,details=fields)
                    if mode=='agent':
                        # A granted rejoin Agent's observe-only history cannot
                        # trip or renew the fresh budget that starts at handback.
                        if at>=owner_since:monitor.event(event,fields,at)
                    elif event=='modal_confirm_clicked':
                        modal=fields['name'];modal_attempts[modal]=modal_attempts.get(modal,0)+1
                        if modal_attempts[modal]>=3:
                            blocked_modal=modal
                            notify('after three confirmations of the recognized popup, still needs checking: '+modal+'; stopping repeated clicks, keeping watch until the screen recovers.')
                    elif event=='handback_released':
                        handback_released=True
                    elif event=='handback_rejected':
                        release.clear();releasing_since=None;next_handback_at=now+ledger.delay()
                    elif event=='worker_exception':
                        error=fields.get('error','unknown protector failure')
                        if error!=last_protection_error:
                            notify('client protection not yet complete: '+error+'; will not treat an unknown switch as on, monitoring continues to retry.')
                            last_protection_error=error
                    if event!='heartbeat':log('worker_event',source_event=event,details=fields)
                drain_rejoin(now)
                for item in _drain(observer_q,32):
                    sample=item['sample']
                    monitor.observe(sample,now)
                    if mode=='client' and sample.get('error'):
                        if not parent_stopping:
                            notify('protection obstructed externally: '+sample['error']+'; return to lobby not confirmed, monitoring continues to retry.')
                            parent_stopping=True
                    elif not sample.get('error'):parent_stopping=False
                if (root/'STOP').exists() or flags['interrupt']:
                    if mode=='agent' and worker is not None:
                        monitor.trip({'kind':'operator_interrupt' if flags['interrupt'] else 'operator_stop'})
                    elif ledger.latched is None or agent_respawn_at is not None:
                        agent_respawn_at=None;mode='client';ledger.client_used=True
                        latch('operator_interrupt' if flags['interrupt'] else 'operator_stop')
                if mode=='agent' and worker is not None and not worker.is_alive():
                    monitor.trip({'kind':'agent_process_exited','exit_code':worker.exitcode})
                if mode=='agent':ensure_standby(now)
                check_rejoin(now)
                reason=monitor.check(now)
                if mode=='agent' and worker is not None and reason:
                    takeover(reason,now)
                if mode=='agent' and worker is None and agent_respawn_at is not None and now>=agent_respawn_at:
                    agent_respawn_at=None
                    monitor.restart_prematch(now)
                    worker=spawn_worker()
                if (mode=='agent' and worker is not None and pc.warm_rejoin and ledger.latched is None and
                        pc.recovery_enabled and rejoin is None and monitor.agent_recognized and
                        (rejoin_at is None or now>=rejoin_at)):
                    rejoin=spawn_rejoin(activate=False)
                if monitor.lobby_verified and (monitor.seen_match or mode=='client'):
                    completed=True
                    log('protected_match_complete',agent_acceptance=mode=='agent' and monitor.reason is None
                        and monitor.agent_recognized and not ledger.client_used,
                        recovery=ledger.report())
                    break
                if mode=='client':
                    if handback_released:
                        handback_released=False
                        if rejoin_healthy(now) and ledger.latched is None:
                            grant(now)
                        else:
                            # Options are OFF but no healthy Agent can take
                            # input: a fresh protector re-enables them now.
                            log('handback_aborted',reason='rejoin_unavailable_or_latched')
                            release.clear();releasing_since=None
                            stop_input_owner(worker,pc.stop_grace_seconds)
                            worker=None;retry_at=now
                if mode=='client':
                    if blocked_modal and (monitor.last_sample or {}).get('modal')!=blocked_modal:
                        blocked_modal=None;modal_attempts={}
                    stale=now-worker_progress>pc.heartbeat_seconds
                    if worker is not None and (not worker.is_alive() or stale or blocked_modal):
                        log('protector_restarting',reason='heartbeat_expired' if stale else 'worker_exited')
                        stop_input_owner(worker,pc.stop_grace_seconds)
                        release_mouse=release_mouse or bool(input_active.value)
                        worker=None;retry_at=now+pc.worker_retry_seconds
                        if releasing_since is not None:
                            release.clear();releasing_since=None
                    if worker is None and now>=retry_at and not blocked_modal:
                        worker=spawn_worker()
                    if ledger.latched is None:
                        if rejoin is None and rejoin_at is not None and now>=rejoin_at:
                            rejoin=spawn_rejoin(activate=True)
                        elif rejoin is not None:
                            activate_rejoin(rejoin)
                        sample=monitor.last_sample or {}
                        if (releasing_since is None and rejoin is not None and rejoin['ready_at'] is not None
                                and now-rejoin['ready_at']<=pc.handback_ready_max_age and rejoin_healthy(now)
                                and worker is not None and worker.is_alive() and not blocked_modal
                                and now>=next_handback_at and ledger.hand_allows(monitor.hand_epoch)
                                and sample.get('table') and sample.get('waiting') and not sample.get('modal')
                                and not sample.get('confirm')
                                and now-monitor.last_observation<=pc.observation_seconds):
                            release.set();releasing_since=now
                            log('handback_requested',session=rejoin['session'],hand=monitor.hand_epoch)
                        if releasing_since is not None and now-releasing_since>pc.handback_release_seconds:
                            release.clear();releasing_since=None;next_handback_at=now+ledger.delay()
                            log('handback_timeout')
                if now-monitor.last_observation>pc.observation_seconds:
                    stop_input_owner(observer,pc.stop_grace_seconds)
                    observer=spawn_observer()
                    monitor.last_observation=now
                    log('observer_restarted')
                try:time.sleep(pc.poll_seconds)
                except KeyboardInterrupt:
                    interrupt()
                    log('stop_deferred_until_lobby')
        finally:
            stop.set()
            release.clear()
            stop_input_owner(worker,pc.stop_grace_seconds)
            stop_input_owner(observer,pc.stop_grace_seconds)
            stop_input_owner(standby['process'] if standby else None,pc.stop_grace_seconds)
            stop_input_owner(rejoin['process'] if rejoin else None,pc.stop_grace_seconds)
            result=dict(returned_lobby=completed,
                agent_acceptance=completed and mode=='agent' and not ledger.client_used and monitor.seen_match
                    and monitor.reason is None and monitor.agent_recognized,
                mixed_control=ledger.client_used or mode=='client',
                agent_in_control_at_end=completed and mode=='agent',
                reason=last_reason if last_reason is not None else monitor.reason,
                second_match_started=False,recovery=ledger.report())
            (root/'protection-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
            queue.close(config.notify.shutdown_seconds)
            signal.signal(signal.SIGINT,previous_sigint)
    return 0 if result['agent_acceptance'] else 1
