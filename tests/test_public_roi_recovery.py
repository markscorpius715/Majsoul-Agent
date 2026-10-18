from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from majsoul.live.collector import CollectConfig
from majsoul.state import MahjongStateTracker, StateValidator, TrackingPhase, GameEventType
from majsoul.state.decision import decision_observation
from majsoul.state.meld_transition import prove_pon_transition
from test_live_decisions import SlowPolicy, make_loop, process, close
from test_live_single_frame import saved_observations, ROOT
from test_live_loop import visible_reading
from test_state_tracking import snapshot
from tools.live_test_supervisor import (AgentRecovery, AcceptanceGuard, AcceptanceStopped,
                                       finish_with_client_takeover)

LATEST = ROOT / 'artifacts/live/acceptance-20261010/agent-recheck-234808/observations.jsonl'


@pytest.mark.parametrize('region', ['river_top', 'melds_top', 'kita_top', 'kita_bottom'])
def test_noncritical_roi_is_unknown_and_does_not_pause_or_block_first_frame(tmp_path, region):
    policy = SlowPolicy()
    loop, _, clock = make_loop(tmp_path, policy)
    read = loop.reader.read_with_visible
    def fail(*args, **kwargs):
        obs, visible = read(*args, **kwargs)
        return replace(obs, issues=(region + ': animation',)), visible
    loop.reader.read_with_visible = fail
    try:
        process(loop)
        assert policy.entered.wait(1)
        pending = loop._pending_decision
        assert pending['snapshot'].decision_ready
        assert region in pending['snapshot'].unavailable_regions
        assert not pending['snapshot'].history_complete
        if region.startswith('kita'):
            assert pending['observation'].player(region[5:]).kita_count is None
        clock.sleep(loop.config.persistent_issue_seconds + 1)
        process(loop)
        assert not loop._paused and loop._issue_since is None
    finally:
        close(loop)


@pytest.mark.parametrize('issue', ['hand: unknown', 'melds_bottom: ambiguous',
                                  'action_buttons: unknown', 'round: unknown'])
def test_critical_readings_still_block_initial_decision(tmp_path, issue):
    policy = SlowPolicy()
    loop, _, _ = make_loop(tmp_path, policy)
    read = loop.reader.read_with_visible
    def fail(*args, **kwargs):
        obs, visible = read(*args, **kwargs)
        return replace(obs, issues=(issue,)), visible
    loop.reader.read_with_visible = fail
    try:
        process(loop)
        assert loop._pending_decision is None and not policy.calls
    finally:
        close(loop)


@pytest.mark.parametrize('source,frame,regions', [
    (LATEST, 107, {'river_left', 'melds_left'}),
    (ROOT/'artifacts/live/acceptance-20261010/agent-recheck-234538/observations.jsonl',
     66, {'kita_left'})])
def test_latest_public_animation_frame_can_drive_model_without_becoming_event_truth(source, frame, regions):
    samples = saved_observations(source)
    obs = samples[frame]
    tracker = MahjongStateTracker(StateValidator())
    assert tracker.validator.validate_observation(obs)
    projected = decision_observation(obs)
    state = tracker.decision_snapshot(projected)
    assert state.decision_ready
    assert regions.issubset(state.unavailable_regions)
    assert tracker.state is None
    assert tracker.validator.validate_observation(projected, allow_ui_occlusion=True)
    assert not tracker.validator.validate_observation(projected, allow_public_uncertainty=True)
    assert projected.action_buttons == obs.action_buttons
    assert projected.hand == obs.hand and projected.countdown == obs.countdown


@pytest.mark.parametrize('pair', [(69, 78), (84, 93)])
def test_latest_pon_transitions_no_longer_report_false_wall_draw_or_meld_conflicts(pair):
    samples = saved_observations(LATEST)
    before, after = (samples[i] for i in pair)
    tracker = MahjongStateTracker(StateValidator(), stable_frames=1)
    tracker.update(before)
    update = tracker.update(after)
    assert not update.validation_issues
    assert not any(e.type == GameEventType.RESYNC for e in update.events)
    pon = next(e for e in update.events if e.type == GameEventType.PON)
    assert pon.details['source_discard_unobserved'] is True
    assert pon.details['source_slot'] is None
    assert update.state.phase == TrackingPhase.TRACKING
    assert not update.state.history_complete


def test_pon_missing_source_only_relaxes_exact_two_tile_consumption():
    before = snapshot(hand=('1p', '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '1s', '2s', '3s', '1z'))
    after = snapshot(timestamp=2, frame=2, hand=tuple(t.tile for t in before.hand)[2:],
                     melds=((('1p', '1p', '1p'),), (), ()))
    validator = StateValidator()
    assert prove_pon_transition(before, after, 'bottom')
    assert not validator.validate_transition(before, after)
    wrong = replace(after, hand=(replace(after.hand[0], tile='9s'), *after.hand[1:]))
    assert prove_pon_transition(before, wrong, 'bottom') is None
    assert validator.validate_transition(before, wrong)


def test_runtime_reset_cancels_pending_and_clears_hand_state_preserving_layout(tmp_path):
    policy = SlowPolicy()
    loop, _, _ = make_loop(tmp_path, policy)
    try:
        process(loop)
        assert policy.entered.wait(1)
        pending = loop._pending_decision
        slots = loop._active_slots
        loop._paused = True
        loop._issue_since = 1
        loop._verified_operation = {'stale': True}
        loop.reset_runtime('test settlement')
        assert pending['future'].cancelled() and policy.cancelled.wait(1)
        assert loop._pending_decision is None and loop.tracker.state is None
        assert loop._active_slots == slots
        assert loop._verified_operation is None and not loop._paused
        assert loop._last_request_key is None and loop._issue_since is None
    finally:
        close(loop)


def test_result_runtime_reset_occurs_once_across_confirm_pages(tmp_path):
    loop, _, _ = make_loop(tmp_path, SlowPolicy())
    calls = []
    fake = SimpleNamespace(lifecycle=SimpleNamespace(round_index=2),
                           observe=lambda *args: (True, ('confirm_appeared',)))
    loop.session_ui = fake
    loop.reset_runtime = lambda reason, **kwargs: calls.append(reason)
    frame, _ = loop._capture()
    try:
        loop._session_frame(frame)
        loop._session_frame(frame)
        assert calls == ['hand settlement']
        fake.lifecycle.round_index = 3
        loop._session_frame(frame)
        assert calls == ['hand settlement', 'hand settlement']
    finally:
        close(loop)


def test_exception_from_observer_still_persists_stop_reason(tmp_path):
    loop, _, _ = make_loop(tmp_path, SlowPolicy())
    loop._wait_for_table = lambda: None
    loop._lock_layout = lambda: None
    def stop():
        raise AcceptanceStopped('test_action_stalled')
    loop._main_loop = stop
    with pytest.raises(AcceptanceStopped):
        loop.run()
    records = [json.loads(line) for line in (loop.artifacts/'events.jsonl').read_text().splitlines()]
    terminal = next(r for r in records if r['event'] == 'agent_stopped')
    assert terminal['reason'] == 'AcceptanceStopped: test_action_stalled'
    assert terminal['exit_code'] == 1


def test_guard_does_not_carry_a_turn_stall_into_settlement_or_next_hand():
    guard = AcceptanceGuard()
    guard.observe('decision_gate', {'countdown': {'status': 'complete'}}, 1)
    guard.observe('confirm_appeared', {}, 7)
    guard.check(100)
    assert guard.failure is None and guard.actionable_since is None


def recovery_loop(tmp_path, monkeypatch, *, bad_after_off=False):
    loop, window, clock = make_loop(tmp_path, SlowPolicy())
    loop.session_ui = SimpleNamespace(lifecycle=SimpleNamespace(round_index=1, complete=False),
                                     classify=lambda frame: (False, None, True))
    loop._session_frame = lambda frame: False
    read = loop.reader.read_with_visible
    controls = []
    def reading(*args, **kwargs):
        obs, _ = read(*args, **kwargs)
        obs = replace(obs, hand=obs.hand[:-1], drawn_index=None, action_buttons=(),
                      issues=('hand: unknown',) if bad_after_off and controls and controls[-1] == 'off' else ())
        return obs, visible_reading(tuple(t.tile for t in obs.hand))
    loop.reader.read_with_visible = reading
    def enable(*args, **kwargs):
        controls.append('on')
        return {'verified': True, 'attempts': [{'missing': []}], 'snapshots': []}
    class Panel:
        def __init__(self, window, config, **kwargs):
            assert config.toggle_targets[-1] == 'auto_discard'
            assert 'auto_sort' not in config.toggle_targets
        def ensure(self, *, enabled):
            assert enabled is False
            controls.append('off')
            return []
    monkeypatch.setattr('tools.live_test_supervisor.enable_client_takeover', enable)
    monkeypatch.setattr('tools.live_test_supervisor.TogglePanel', Panel)
    config = CollectConfig.load(ROOT/'configs/collect.steam.yaml')
    recovery = AgentRecovery(config, tmp_path/'recovery', maximum_same_failures=2)
    return loop, window, clock, recovery, controls


def test_failure_protects_before_reset_and_clean_table_restores_agent(tmp_path, monkeypatch):
    loop, window, _, recovery, controls = recovery_loop(tmp_path, monkeypatch)
    order = []
    original = loop.reset_runtime
    def reset(*args, **kwargs):
        order.append(list(controls))
        original(*args, **kwargs)
    loop.reset_runtime = reset
    try:
        recovery.start(loop, {'event': 'test_action_stalled'})
        assert recovery.active and controls == ['on'] and order == [['on']]
        frame, index = loop._capture()
        assert recovery.step(loop, frame, index) is False
        assert controls == ['on', 'off'] and not recovery.active
        assert window.clicks == [] and loop._pending_decision is None
        records = [json.loads(line) for line in (loop.artifacts/'events.jsonl').read_text().splitlines()]
        assert [r['owner'] for r in records if r['event'] == 'control_transferred'] == ['client', 'agent']
    finally:
        close(loop)


def test_failed_fresh_handoff_restores_protection_and_never_executes(tmp_path, monkeypatch):
    loop, window, clock, recovery, controls = recovery_loop(tmp_path, monkeypatch, bad_after_off=True)
    try:
        recovery.start(loop, {'event': 'decision_failed', 'details': {'error': 'transport'}})
        frame, index = loop._capture()
        recovery.step(loop, frame, index)
        assert controls == ['on', 'off', 'on'] and recovery.active
        assert loop._pending_decision is None and window.clicks == []
        frame, index = loop._capture()
        recovery.step(loop, frame, index)
        assert controls == ['on', 'off', 'on']  # Bounded retry cadence.
    finally:
        close(loop)


def test_new_hand_reenables_client_before_recovery_and_repeated_failure_stops(tmp_path, monkeypatch):
    loop, _, _, recovery, controls = recovery_loop(tmp_path, monkeypatch)
    loop._notify_session = lambda *args, **kwargs: True
    try:
        failure = {'event': 'test_action_stalled'}
        recovery.start(loop, failure)
        loop.session_ui.lifecycle.round_index = 2
        recovery.retry_at = 0
        frame, index = loop._capture()
        recovery.step(loop, frame, index)
        assert controls == ['on', 'on'] and recovery.active
        assert recovery.setup_round == 2
        with pytest.raises(AcceptanceStopped, match='repeated_same_failure'):
            recovery.start(loop, failure)
        assert recovery.active and controls[-1] == 'on'
    finally:
        close(loop)


def test_settlement_rebuilds_owned_policy_worker_without_reusing_old_http_client(tmp_path):
    old = SlowPolicy()
    loop, _, _ = make_loop(tmp_path, old)
    fresh = SlowPolicy()
    loop._policy_factory = lambda: fresh
    try:
        process(loop)
        assert old.entered.wait(1)
        worker, future = loop.decision_worker, loop._pending_decision['future']
        loop.reset_runtime('settlement')
        assert loop.policy is fresh and loop.decision_worker is not worker
        assert future.cancelled() and old.cancelled.wait(1)
        worker._thread.join(1)
        assert not worker._thread.is_alive() and old.closed
        process(loop)
        assert fresh.entered.wait(1)
    finally:
        close(loop)


def test_stop_diagnostics_use_protective_collector_notification_queue(tmp_path, monkeypatch):
    sent = []
    class Collector:
        def __init__(self, window, config, live, **kwargs):
            self.config = config
            self.sleep = kwargs['sleep']
        def _capture(self):
            return SimpleNamespace(image=None)
        def _resolve_roi(self, frame):
            pass
        def _menu_score(self, frame):
            return 1
        def _save_frame(self, *args):
            pass
        def _notify(self, message, image=None, **kwargs):
            sent.append(message)
        def run(self):
            self._enter_match()
            self._play_match()
            return 0
    monkeypatch.setattr('tools.live_test_supervisor.MatchCollector', Collector)
    report = finish_with_client_takeover(object(), CollectConfig.load(ROOT/'configs/collect.steam.yaml'),
        object(), tmp_path/'through-match', sleep=lambda _: None,
        failure_report={'event': 'repeated_same_failure', 'reason': 'river ROI'})
    assert report['returned_lobby'] is True
    assert len(sent) == 1 and 'repeated_same_failure' in sent[0] and 'river ROI' in sent[0]
    assert 'not model acceptance' in sent[0]
