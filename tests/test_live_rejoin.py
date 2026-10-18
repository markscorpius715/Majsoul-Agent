"""Observe-only rejoin Agent: no input before the grant, readiness from opponent turns."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from majsoul.live.config import LiveConfig
from majsoul.live.loop import LiveAgentLoop

from test_live_loop import (HAND13, HAND14, LIVE_CONFIG, SLOTS, Clock, FakePolicy,
                            FakeWindow, mock_reader, small_roi)


class Gate:
    def __init__(self, value=False):
        self.value = value

    def __call__(self):
        return self.value


@pytest.fixture
def make_rejoin(tmp_path):
    loops = []

    def make(*, hand=HAND13, execute=True, handback_frames=2, checkpoint=None,
             stable_seconds=1.5, session='rejoin'):
        base = LiveConfig.load(LIVE_CONFIG)
        config = replace(base, artifacts_root=tmp_path, lifecycle_ui=None,
                         notify=replace(base.notify, enabled=False),
                         protection=replace(base.protection, handback_stable_seconds=stable_seconds))
        clock = Clock()
        window = FakeWindow(clock)
        policy = FakePolicy()
        gate = Gate()
        loop = LiveAgentLoop(window, config, execute=execute, skip_menu=True, join=True,
                             session=session, sleep=clock.sleep, now=clock, roi=small_roi(),
                             reader=mock_reader(hand=hand, window=window), policy=policy,
                             input_gate=gate, handback_frames=handback_frames,
                             checkpoint=checkpoint)
        loop.artifacts.mkdir(parents=True, exist_ok=True)
        loop._last_activity = clock()
        loop._active_slots = SLOTS
        frame, index = loop._capture()
        loop._track(frame.image, small_roi(), SLOTS, frame.captured_at, index)
        loops.append(loop)
        return SimpleNamespace(loop=loop, clock=clock, window=window, policy=policy, gate=gate)

    yield make
    for loop in loops:
        loop.decision_worker.close()


def step(rig, seconds=0.6):
    rig.clock.sleep(seconds)
    frame, index = rig.loop._capture()
    return rig.loop._process_frame(frame, index)


def events(rig, name=None):
    lines = (rig.loop.artifacts / 'events.jsonl').read_text(encoding='utf-8').splitlines()
    records = [json.loads(line) for line in lines]
    return [record for record in records if name is None or record['event'] == name]


def settle(rig):
    pending = rig.loop._pending_decision
    if pending is not None:
        pending['future'].result(timeout=1)


def test_observer_on_own_turn_never_decides_clicks_or_signals(make_rejoin):
    rig = make_rejoin(hand=HAND14)
    for _ in range(6):
        step(rig)
    assert rig.window.clicks == []
    assert rig.policy.calls == []
    assert rig.loop._pending_decision is None
    gates = events(rig, 'decision_gate')
    assert gates and all('input_not_granted' in gate['blocking_reasons'] for gate in gates)
    assert gates[-1]['phase'] == 'discard'
    # Own turn is not a safe handback moment.
    assert events(rig, 'agent_ready_for_handback') == []


def test_observer_signals_after_stable_opponent_turn_frames_and_span(make_rejoin):
    rig = make_rejoin(hand=HAND13, handback_frames=3, stable_seconds=1.5)
    step(rig)
    step(rig)
    assert events(rig, 'agent_ready_for_handback') == []  # 2 frames, 0.6 s span.
    step(rig)
    assert events(rig, 'agent_ready_for_handback') == []  # 3 frames, but only 1.2 s.
    step(rig)
    ready = events(rig, 'agent_ready_for_handback')
    assert len(ready) == 1
    assert ready[0]['frames'] == 4 and ready[0]['span'] >= 1.5
    assert ready[0]['round_id'] == '1z:1:0'
    assert rig.window.clicks == [] and rig.policy.calls == []


def test_callable_handback_frames_are_read_each_frame(make_rejoin):
    need = {'frames': 50}
    rig = make_rejoin(hand=HAND13, handback_frames=lambda: need['frames'], stable_seconds=0.1)
    for _ in range(4):
        step(rig)
    assert events(rig, 'agent_ready_for_handback') == []
    need['frames'] = 2  # The supervisor relaxed its flap-tightened requirement.
    step(rig)
    assert len(events(rig, 'agent_ready_for_handback')) == 1


def test_perception_failure_restarts_the_stability_count(make_rejoin):
    rig = make_rejoin(hand=HAND13, handback_frames=3, stable_seconds=0.1)
    step(rig)
    step(rig)
    rig.loop._track = lambda *args, **kwargs: None
    step(rig)
    assert rig.loop._handback_count == 0 and rig.loop._handback_since is None
    del rig.loop._track  # Restore the real tracker method.
    step(rig)
    step(rig)
    assert events(rig, 'agent_ready_for_handback') == []
    step(rig)
    assert len(events(rig, 'agent_ready_for_handback')) == 1


def test_stale_frame_does_not_count_toward_handback(make_rejoin):
    rig = make_rejoin(hand=HAND13, handback_frames=1, stable_seconds=0)
    capture = rig.window.capture

    def stale():
        frame = capture()
        age = rig.loop.config.decision.max_decision_age + 1
        return replace(frame, captured_at=frame.captured_at - age)

    rig.window.capture = stale
    step(rig)
    assert events(rig, 'agent_ready_for_handback') == []
    rig.window.capture = capture
    step(rig)
    assert len(events(rig, 'agent_ready_for_handback')) == 1


def test_grant_resets_clocks_and_resumes_decisions(make_rejoin):
    rig = make_rejoin(hand=HAND14, execute=False)
    for _ in range(3):
        step(rig)
    assert rig.policy.calls == []
    rig.gate.value = True
    step(rig)
    settle(rig)
    granted = events(rig, 'input_granted')
    assert len(granted) == 1 and granted[0]['active_slots'] == list(SLOTS)
    assert len(rig.policy.calls) == 1
    step(rig)
    assert len(events(rig, 'decision_dry_run')) == 1
    assert rig.window.clicks == []  # Dry run: the grant enables decisions, not clicks.


def test_grant_lets_an_executing_rejoin_agent_click(make_rejoin):
    rig = make_rejoin(hand=HAND14, execute=True)
    step(rig)
    step(rig)
    assert rig.window.clicks == []
    rig.gate.value = True
    step(rig)
    settle(rig)
    step(rig)
    assert len(rig.policy.calls) == 1
    assert rig.window.clicks


def test_revoked_grant_cancels_a_pending_decision_without_clicking(make_rejoin):
    rig = make_rejoin(hand=HAND14, execute=True)
    rig.gate.value = True
    step(rig)
    settle(rig)
    assert len(rig.policy.calls) == 1
    rig.gate.value = False
    step(rig)
    assert rig.loop._pending_decision is None
    assert rig.window.clicks == []
    assert 'input_not_granted' in events(rig, 'decision_gate')[-1]['blocking_reasons']


def test_lifecycle_overlay_is_observe_only_until_granted(make_rejoin):
    rig = make_rejoin(hand=HAND13, execute=True)
    seen = []

    def observe(frame, now):
        seen.append(ui.execute)
        return True, set()

    ui = SimpleNamespace(execute=True, observe=observe,
                         lifecycle=SimpleNamespace(complete=False, round_index=1))
    rig.loop.session_ui = ui
    rig.loop._handback_count = 5
    step(rig)
    assert seen == [False]
    assert rig.loop._handback_count == 0  # A settlement overlay is never a handback moment.
    rig.gate.value = True
    step(rig)
    assert seen == [False, True]


def test_checkpoint_records_locked_layout_atomically(make_rejoin, tmp_path):
    path = tmp_path / 'agent-checkpoint.json'
    rig = make_rejoin(checkpoint=path)
    rig.loop._write_checkpoint(source='layout_locked')
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['active_slots'] == list(SLOTS)
    assert data['source'] == 'layout_locked'
    assert data['session'] == 'rejoin'
    assert [item.name for item in tmp_path.iterdir() if item.name.endswith('.tmp')] == []


def test_checkpoint_is_skipped_without_a_locked_layout(make_rejoin, tmp_path):
    path = tmp_path / 'agent-checkpoint.json'
    rig = make_rejoin(checkpoint=path)
    rig.loop._active_slots = None
    rig.loop._write_checkpoint(source='layout_locked')
    assert not path.exists()


def test_layout_hint_is_normalized_and_cleared():
    loop = object.__new__(LiveAgentLoop)
    loop.set_layout_hint(['bottom', 'top', 'left'])
    assert loop._layout_hint == SLOTS
    loop.set_layout_hint(None)
    assert loop._layout_hint is None
