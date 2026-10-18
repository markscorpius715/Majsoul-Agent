import asyncio
import json
from dataclasses import replace
from threading import Event
import time

import httpx
import pytest

from majsoul.domain import Action, Box
from majsoul.jev import FallbackPolicy, JevPolicy, JevSettings
from majsoul.live.config import LiveConfig
from majsoul.live.decisions import DecisionWorker
from majsoul.live.loop import LiveAgentLoop
from majsoul.strategy import StrategyError
from test_live_loop import (Clock, FakePolicy, FakeWindow, LIVE_CONFIG, SLOTS,
                            HAND14, mock_reader, small_roi, recognized_tile,
                            visible_reading)


class SlowPolicy(FakePolicy):
    def __init__(self, *, ignore_cancel=False, on_enter=None):
        super().__init__()
        self.entered, self.release, self.cancelled = Event(), Event(), Event()
        self.ignore_cancel = ignore_cancel
        self.on_enter = on_enter

    async def achoose(self, state, actions, *, deadline=None):
        self.calls.append((state, actions, deadline))
        self.entered.set()
        if self.on_enter:
            self.on_enter()
        try:
            while not self.release.is_set():
                await asyncio.sleep(.005)
        except asyncio.CancelledError:
            self.cancelled.set()
            if not self.ignore_cancel:
                raise
            while not self.release.is_set():
                await asyncio.sleep(.005)
        return actions[0]


def test_worker_has_one_request_and_immutable_input_and_result(state):
    policy = SlowPolicy()
    worker = DecisionWorker(policy)
    actions = [Action("pass", "pass", features={"source": "test"})]
    future = worker.submit(state, actions, deadline=time.monotonic() + 5)
    try:
        assert policy.entered.wait(1)
        state.observation.phase = "result"
        actions[0].features["source"] = "changed"
        assert worker.submit(state, actions, deadline=time.monotonic() + 5) is None
        policy.release.set()
        result = future.result(timeout=1)
        policy.last_timing["status"] = "changed"
        assert policy.calls[0][0].observation.phase == "discard"
        assert result.action.features["source"] == "test"
        assert result.timing["status"] == "ok"
    finally:
        worker.close()
        worker._thread.join(1)
    assert policy.closed


def test_cancellation_aborts_real_http_and_never_starts_fallback(state):
    entered, interrupted = Event(), Event()
    fallback_calls = []

    async def respond(request):
        entered.set()
        try:
            await asyncio.sleep(30)
        finally:
            interrupted.set()

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    policy = FallbackPolicy(lambda: JevPolicy(JevSettings("test"), client),
                            lambda: fallback_calls.append(True), timeout=30)
    worker = DecisionWorker(policy)
    future = worker.submit(state, [Action("pass", "pass")], deadline=time.monotonic()+30)
    assert entered.wait(1)
    worker.cancel(future)
    assert future.cancelled()
    assert interrupted.wait(1)
    worker.close()
    worker._thread.join(1)
    assert not worker._thread.is_alive()
    assert client.is_closed
    assert fallback_calls == []
    assert policy.last_timing["attempts"][0]["status"] == "cancelled"


def test_stop_does_not_wait_for_uncooperative_sync_policy(state):
    entered, release = Event(), Event()

    class Blocking(FakePolicy):
        def choose(self, state, actions, *, deadline=None):
            entered.set()
            assert release.wait(2)
            return actions[0]

    policy = Blocking()
    worker = DecisionWorker(policy)
    future = worker.submit(state, [Action("pass", "pass")], deadline=time.monotonic()+30)
    try:
        assert entered.wait(1)
        started = time.monotonic()
        worker.close()
        assert time.monotonic()-started < .2
        assert future.cancelled()
        assert worker.submit(state, [Action("pass", "pass")], deadline=30) is None
    finally:
        release.set()
        worker._thread.join(1)
    assert not worker._thread.is_alive()
    assert policy.closed


def make_loop(tmp_path, policy, *, execute=False, once=False):
    config = replace(LiveConfig.load(LIVE_CONFIG), artifacts_root=tmp_path, lifecycle_ui=None)
    clock = Clock()
    window = FakeWindow(clock)
    loop = LiveAgentLoop(window, config, execute=execute, skip_menu=True,
                         once=once, session="async", sleep=clock.sleep,
                         now=clock, roi=small_roi(), reader=mock_reader(), policy=policy)
    loop._active_slots = SLOTS
    loop._last_activity = clock()
    return loop, window, clock


def process(loop):
    frame, index = loop._capture()
    loop._process_frame(frame, index)


def start(loop, policy):
    process(loop)
    assert policy.entered.wait(1)


def close(loop):
    loop._cancel_decision("test complete")
    loop.decision_worker.close()
    if loop.decision_worker._thread:
        loop.decision_worker._thread.join(1)
    loop.notifications.close(.1)


def test_loop_keeps_observing_and_cancels_late_answer_after_window_changes(tmp_path):
    policy = SlowPolicy(ignore_cancel=True)
    loop, window, clock = make_loop(tmp_path, policy, execute=True)
    try:
        start(loop, policy)
        count = loop._frame_index
        for _ in range(3):
            process(loop)
        assert loop._frame_index == count+3
        assert len(policy.calls) == 1
        assert window.clicks == []
        loop.button_reader._buttons = {"riichi": Box(200, 100, 250, 140)}
        process(loop)
        assert policy.cancelled.wait(1)
        assert loop._pending_decision is None
        # Cancellation-resistant adapter keeps the slot occupied; no queue grows.
        for _ in range(3):
            process(loop)
        assert len(policy.calls) == 1
        policy.release.set()
    finally:
        close(loop)
    assert window.clicks == []
    records = [json.loads(line) for line in (loop.artifacts / "decisions.jsonl").read_text().splitlines()]
    assert records[0]["mode"] == "cancelled"
    assert records[0]["action"] is None


def test_deadline_cancels_and_does_not_renew_same_window_budget(tmp_path):
    policy = SlowPolicy()
    loop, window, clock = make_loop(tmp_path, policy)
    try:
        start(loop, policy)
        pending = loop._pending_decision
        clock.value = pending["deadline"] + .1
        process(loop)
        assert policy.cancelled.wait(1)
        assert pending["future"].cancelled()
        for _ in range(3):
            process(loop)
        assert len(policy.calls) == 1
        assert loop._decisions_made == 0
    finally:
        close(loop)
    assert window.clicks == []


def test_stop_file_interrupts_loop_while_model_is_inflight(tmp_path):
    policy = SlowPolicy()
    loop, window, clock = make_loop(tmp_path, policy)
    policy.on_enter = lambda: (loop.artifacts / "STOP").write_text("stop")
    started = time.monotonic()
    assert loop.run() == 130
    assert time.monotonic()-started < 1
    assert policy.cancelled.wait(1)
    assert policy.closed_event.wait(1)
    assert window.clicks == []


def test_failed_request_is_not_repeated_each_frame(tmp_path):
    class Failing(SlowPolicy):
        async def achoose(self, state, actions, *, deadline=None):
            self.calls.append((state, actions, deadline))
            self.entered.set()
            raise StrategyError("test failure")

    policy = Failing()
    loop, window, clock = make_loop(tmp_path, policy)
    try:
        start(loop, policy)
        pending = loop._pending_decision
        try:
            pending["future"].result(timeout=1)
        except StrategyError:
            pass
        for _ in range(4):
            process(loop)
        assert len(policy.calls) == 1
    finally:
        close(loop)
    assert '"mode": "failed"' in (loop.artifacts / "decisions.jsonl").read_text()


def test_failed_request_evidence_is_saved_before_supervisor_stops_agent(tmp_path):
    from tools.live_test_supervisor import AcceptanceStopped
    class Failing(SlowPolicy):
        async def achoose(self, state, actions, *, deadline=None):
            self.calls.append((state, actions, deadline))
            self.entered.set()
            raise StrategyError('test transport failure')
    policy = Failing()
    loop, window, clock = make_loop(tmp_path, policy)
    original_log = loop.log
    def supervised_log(event, **fields):
        original_log(event, **fields)
        if event == 'decision_failed':
            assert '"mode": "failed"' in (loop.artifacts/'decisions.jsonl').read_text()
            raise AcceptanceStopped('decision_failed')
    loop.log = supervised_log
    try:
        start(loop, policy)
        with pytest.raises(StrategyError):
            loop._pending_decision['future'].result(timeout=1)
        with pytest.raises(AcceptanceStopped):
            process(loop)
        assert window.clicks == []
    finally:
        close(loop)


def test_async_answer_executes_once_with_fresh_target_and_stable_discard_proof(tmp_path):
    from majsoul.perception.schemas import TileSequenceReading

    policy = SlowPolicy()
    loop, window, clock = make_loop(tmp_path, policy, execute=True)
    # This test pins execution timing of the first candidate, not tile choice.
    loop.config = replace(loop.config, decision=replace(loop.config.decision, shanten_guard=False))
    original = visible_reading(HAND14)
    # Use physical coordinates inside this test's actual 400x200 window.
    def placed(item, index):
        return replace(item, detection=replace(item.detection,
                       box=Box(20+index*24, 130, 38+index*24, 175)))
    hand_before = tuple(placed(item, index) for index, item in enumerate(original.hand.tiles))
    original.hand = replace(original.hand, tiles=hand_before)
    after = visible_reading(HAND14[1:])
    after.hand = replace(after.hand, tiles=tuple(
        placed(item, index) for index, item in enumerate(after.hand.tiles)))
    after.rivers["bottom"] = TileSequenceReading("river_bottom", (recognized_tile("1p", 0),))
    loop.reader.visible_reader.read = lambda *_: after if window.clicks else original
    try:
        start(loop, policy)
        policy.release.set()
        loop._pending_decision["future"].result(timeout=1)
        process(loop)
        assert window.clicks == [hand_before[0].detection.box.center]
        assert len(policy.calls) == 1
        assert loop._decisions_made == 1
        assert loop.decision_budget.started_at is None
        records = [json.loads(line) for line in (loop.artifacts / "decisions.jsonl").read_text().splitlines()]
        assert records[0]["mode"] == "verified"
        assert records[0]["action"]["tile"] == "1p"
        verification = json.loads((loop.artifacts / "frames" / "00001-verification.json").read_text())
        assert verification["status"] == "verified"
        assert verification["evidence"]["discarded_tile"] == "1p"
        assert verification["assessed_frame_index"] == verification["after"]["frame_index"]
        for key in ("before_image", "after_image"):
            from majsoul.calibration.frames import read_image
            assert read_image(loop.artifacts / verification[key]).shape == (200, 400, 3)
        assert verification["before"]["timestamp"] < verification["after"]["timestamp"]
    finally:
        close(loop)


@pytest.mark.parametrize(("stable", "issues"), [
    (False, ()), (True, ("physical_tile_count_conflict",)),
])
def test_public_instability_does_not_cancel_frozen_decision(tmp_path, monkeypatch, stable, issues):
    policy = SlowPolicy()
    loop, window, clock = make_loop(tmp_path, policy, execute=True)
    try:
        start(loop, policy)
        pending = loop._pending_decision
        assert pending['snapshot'].decision_ready
        track = loop._track
        def rejected(*args):
            observation, visible, update = track(*args)
            return observation, visible, replace(update, stable=stable, validation_issues=issues)
        monkeypatch.setattr(loop, "_track", rejected)
        process(loop)
        assert not pending["future"].cancelled()
        assert not policy.cancelled.is_set()
        assert loop._pending_decision is pending
        assert window.clicks == [] and len(policy.calls) == 1
        assert loop._window_deadline(pending["observation"]) <= pending["deadline"]
        assert loop._issue_since is None
    finally:
        close(loop)


@pytest.mark.parametrize(("stable", "issues"), [
    (False, ()), (True, ("physical_tile_count_conflict",)),
])
def test_preclick_uses_fresh_hand_without_waiting_for_public_stability(tmp_path, monkeypatch, stable, issues):
    policy = SlowPolicy()
    loop, window, clock = make_loop(tmp_path, policy, execute=True)
    try:
        start(loop, policy)
        pending = loop._pending_decision
        track = loop._track
        def rejected(*args):
            observation, visible, update = track(*args)
            assert observation.fingerprint == pending["observation"].fingerprint
            return observation, visible, replace(update, stable=stable, validation_issues=issues)
        monkeypatch.setattr(loop, "_track", rejected)
        monkeypatch.setattr(loop, '_verify', lambda *args: 'verified')
        result = loop._execute_action(pending["frame"], pending["observation"],
                                      pending["actions"][0], pending["domain"], deadline=pending["deadline"])
        assert result == 'verified' and len(window.clicks) == 1
    finally:
        close(loop)
