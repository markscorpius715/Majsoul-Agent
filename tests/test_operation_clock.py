"""Read the clock once per operation window, then count down locally."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from majsoul.live.config import LiveConfig
from majsoul.live.loop import LiveAgentLoop
from majsoul.live.operation_clock import OperationClock
from majsoul.live.protection import ProtectionConfig

from test_live_loop import LIVE_CONFIG, Clock, FakePolicy, FakeWindow, mock_reader, small_roi


def clock():
    return OperationClock.from_protection(ProtectionConfig())


def test_unknown_readings_never_move_a_confirmed_deadline():
    c = clock()
    c.start(10.)
    assert c.observe(10., 25)
    assert c.due == 18.  # maximum operation time caps the 21.5 s clock deadline
    for value in (None, float('nan'), -1, True, '5'):
        assert not c.observe(12., value)
    assert c.due == 18. and c.readings == 1


def test_later_readings_can_only_shorten_the_deadline():
    c = clock()
    c.start(10.)
    c.observe(10., 6)
    c.observe(11., 25)
    assert c.due == pytest.approx(12.5)
    c.observe(11.5, 2)
    assert c.due == pytest.approx(10.)


def test_without_reading_only_base_time_from_earliest_possible_start():
    c = clock()
    c.start(14.4, earliest=10.)
    assert not c.confirmed
    assert c.due == pytest.approx(11.5)
    c.observe(14.4, 25)  # a reading replaces the assumption
    assert c.due == pytest.approx(22.4)


def test_earliest_start_never_after_first_sight_and_reset_clears_window():
    c = clock()
    c.start(5., earliest=9.)
    assert c.earliest_start == 5.
    c.observe(5., 25)
    c.reset()
    assert not c.open and c.due is None and c.last_reading is None


def test_agent_margin_ends_its_budget_before_the_supervisor():
    config = ProtectionConfig()
    supervisor = OperationClock.from_protection(config)
    agent = OperationClock.from_protection(config, extra_margin=1.)
    for c in (supervisor, agent):
        c.start(0.)
        c.observe(0., 25)
    assert agent.due == supervisor.due-1
    supervisor.reset(); agent.reset()
    for c in (supervisor, agent):
        c.start(0.)
    assert agent.due == supervisor.due-1


@pytest.fixture
def supervised_loop(tmp_path):
    base = LiveConfig.load(LIVE_CONFIG)
    config = replace(base, artifacts_root=tmp_path, lifecycle_ui=None,
                     notify=replace(base.notify, enabled=False))
    now = Clock()
    loop = LiveAgentLoop(FakeWindow(now), config, execute=False, skip_menu=True,
                         session='clock', sleep=now.sleep, now=now, roi=small_roi(),
                         reader=mock_reader(), policy=FakePolicy(), supervised=True)
    yield loop
    loop.decision_worker.close()


def observation(at, **countdown):
    return SimpleNamespace(timestamp=at, countdown=countdown or {'status': 'unknown'})


def test_supervised_agent_counts_one_reading_down_and_ignores_unreadable_frames(supervised_loop):
    loop = supervised_loop
    e = loop.config.emergency
    margin = loop.config.decision.action_margin_seconds
    loop._reset_window_clock(100.)
    loop._actionable_since = 100.4
    first = loop._window_deadline(observation(100.4, status='complete', total_seconds=25,
                                              base_seconds=5, reserve_seconds=20, mode='base_plus_reserve'))
    # Base left + the reserve an ordinary turn may spend, minus rounding,
    # the tsumogiri lead and the click margin.
    expected = 100.4+5-e.quantization_seconds+e.reserve_per_turn_seconds-e.lead_seconds-margin
    assert first == pytest.approx(expected)
    assert loop._window_deadline(observation(104.8)) == pytest.approx(first)
    assert loop.match_timer.reserve_left == 20


def test_supervised_agent_without_reading_uses_the_hand_timer(supervised_loop):
    loop = supervised_loop
    e = loop.config.emergency
    rules = loop.engine.rules.data
    margin = loop.config.decision.action_margin_seconds
    loop.match_timer.new_hand('1z:1:0')
    loop._reset_window_clock(200.)
    loop._actionable_since = 200.3
    expected = (200.+rules['turn_seconds']-e.quantization_seconds+e.reserve_per_turn_seconds
                -e.lead_seconds-margin)
    assert loop._window_deadline(observation(200.3)) == pytest.approx(expected)


def test_supervised_agent_with_spent_reserve_gets_timeout_turns(supervised_loop):
    loop = supervised_loop
    e = loop.config.emergency
    rules = loop.engine.rules.data
    margin = loop.config.decision.action_margin_seconds
    loop.match_timer.new_hand('1z:1:0')
    loop.match_timer.spend(rules['reserve_seconds'])
    loop._reset_window_clock(300.)
    loop._actionable_since = 300.2
    expected = 300.+rules['fallback_turn_seconds']-e.quantization_seconds-e.lead_seconds-margin
    assert loop._window_deadline(observation(300.2)) == pytest.approx(expected)


def test_unsupervised_agent_keeps_its_own_budget(tmp_path):
    base = LiveConfig.load(LIVE_CONFIG)
    config = replace(base, artifacts_root=tmp_path, lifecycle_ui=None,
                     notify=replace(base.notify, enabled=False))
    now = Clock()
    loop = LiveAgentLoop(FakeWindow(now), config, execute=False, skip_menu=True,
                         session='plain', sleep=now.sleep, now=now, roi=small_roi(),
                         reader=mock_reader(), policy=FakePolicy())
    try:
        assert loop.operation_clock is None
        loop._reset_window_clock(1.)
    finally:
        loop.decision_worker.close()
