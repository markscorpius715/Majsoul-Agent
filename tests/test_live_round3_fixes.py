"""Fixes from live-20261018-204720 / 212426: pon, kita, yaku, timing, recovery."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from majsoul.domain import Action, Box, State
from majsoul.live.emergency import EmergencyConfig, TurnWatchdog
from majsoul.live.execution_verification import assess_execution_action
from majsoul.live.turn_timer import MatchTimer, TurnClock, TurnTiming
from majsoul.state.schemas import ObservedTile
from majsoul.strategy import EfficiencyPolicy

from conftest import observation
from test_live_bridge import make_observation
from test_live_play_fixes import make_loop  # noqa: F401  (fixture)
import test_live_watchdog as watchdog_tests
import tools.watchdog_fault_probe as probe

PON = Box(400, 500, 480, 550)


# ------------------------------------------------------------------ pon

def pon_state(hand, tile, **values):
    obs = observation(hand, phase='response', draw_index=None, buttons={'pon': PON, 'pass': PON},
                      call_tile=tile, call_from=1, **values)
    return State(obs, True, ((), ('9m',), ()))


def test_pon_option_now_describes_the_call(engine):
    """live 204720: pon reached the model as a bare button; it always passed."""
    pon = next(a for a in engine.legal_actions(pon_state('19m12344p1255s55z', '5z')) if a.kind == 'pon')
    f = pon.features
    assert f['called_tile'] == '5z' and f['call_tile_in_hand'] and f['gives_yaku']
    assert f['yakuhai_han_from_call'] == 1 and f['opens_closed_hand']
    assert f['shanten_after_call_best_discard'] <= f['shanten_before_call']
    assert isinstance(f['best_discards_after_call'], list)


def test_non_value_wind_pon_gives_no_yaku_and_north_is_never_yakuhai(engine):
    south = next(a for a in engine.legal_actions(pon_state('19m12344p1255s22z', '2z')) if a.kind == 'pon')
    assert south.features['gives_yaku'] is False  # seat 1z, round 1z: South is not yakuhai
    double = next(a for a in engine.legal_actions(pon_state('19m12344p1255s11z', '1z')) if a.kind == 'pon')
    assert double.features['yakuhai_han_from_call'] == 2  # seat and round wind
    assert engine.yakuhai_han('4z', observation()) == 0


def test_unknown_call_tile_is_reported_not_guessed(engine):
    pon = next(a for a in engine.legal_actions(pon_state('19m12344p1255s55z', None)) if a.kind == 'pon')
    assert pon.features['call_tile_known'] is False and 'gives_yaku' not in pon.features


# ---------------------------------------------------------- discard features

def test_discard_features_expose_form_shanten_triplets_and_yakuhai(engine):
    """live 212426 f604: the model broke a South triplet for a worse shanten."""
    obs = observation('1p5p5p6p8p1s3s4s6s9s2z2z2z6s'.replace('p', 'p ').replace('s', 's ').replace('z', 'z ')
                      .replace(' ', '') or '', seat_wind='3z')
    feats = {a.id: a.features for a in engine.legal_actions(State(obs, True, ((), (), ())))}
    south = next(f for i, f in feats.items() if i.endswith(':2z'))
    assert south['breaks_triplet'] and south['discards_yakuhai'] is False
    assert set(south['shanten_by_form']) == {'regular', 'seven_pairs', 'thirteen_orphans'}
    assert south['shanten'] > min(f['shanten'] for f in feats.values())


def test_yakuhai_discard_and_hints(engine):
    obs = observation('19m12344p1255s557z', seat_wind='1z')
    feats = {a.id: a.features for a in engine.legal_actions(State(obs, True, ((), (), ())))}
    haku = next(f for i, f in feats.items() if i.endswith(':5z'))
    assert haku['discards_yakuhai'] and not haku['breaks_triplet']
    chun = next(f for i, f in feats.items() if i.endswith(':7z'))
    assert chun['yaku_hints_after']['yakuhai_pairs_or_sets'] == {'5z': {'copies': 2, 'han_if_triplet': 1}}
    assert chun['yaku_hints_after']['closed_hand_riichi_available']


def test_guardrail_hides_shanten_losing_discards_without_a_threat(make_loop):
    loop = make_loop()
    good = Action('discard:0:9m', 'discard', '9m', 0, features={'shanten': 1})
    bad = Action('discard:1:2z', 'discard', '2z', 1, features={'shanten': 2})
    nuki = Action('button:nuki', 'nuki', button='nuki')
    calm = SimpleNamespace(observation=SimpleNamespace(riichi=(False, False, None), riichi_sticks=0))
    assert loop._model_candidates([good, bad, nuki], calm) == [good, nuki]
    threat = SimpleNamespace(observation=SimpleNamespace(riichi=(False, True, None), riichi_sticks=1))
    assert loop._model_candidates([good, bad, nuki], threat) == [good, bad, nuki]
    unattributed = SimpleNamespace(observation=SimpleNamespace(riichi=(False, None, None), riichi_sticks=1))
    assert bad in loop._model_candidates([good, bad, nuki], unattributed)


# -------------------------------------------------------------- kita / pass

def own_turn(hand, *, buttons=('nuki', 'pass'), drawn=13, ts=0.):
    obs = make_observation(hand, action_buttons=buttons, drawn_index=drawn, timestamp=ts)
    tiles = tuple(replace(t, box=(10+i*40, 600, 40+i*40, 660)) for i, t in enumerate(obs.hand))
    return replace(obs, hand=tiles, countdown={'status': 'complete', 'mode': 'base_plus_reserve',
                                               'base_seconds': 7, 'reserve_seconds': 20, 'total_seconds': 27})


HAND_NORTH = ('1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p', '1s', '2s', '3s', '5z', '4z')


def test_fallback_declares_kita_for_a_drawn_north_and_may_repeat(samples=None):
    """live 212426 agent-004 f148: a replacement North was discarded."""
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    first = own_turn(HAND_NORTH)
    watchdog.observe(first)
    boxes = {'nuki': PON, 'pass': PON}
    action, _ = watchdog.select(replace(first, timestamp=7.), boxes)
    assert action.kind == 'nuki' and action.button == 'nuki'
    watchdog.nuki_clicks = 4  # four Norths at most
    action, _ = watchdog.select(replace(first, timestamp=7.), boxes)
    assert action.kind == 'discard' and action.tile == '4z'


def test_kita_verification_accepts_a_changed_hand_or_vanished_button():
    before = own_turn(HAND_NORTH)
    replaced = own_turn(HAND_NORTH[:-1]+('6s',), buttons=())
    assert TurnWatchdog.verification(Action('emergency:nuki', 'nuki', button='nuki'), before, replaced).status == 'verified'
    same = own_turn(HAND_NORTH)
    assert TurnWatchdog.verification(Action('emergency:nuki', 'nuki', button='nuki'), before, same).status == 'unknown'


def test_pass_resolves_when_our_own_draw_replaces_the_call_window():
    """live 204720 f198: 'pass' on our own nuki prompt looked like the call
    window still open; the 4 s wait cost the whole base time."""
    call = replace(make_observation(HAND_NORTH[:-1], action_buttons=('pon', 'pass'), drawn_index=None),
                   frame_index=10, timestamp=1.)
    draw = replace(make_observation(HAND_NORTH, action_buttons=('nuki', 'pass')), frame_index=11, timestamp=2.)
    result = assess_execution_action(Action('button:pass', 'pass', button='pass'), call, draw, 'bottom')
    assert result.status == 'resolved' and result.reason == 'response_window_closed_own_draw'


# ---------------------------------------------------------------- timing

def test_turn_seen_late_still_gets_a_usable_window_from_the_reserve():
    timer = MatchTimer()
    timer.new_hand('1z:1:0')
    clock = TurnClock(timer, TurnTiming(), action_margin=1.)
    clock.start(100.)
    clock.observe(100., {'status': 'complete', 'mode': 'base_plus_reserve',
                         'base_seconds': 1, 'reserve_seconds': 20, 'total_seconds': 21})
    # Was: 1 s base + 2 s reserve -> decision already due. Now 5 s from first sight.
    assert clock.ordinary_due == pytest.approx(105.)
    assert clock.decision_due == pytest.approx(105.-1.5-1.)
    timer.spend(30)
    timeout = TurnClock(timer, TurnTiming(), action_margin=1.)
    timeout.start(200.)
    assert timeout.ordinary_due == pytest.approx(204.)  # never beyond the hard end


def test_local_efficiency_tolerates_missing_features_and_prefers_kita():
    """live 212426 f59: a None dora count crashed the local rung (TypeError)."""
    policy = EfficiencyPolicy()
    a = Action('discard:0:9m', 'discard', '9m', 0, features={'shanten': 1, 'ukeire': 8, 'retained_dora': None})
    b = Action('discard:1:1z', 'discard', '1z', 1, features={'shanten': 1, 'ukeire': None, 'retained_dora': None})
    assert policy.choose(None, [a, b]) == a
    nuki = Action('button:nuki', 'nuki', button='nuki')
    assert policy.choose(None, [a, b, nuki]) == nuki


# ---------------------------------------------------------------- recovery

def test_saved_layout_is_accepted_when_only_counters_are_unreadable(make_loop):
    """live 212426 agent-003: an unreadable honba digit kept a rejoin off the
    table for 180 s although the saved layout was right."""
    loop = make_loop(join=True)
    loop._layout_hint = ('bottom', 'left', 'right')
    item = lambda layout, issues: SimpleNamespace(layout=list(layout), issues=issues)
    blocked = SimpleNamespace(evaluations=[
        item(('bottom', 'top', 'left'), ['Metadata reading is incomplete', 'seat_top: Seat wind glyph is unknown']),
        item(('bottom', 'left', 'right'), ['Metadata reading is incomplete', 'honba: numeric_ocr_rejected'])])
    assert loop._hint_blocked_only_by_counters(blocked)
    seats = SimpleNamespace(evaluations=[item(('bottom', 'left', 'right'),
                                              ['Metadata reading is incomplete', 'seat_left: Seat wind glyph is unknown'])])
    assert not loop._hint_blocked_only_by_counters(seats)


def test_resumed_control_is_reported_after_a_restart(tmp_path):
    code, result, events, starts = watchdog_tests.run(tmp_path, probe.crash_then_complete)
    resumed = [e for e in events if e['event'] == 'agent_resumed']
    assert resumed and resumed[0]['session'] == 'agent-002' and resumed[0]['restart'] == 1


def test_riichi_finishes_with_the_planned_tile_when_the_selection_is_unprovable(tmp_path):
    """live 212426 agent-004 f60: dimmed tiles made the selection unprovable;
    the declaration was never finished and the user clicked by hand."""
    from majsoul.live.config import LiveConfig
    from majsoul.live.loop import LiveAgentLoop
    from test_live_loop import LIVE_CONFIG, Clock, FakePolicy, FakeWindow, mock_reader, small_roi
    base = LiveConfig.load(LIVE_CONFIG)
    config = replace(base, artifacts_root=tmp_path, lifecycle_ui=None, notify=replace(base.notify, enabled=False))
    now = Clock()
    window = FakeWindow(now)
    loop = LiveAgentLoop(window, config, execute=True, skip_menu=True, session='plan', sleep=now.sleep,
                         now=now, roi=small_roi(), reader=mock_reader(), policy=FakePolicy(), supervised=True)
    try:
        loop.reader.read_local = lambda *a, **k: None
        loop.reader.supports_local_input = True
        hand = HAND_NORTH[:-1]+('6s',)
        loop._riichi_plan, loop._pending_riichi = ('6s', 13), True
        def read(frame, index):
            tiles = hand if not window.clicks else hand[:-1]
            obs = own_turn(tiles, buttons=(), drawn=(13 if not window.clicks else None))
            return obs, {}
        loop._read_emergency = read
        assert loop._click_planned_riichi_tile(1) == 'verified'
        assert len(window.clicks) == 1 and window.clicks[0] == (10+13*40+15, 630)
        assert loop._own_riichi_round == '1z:1:0' and loop._pending_riichi is False
    finally:
        loop.decision_worker.close()
