"""Fixes from live-20261018-164552: own riichi, wins, compound riichi, 7+20 timer."""
from dataclasses import replace
from pathlib import Path

import pytest

from majsoul.domain import Action, Box, State
from majsoul.live.config import LiveConfig
from majsoul.live.loop import LiveAgentLoop, action_point
from majsoul.live.turn_timer import MatchTimer, TurnClock, TurnTiming
from majsoul.state.schemas import ObservedTile

from conftest import observation
from test_live_bridge import make_observation
from test_live_loop import LIVE_CONFIG, Clock, FakePolicy, FakeWindow, mock_reader, small_roi

ROOT = Path(__file__).resolve().parents[1]
RIICHI = Box(700, 500, 800, 550)


# ---------------------------------------------------------------- rules layer

def test_riichi_is_one_compound_choice_per_tenpai_discard(engine):
    obs = observation('123456789p123s12z', buttons={'riichi': RIICHI}, riichi=(False, None, None))
    state = State(obs, True, ((), (), ()))
    actions = engine.legal_actions(state)
    ids = [a.id for a in actions]
    assert 'button:riichi' not in ids
    compound = [a for a in actions if a.kind == 'riichi_declare']
    assert {a.id for a in compound} == {'riichi:12:1z', 'riichi:13:2z'}
    assert all(a.button == 'riichi' and a.features['shanten'] == 0 and a.features['declares_riichi']
               for a in compound)
    assert action_point(compound[0], state) == RIICHI.center
    # Opponents' unknown riichi never restricts our own discards.
    assert sum(a.kind == 'discard' for a in actions) == 14


def test_unknown_own_riichi_is_what_locked_decisions_to_tsumogiri(engine):
    """The rules guard stays; the loop must supply our own known status."""
    obs = observation('123456789p123s12z', riichi=(None, None, None))
    assert [a.id for a in engine.legal_actions(State(obs, True, ((), (), ())))] == ['discard:13:2z']


# --------------------------------------------------------------------- loop

@pytest.fixture
def make_loop(tmp_path):
    loops = []

    def make(*, join=False, always_win=True):
        base = LiveConfig.load(LIVE_CONFIG)
        config = replace(base, artifacts_root=tmp_path, lifecycle_ui=None,
                         notify=replace(base.notify, enabled=False),
                         decision=replace(base.decision, always_win=always_win))
        now = Clock()
        loop = LiveAgentLoop(FakeWindow(now), config, execute=False, skip_menu=True, join=join,
                             session=f'fix-{len(loops)}', sleep=now.sleep, now=now, roi=small_roi(),
                             reader=mock_reader(), policy=FakePolicy(), supervised=True)
        loops.append(loop)
        return loop

    yield make
    for loop in loops:
        loop.decision_worker.close()


def unknown_riichi(obs, *, own_river=()):
    players = tuple(replace(p, riichi=None, river=own_river if i == 0 else p.river)
                    for i, p in enumerate(obs.players))
    return replace(obs, players=players)


def test_opponent_riichi_never_makes_our_own_status_unknown(make_loop):
    """f337-f399: an opponent's stick made every status None, including ours."""
    loop = make_loop()
    obs = loop._apply_own_state(unknown_riichi(make_observation()))
    own = obs.players[0]
    assert own.riichi is False and own.riichi_evidence['own_status_source'] == 'no_own_declaration_this_hand'
    assert all(p.riichi is None for p in obs.players[1:])
    loop._own_riichi_round = obs.round_id
    assert loop._apply_own_state(unknown_riichi(make_observation())).players[0].riichi is True


def test_own_declaration_belongs_to_one_hand_and_new_hand_restores_reserve(make_loop):
    loop = make_loop()
    first = make_observation()
    loop._apply_own_state(first)
    loop._own_riichi_round = first.round_id
    loop.match_timer.spend(12)
    assert loop.match_timer.reserve_left == 8
    nxt = replace(first, hand_number=2)
    assert loop._apply_own_state(unknown_riichi(nxt)).players[0].riichi is False
    assert loop.match_timer.reserve_left == 20 and not loop.match_timer.timeout


def test_joined_agent_uses_its_own_sideways_river_tile(make_loop):
    loop = make_loop(join=True)
    sideways = (ObservedTile('5z', 'horizontal'),)
    assert loop._apply_own_state(unknown_riichi(make_observation(), own_river=sideways)).players[0].riichi is True
    assert loop.match_timer.reserve_left is None  # joined mid-hand: reserve unknown
    plain = (ObservedTile('5z'),)
    assert loop._apply_own_state(unknown_riichi(make_observation(), own_river=plain)).players[0].riichi is False


def test_legal_win_is_taken_without_asking_the_model(make_loop):
    loop = make_loop()
    win = Action('button:ron', 'ron', button='ron')
    actions = [Action('button:pass', 'pass', button='pass'), win]
    assert loop._immediate_choice(actions, 'response') == (win, 'always_win')
    assert make_loop(always_win=False)._immediate_choice(actions, 'response') is None


def test_planned_riichi_tile_is_selected_without_a_second_model_call(make_loop):
    loop = make_loop()
    loop._riichi_plan = ('5z', 12)
    options = [Action('riichi_discard:3:4p', 'riichi', '4p', 3), Action('riichi_discard:12:5z', 'riichi', '5z', 12)]
    assert loop._immediate_choice(options, 'riichi_select') == (options[1], 'riichi_plan')
    # The planned tile moved: the same tile elsewhere, else any selectable tile.
    moved = [Action('riichi_discard:3:4p', 'riichi', '4p', 3), Action('riichi_discard:11:5z', 'riichi', '5z', 11)]
    assert loop._immediate_choice(moved, 'riichi_select')[0].hand_index == 11
    assert loop._immediate_choice(moved[:1], 'riichi_select')[0].hand_index == 3


# -------------------------------------------------------------------- timer

def timer():
    t = MatchTimer(base_seconds=7, reserve_seconds=20, timeout_turn_seconds=5)
    t.new_hand('1z:1:0')
    return t


def test_twelve_second_turn_spends_five_seconds_of_the_shared_reserve():
    t = timer()
    clock = TurnClock(t, TurnTiming())
    clock.start(0.)
    clock.close(12.)
    assert t.reserve_left == 15 and t.turn_base == 7 and t.turn_reserve == 15


def test_spent_reserve_leaves_five_second_turns_until_the_next_hand():
    t = timer()
    t.spend(20)
    assert t.timeout and t.turn_base == 5 and t.turn_reserve == 0
    clock = TurnClock(t, TurnTiming())
    clock.start(100.)
    assert clock.hard_due == pytest.approx(104.)  # 5 s, minus the rounding second
    t.new_hand('1z:2:0')
    assert not t.timeout and t.turn_reserve == 20


def test_screen_reading_is_the_authority_for_the_reserve():
    t = timer()
    clock = TurnClock(t, TurnTiming())
    clock.start(0.)
    clock.observe(0.5, {'status': 'complete', 'mode': 'base_plus_reserve', 'base_seconds': 6,
                        'reserve_seconds': 9, 'total_seconds': 15})
    assert t.reserve_left == 9
    assert clock.hard_due == pytest.approx(0.5+15-1)
    assert clock.base_due == pytest.approx(0.5+6-1)
    t2 = timer()
    TurnClock(t2, TurnTiming()).observe(3., {'status': 'complete', 'mode': 'base_only',
                                             'base_seconds': 4, 'reserve_seconds': 0, 'total_seconds': 4})
    assert t2.timeout and t2.turn_reserve == 0


def test_deadlines_are_strictly_ordered_inside_a_turn():
    t = timer()
    clock = TurnClock(t, TurnTiming(), action_margin=1.)
    clock.start(0.)
    assert clock.decision_due < clock.commit_at < clock.selection_due < clock.last_safe_input < clock.hard_due
    # An ordinary turn spends at most two seconds of the shared reserve.
    assert clock.commit_at == pytest.approx(7-1+2-1.5)


def test_reading_after_the_base_ran_out_uses_only_the_reserve():
    t = timer()
    clock = TurnClock(t, TurnTiming())
    clock.start(0.)
    clock.observe(8., {'status': 'complete', 'mode': 'reserve_only', 'base_seconds': 0,
                       'reserve_seconds': 18, 'total_seconds': 18})
    assert t.reserve_left == 18
    assert clock.hard_due == pytest.approx(8+18-1)
    clock.close(10.)
    assert t.reserve_left == pytest.approx(16)
