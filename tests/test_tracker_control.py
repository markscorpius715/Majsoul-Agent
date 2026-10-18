from dataclasses import replace
import time

import pytest

from majsoul.controller import ControlError, action_verified, preflight
from majsoul.domain import Action, Box, MeldView, State
from majsoul.tracker import StateTracker
from majsoul.vision import Layout
from conftest import observation


def commit(tracker, obs):
    tracker.update(obs)
    return tracker.update(replace(obs, captured_at=obs.captured_at+0.1))


def test_stability_and_duplicate_frames():
    tracker = StateTracker()
    obs = observation()
    assert not tracker.update(obs).stable
    assert tracker.update(replace(obs, captured_at=2)).stable
    with pytest.raises(ValueError):
        tracker.update(replace(obs, captured_at=2))
    assert not tracker.update(replace(obs, captured_at=3, issues=["low confidence"])).stable
    assert not tracker.update(replace(obs, captured_at=4)).stable


def test_discard_history_survives_called_tile_removal():
    tracker = StateTracker()
    commit(tracker, observation(rivers=((), ("1m", "9m"), ())))
    second = replace(tracker.last, captured_at=2, rivers=((), ("1m",), ()))
    state = commit(tracker, second)
    assert state.discard_history[1] == ("1m", "9m")
    state = commit(tracker, replace(second, captured_at=3, rivers=((), ("1m", "2p"), ())))
    assert state.discard_history[1] == ("1m", "9m", "2p")


def test_passed_ron_temporary_and_riichi_furiten():
    for riichi in (False, True):
        tracker = StateTracker()
        obs = observation("123456789p123s1z", draw_index=None, phase="response", riichi=(riichi, False, False), buttons={"ron": Box(1,1,2,2)})
        state = commit(tracker, obs)
        tracker.record_action(Action("button:pass", "pass", button="pass"), state)
        state = commit(tracker, replace(obs, captured_at=2, phase="waiting", buttons={}))
        assert state.temporary_furiten
        assert state.riichi_furiten == riichi
        draw = observation("123456789p123s11z", time=3, riichi=(riichi, False, False))
        state = commit(tracker, draw)
        assert not state.temporary_furiten
        assert state.riichi_furiten == riichi
        reset = commit(tracker, replace(draw, captured_at=4, round_id="1z:2:0"))
        assert not reset.riichi_furiten


def test_preflight_rejects_stale_changed_and_moved_targets(engine, state):
    action = engine.legal_actions(state)[0]
    layout = Layout({}, {})
    fresh = replace(state, observation=replace(state.observation, captured_at=2))
    assert preflight(action, state, fresh, engine, layout, now=2)
    with pytest.raises(ControlError, match="expired"):
        preflight(action, state, fresh, engine, layout, now=30)
    changed = replace(fresh, observation=replace(fresh.observation, phase="waiting"))
    with pytest.raises(ControlError):
        preflight(action, state, changed, engine, layout, now=2)
    tiles = list(fresh.observation.hand)
    tiles[0] = replace(tiles[0], box=Box(500, 600, 530, 660))
    moved = replace(fresh, observation=replace(fresh.observation, hand=tuple(tiles)))
    with pytest.raises(ControlError, match="target moved"):
        preflight(action, state, moved, engine, layout, now=2)


def test_verification_requires_action_specific_evidence(state):
    before = state.observation
    action = Action("discard:0:1m", "discard", "1m", 0)
    assert not action_verified(action, before, replace(before, buttons={"pass": Box(1,1,2,2)}))
    assert not action_verified(action, before, replace(before, rivers=(("9m",), (), ())))
    after = replace(before, rivers=(("1m",), (), ()))
    assert action_verified(action, before, after)
    assert not action_verified(action, before, replace(after, issues=["bad frame"]))
    assert action_verified(Action("button:nuki", "nuki"), before, replace(before, nuki=(1,0,0)))


def test_result_overlay_retains_round_context():
    tracker = StateTracker()
    before = observation()
    commit(tracker, before)
    overlay = replace(before, captured_at=2, phase="result", round_id=None, seat_wind=None, round_wind=None)
    after = commit(tracker, overlay)
    assert after.observation.round_id == before.round_id
    assert action_verified(Action("button:tsumo", "tsumo"), before, after.observation)


def test_forged_action_with_matching_id_is_rejected(engine, state):
    action = engine.legal_actions(state)[0]
    forged = replace(action, hand_index=1, tile="9m")
    fresh = replace(state, observation=replace(state.observation, captured_at=2))
    with pytest.raises(ControlError, match="no longer legal"):
        preflight(forged, state, fresh, engine, Layout({},{}), 2)


def test_nuki_replacement_survives_intermediate_waiting_frame():
    tracker = StateTracker()
    commit(tracker, observation("123456789p123s14z"))
    mid = observation("123456789p123s1z", time=2, draw_index=None, phase="waiting", nuki=(1,0,0))
    commit(tracker, mid)
    after = commit(tracker, observation("123456789p123s11z", time=3, nuki=(1,0,0)))
    assert after.replacement_draw
