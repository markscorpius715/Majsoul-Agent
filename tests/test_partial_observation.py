"""Current action evidence can survive UI loss; missing history stays missing."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from majsoul.domain import Box
from majsoul.live.bridge import BridgeError, build_domain_state
from majsoul.perception.schemas import RecognizedTile, SegmentedTile, TilePrediction
from majsoul.rules import RuleEngine, Rules
from majsoul.state import MahjongStateTracker, StateValidator
from test_state_tracking import BASE_HAND, snapshot, tiles


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("river_bottom", "melds_bottom", "kita_bottom")


def no_scores(observation):
    return replace(observation, players=tuple(replace(p, score=None, score_available=False)
                                             for p in observation.players))


def partial(observation=None, *, fields=FIELDS):
    observation = observation or snapshot(hand=BASE_HAND+("4z",), drawn_index=13)
    return replace(no_scores(observation),
                   players=tuple(replace(p, score=None, score_available=False,
                                         kita_count=None if f"kita_{p.slot}" in fields else p.kita_count)
                                 for p in observation.players),
                   unavailable_regions=fields, ui_occlusions=((900, 700, 1200, 800),),
                   issues=tuple(f"{name}: UI occlusion; region unavailable" for name in fields))


def tracker(stable_frames=1):
    return MahjongStateTracker.load(ROOT / "configs/sanma.json", stable_frames=stable_frames)


def domain(state, observation):
    items = tuple(RecognizedTile(
        SegmentedTile(Box(i*30, 900, i*30+20, 940), np.zeros((4,4,3), dtype=np.uint8)),
        TilePrediction(label, .99, 0., 1., "upright")) for i, label in enumerate(state.hand))
    visible = SimpleNamespace(hand=SimpleNamespace(tiles=items))
    return build_domain_state(state, observation, visible, {}, pending_riichi=False,
                              window=(0, 0, 1500, 1000))


def test_partial_requires_stable_current_fields_and_never_claims_complete_history():
    t = tracker(2)
    observation = partial()
    assert not t.update(observation).stable
    assert t.state is None
    update = t.update(replace(observation, timestamp=2., frame_index=2))
    assert update.stable and update.state.decision_ready
    assert [event.type.value for event in update.events] == ["RESYNC"]
    assert not observation.complete and not update.state.history_complete
    assert update.state.player("bottom").kita_count is None
    model = domain(update.state, observation).for_model()
    assert model["nuki"][0] is None
    assert not model["visible_tile_inventory_complete"]


@pytest.mark.parametrize("changes", [
    {"hand": tiles(BASE_HAND+(None,))},
    {"hand": tiles(BASE_HAND+("2m",))},
    {"hand": tiles(("1p",)*5+BASE_HAND[:9])},
    {"hand": tiles(("0p", "0p")+BASE_HAND[:12])},
    {"hand": tiles(BASE_HAND[:12])},
    {"remaining_tiles": None},
    {"round_wind": None},
    {"drawn_index": 14},
    {"ui_occlusions": ()},
    {"ui_occlusions": ((0, 0, float("nan"), 10),)},
    {"ui_occlusions": ((10, 10, 0, 0),)},
    {"unavailable_regions": ("hand",)},
    {"unavailable_regions": ("river_garbage_bottom",)},
    {"issues": ("hand: unknown",)},
])
def test_ui_loss_does_not_relax_hand_metadata_physical_or_provenance_gates(changes):
    update = tracker().update(replace(partial(), **changes))
    assert not update.stable and not update.committed
    assert update.state is None and update.validation_issues


@pytest.mark.parametrize("change", ["seat", "wall", "dora", "kita"])
def test_partial_reverse_invariants_reject_current_view_without_reusing_old_state(change):
    t = tracker()
    before = partial()
    assert t.update(before).stable
    after = replace(before, timestamp=2., frame_index=2)
    if change == "seat":
        after = replace(after, players=(replace(after.players[0], seat="2z", dealer=False),
                                        replace(after.players[1], seat="1z", dealer=True), after.players[2]))
    elif change == "wall":
        after = replace(after, remaining_tiles=before.remaining_tiles+1)
    elif change == "dora":
        after = replace(after, dora_indicators=tiles(("8s",)))
    else:
        # An available opponent count cannot become negative, even behind an
        # unrelated bottom UI row.
        after = replace(after, players=(after.players[0], replace(after.players[1], kita_count=-1), after.players[2]))
    update = t.update(after)
    assert not update.stable and update.validation_issues
    assert update.state.source_frame == before.frame_index
    with pytest.raises(BridgeError, match="differs from the validated state"):
        domain(update.state, after)


def test_unknown_riichi_allows_only_proved_draw_discard_and_no_guessed_value():
    observation = partial(snapshot(hand=BASE_HAND+("4z",), drawn_index=13, sticks=1))
    state = tracker().update(observation).state
    assert state.player("bottom").riichi is None
    d = domain(state, observation)
    actions = RuleEngine(Rules.load(ROOT / "configs/sanma.json")).legal_actions(d)
    assert [a.hand_index for a in actions if a.kind == "discard"] == [13]
    features = actions[0].features
    assert features["retained_dora"] is None and features["hand_value_if_next_tsumo"] == {}
    assert all(item["genbutsu"] is None for item in features["tile_danger"])
    assert features["effective_tile_count_is_upper_bound"]


def test_known_riichi_and_historical_discards_survive_ui_loss_without_invented_events():
    t = tracker()
    first = no_scores(snapshot(hand=BASE_HAND+("4z",), drawn_index=13, sticks=1,
                               rivers=(("7s",), (), ()), horizontal=(0, None, None)))
    assert t.update(first).state.player("bottom").riichi is True
    hidden = partial(replace(first, timestamp=2., frame_index=2,
                             players=(replace(first.players[0], river=()), *first.players[1:])))
    update = t.update(hidden)
    assert update.state.player("bottom").riichi is True
    assert update.state.player("bottom").river == ()
    assert update.state.player("bottom").discard_history == ("7s",)
    assert [e.type.value for e in update.events] == ["RESYNC"]
    visible_again = replace(first, timestamp=3., frame_index=3)
    update = t.update(visible_again)
    assert not update.events and not update.state.history_complete
    assert update.state.player("bottom").riichi is True
    assert not update.state.unavailable_regions


def test_hidden_positive_meld_count_never_infers_faces_or_openness_and_blocks_post_call_discard():
    engine = RuleEngine(Rules.load(ROOT / "configs/sanma.json"))
    observation = partial(snapshot(hand=BASE_HAND[:10]+("4z",), drawn_index=10))
    state = tracker().update(observation).state
    assert state.self_meld_count == 1
    assert state.player("bottom").melds == ()
    assert len([a for a in engine.legal_actions(domain(state, observation)) if a.kind == "discard"]) == 11
    # No draw evidence after a possible call: do not invent kuikae freedom.
    observation = replace(observation, drawn_index=None, action_buttons=("riichi",))
    state = tracker().update(observation).state
    assert engine.legal_actions(domain(state, observation)) == []


def test_hidden_known_kita_count_is_rejected_instead_of_silently_zeroed():
    observation = partial()
    observation = replace(observation, players=(replace(observation.players[0], kita_count=0), *observation.players[1:]))
    assert tracker().update(observation).validation_issues == ("UI-hidden kita count must remain unknown",)


def test_default_external_validator_stays_strict_for_partial_table():
    observation = partial()
    validator = StateValidator.load(ROOT / "configs/sanma.json")
    assert validator.validate_observation(observation)
    assert not validator.validate_observation(observation, allow_ui_occlusion=True)
