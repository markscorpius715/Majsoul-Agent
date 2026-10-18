"""Hidden physical labels remain unknown while proved tile-family counts apply."""

from dataclasses import replace
from pathlib import Path

import pytest

from conftest import observation
from majsoul.domain import MeldView, State
from majsoul.live.bridge import BridgeError, build_domain_state
from majsoul.live.verification import assess_action
from majsoul.state import MahjongStateTracker, ObservedMeld, ObservedTile, StateValidator
from majsoul.tile_inventory import MeldKnowledge, TileInventory, observed_meld_knowledge
from majsoul.tiles import tile_index
from test_live_bridge import (HAND14, WINDOW, make_observation, make_player, make_state,
                              visible_reading)
from test_live_verification import discard_case
from test_state_tracking import BASE_HAND, commit, snapshot

ROOT = Path(__file__).resolve().parents[1]
BACKS = (True, False, False, True)


def knowledge(tile="5z", second=None):
    return MeldKnowledge((None, tile, second or tile, None), "ankan", False, BACKS)


def observed(tile="5z"):
    return ObservedMeld(tuple(ObservedTile(label, face_down=back, confidence=.99)
                              for label, back in zip(knowledge(tile).tiles, BACKS)), "ankan", False)


def with_opponent_quad(frame, tile="5z", slot="left"):
    return replace(frame, players=tuple(replace(p, melds=(observed(tile),)) if p.slot == slot else p
                                        for p in frame.players))


@pytest.mark.parametrize("tile,second", [("1m", "1m"), ("9m", "9m"), ("1p", "1p"),
                                         ("5p", "5p"), ("0p", "5p"), ("5s", "0s"),
                                         ("4z", "4z"), ("7z", "7z")])
def test_quad_constraint_counts_family_without_filling_hidden_labels(tile, second):
    meld = knowledge(tile, second)
    assert not meld.complete and meld.bounded_concealed_quad
    inventory = TileInventory.build((), (meld,))
    assert inventory.known == (tile, second)
    assert inventory.family_counts[tile_index(tile)] == 4
    assert sum(inventory.family_counts) == 4 and inventory.unresolved == ()
    assert meld.tiles == (None, tile, second, None)
    assert inventory.constraints[0]["hidden_red_identity"] == "unknown"
    inventory.validate_family_counts()


@pytest.mark.parametrize("changes", [
    {"kind": "unknown"}, {"opened": None}, {"opened": True}, {"kind": "daiminkan"},
    {"face_down": ()}, {"face_down": (False, False, False, False)},
    {"face_down": (True, True, False, False)}, {"face_down": (1, 0, 0, 1)},
    {"tiles": (None, None, "5z", None)}, {"tiles": (None, "5z", "6z", None)},
    {"tiles": (None, "5z", None)}, {"tiles": (None, "0z", "0z", None)},
])
def test_generic_unknown_or_wrong_quad_grammar_is_not_a_blind_spot(changes):
    assert not replace(knowledge(), **changes).bounded_concealed_quad


def test_quad_with_external_copy_is_impossible_even_when_only_three_faces_known():
    inventory = TileInventory.build(("5z",), (knowledge(),))
    assert inventory.known.count("5z") == 3
    assert inventory.family_counts[tile_index("5z")] == 5
    with pytest.raises(ValueError, match="four copies"):
        inventory.validate_family_counts()


def test_rule_features_exclude_structurally_exhausted_wait_without_inventing_red(engine):
    obs = observation("123789s111z4446p9m", dora_indicators=("1m",))
    state = State(obs, True, ((), (), ()))
    ordinary = engine.discard_features(state, 13)
    assert ordinary["shanten"] == 0 and ordinary["effective_tiles"]["5p"] == 4
    hidden = knowledge("5p")
    obs.melds = ((), (MeldView(hidden.tiles, False, "ankan", BACKS),), ())
    constrained = engine.discard_features(state, 13)
    assert "5p" not in constrained["effective_tiles"]
    assert constrained["ukeire"] == ordinary["ukeire"]-4
    assert constrained["effective_tile_count_basis"] == "known_faces_and_structural_ankan_constraints"
    assert engine.visible(state).count("5p") == 2 and "0p" not in engine.visible(state)
    assert engine.legal_actions(state)
    payload = state.for_model()
    assert payload["melds"][1][0]["tiles"] == (None, "5p", "5p", None)
    assert payload["melds"][1][0]["face_down"] == BACKS
    assert payload["visible_meld_constraints"][0]["family_count"] == 4
    assert payload["hidden_opponent_hands"] == "unknown"


def test_unreadable_face_and_own_unresolved_quad_still_generate_no_actions(engine):
    state = State(observation(), True, ((), (), ()))
    state.observation.melds = ((), (MeldView(("5z", None, "5z"), True, "pon"),), ())
    assert engine.legal_actions(state) == []
    state.observation = observation("123456p123s11z", draw_index=10)
    hidden = knowledge()
    state.observation.melds = ((MeldView(hidden.tiles, False, "ankan", BACKS),), (), ())
    assert engine.legal_actions(state) == []


def test_known_red_multiplicity_is_not_hidden_by_the_constraint_gate(engine):
    state = State(observation(), True, ((), (), ()))
    hidden = knowledge("0p")
    state.observation.melds = ((), (MeldView(hidden.tiles, False, "ankan", BACKS),), ())
    with pytest.raises(ValueError, match="red tile multiplicity"):
        engine.legal_actions(state)


def test_state_validator_includes_quad_constraint_and_preserves_unknown_positions():
    validator = StateValidator.load(ROOT / "configs/sanma.json")
    valid = with_opponent_quad(snapshot())
    assert validator.validate_observation(valid) == ()
    invalid = replace(valid, dora_indicators=(ObservedTile("5z"),))
    assert any("More than four" in issue for issue in validator.validate_observation(invalid))
    assert tuple(t.tile for t in valid.player("left").melds[0].tiles) == (None, "5z", "5z", None)


def test_continuous_history_survives_opponent_ankan_and_allows_next_self_decision(engine):
    validator = StateValidator.load(ROOT / "configs/sanma.json")
    tracker = MahjongStateTracker(validator, stable_frames=2)
    stages = [
        dict(hand=BASE_HAND, remaining=55),
        dict(hand=BASE_HAND+("9p",), remaining=54, drawn_index=13),
        dict(hand=BASE_HAND, remaining=54, rivers=(("9p",), (), ())),
        dict(hand=BASE_HAND, remaining=53, rivers=(("9p",), (), ())),
        dict(hand=BASE_HAND, remaining=53, rivers=(("9p",), ("3z",), ())),
        dict(hand=BASE_HAND, remaining=52, rivers=(("9p",), ("3z",), ())),
        dict(hand=BASE_HAND, remaining=52, rivers=(("9p",), ("3z",), ())),
        dict(hand=BASE_HAND, remaining=51, rivers=(("9p",), ("3z",), ())),
        dict(hand=BASE_HAND, remaining=51, rivers=(("9p",), ("3z",), ("2z",))),
        dict(hand=BASE_HAND+("9p",), remaining=50, drawn_index=13,
             rivers=(("9p",), ("3z",), ("2z",))),
    ]
    for index, fields in enumerate(stages):
        frame = snapshot(timestamp=index+1., frame=2*index+1, **fields)
        frame = replace(frame, players=tuple(replace(p, score=None, score_available=False)
                                             for p in frame.players))
        if index >= 6:
            frame = with_opponent_quad(frame)
        update = commit(tracker, frame)
        assert not update.validation_issues and update.state.history_complete
        assert update.state.decision_ready
    state = update.state
    assert state.uncertain_fields == ("melds_left",)
    assert state.blind_spots == ("melds_left",) and state.blocking_uncertain_fields == ()
    assert state.to_dict()["blind_spots"] == ("melds_left",)
    assert state.player("left").melds[0].tiles == (None, "5z", "5z", None)
    domain = build_domain_state(state, update.observation, visible_reading(state.hand), {},
                                pending_riichi=False, window=WINDOW)
    assert domain.observation.phase == "discard" and engine.legal_actions(domain)
    assert domain.observation.melds[2][0].face_down == BACKS


def test_expected_blind_spot_does_not_hide_other_uncertainty_or_incomplete_history():
    from majsoul.state.schemas import TrackedMeld
    hidden = knowledge()
    players = (make_player("bottom", "1z", dealer=True),
               make_player("top", "2z", melds=(TrackedMeld(hidden.tiles, "ankan", False, BACKS),)),
               make_player("left", "3z"))
    state = make_state(players=players, uncertain_fields=("melds_top",))
    assert state.decision_ready and state.blind_spots == ("melds_top",)
    assert not replace(state, history_complete=False).decision_ready
    mixed = replace(state, uncertain_fields=("melds_top", "action_buttons"))
    assert mixed.blocking_uncertain_fields == ("action_buttons",) and not mixed.decision_ready
    bad = replace(players[1], melds=players[1].melds+(TrackedMeld(("6z", None, "6z")),))
    mixed = replace(state, players=(players[0], bad, players[2]))
    assert mixed.blind_spots == () and mixed.blocking_uncertain_fields == ("melds_top",)


def test_unresolved_own_quad_is_blocking_even_when_legacy_diagnostics_omit_it():
    from majsoul.state.schemas import TrackedMeld
    hidden = knowledge()
    players = (make_player("bottom", "1z", dealer=True,
                           melds=(TrackedMeld(hidden.tiles, "ankan", False, BACKS),)),
               make_player("top", "2z"), make_player("left", "3z"))
    state = make_state(players=players, uncertain_fields=())
    assert not state.decision_ready and state.blocking_uncertain_fields == ("melds_bottom",)
    with pytest.raises(BridgeError, match="unknown tiles"):
        build_domain_state(state, make_observation(), visible_reading(HAND14), {},
                           pending_riichi=False, window=WINDOW)


def test_verification_rejects_fifth_family_copy_concealed_by_two_unknown_back_labels():
    action, before, after = discard_case()
    before = with_opponent_quad(before)
    after = with_opponent_quad(after)
    assert assess_action(action, before, after, "bottom").status == "verified"
    # A known dora with this family makes the table physically impossible.
    after = replace(after, dora_indicators=(ObservedTile("5z"),))
    result = assess_action(action, before, after, "bottom")
    assert result.status == "unknown" and result.reason == "physical_tile_count_conflict"


def test_real_gold_ankan_reading_provides_the_same_blind_spot_grammar():
    from majsoul.calibration.frames import read_image
    from majsoul.calibration.roi_config import ROIConfig
    from majsoul.live.classifier import build_tile_classifier
    from majsoul.live.config import LiveConfig
    from majsoul.perception import VisionConfig
    from majsoul.perception.meld_reader import MeldReader
    from majsoul.perception.roi_extractor import extract_roi
    torch = pytest.importorskip("torch")
    config = LiveConfig.load(ROOT / "configs/live.steam.yaml")
    vision = VisionConfig.load(config.vision)
    roi = ROIConfig.load(config.roi)
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        classifier = build_tile_classifier("cnn", vision, cnn_checkpoint=config.cnn_checkpoint,
                                            expected_metadata=config.cnn_expected)
        reading = MeldReader(vision, classifier, "bottom").read(
            extract_roi(read_image(ROOT / "datasets/暗杠.jpg"), roi, "melds_bottom"))
    finally:
        torch.set_num_threads(previous)
    assert reading.complete and reading.kinds[0] == "ankan"
    group = ObservedMeld(tuple(ObservedTile(item.prediction.tile,
                                          face_down=item.detection.face_down) for item in reading.groups[0]),
                        reading.kinds[0], reading.opened[0])
    parsed = observed_meld_knowledge(group)
    assert parsed.tiles == (None, "9m", "9m", None)
    assert parsed.bounded_concealed_quad
    assert TileInventory.build((), (parsed,)).family_counts[tile_index("9m")] == 4
