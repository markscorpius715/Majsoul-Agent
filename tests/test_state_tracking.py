from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.perception import (GlyphTemplateClassifier, TileTemplateClassifier,
                                VisionConfig)
from majsoul.actions import AgentActionType
from majsoul.state import (FrameObservation, FrameObservationReader,
                           GameEventType, MahjongStateTracker, ObservedMeld,
                           ObservedTile, PlayerObservation, StateValidator,
                           TrackingPhase)
from majsoul.state.__main__ import main as state_main


ROOT = Path(__file__).resolve().parents[1]
SLOTS = ("bottom", "top", "left")
BASE_HAND = ("1m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7p",
             "1s", "2s", "3s", "1z")


def tiles(labels, *, horizontal_index=None):
    return tuple(ObservedTile(
        label,
        "horizontal" if index == horizontal_index else "upright",
        0.99,
    ) for index, label in enumerate(labels))


def player(slot, seat, *, dealer=False, score=35000, river=(), melds=(), kita=0):
    return PlayerObservation(
        slot, seat, dealer, score,
        river=river,
        melds=tuple(ObservedMeld(tiles(group)) for group in melds),
        kita_count=kita,
    )


def snapshot(*, timestamp=1.0, frame=1, hand=BASE_HAND, drawn_index=None,
             remaining=55, honba=0, sticks=0, round_wind="1z", hand_number=1,
             scores=(35000, 35000, 35000), rivers=((), (), ()),
             horizontal=(None, None, None), melds=((), (), ()), kita=(0, 0, 0),
             issues=()):
    players = tuple(player(
        slot, seat, dealer=index == 0, score=scores[index],
        river=tiles(rivers[index], horizontal_index=horizontal[index]),
        melds=melds[index], kita=kita[index],
    ) for index, (slot, seat) in enumerate(zip(SLOTS, ("1z", "2z", "3z"))))
    return FrameObservation(
        timestamp, frame, SLOTS, round_wind, hand_number, honba, sticks,
        remaining, players, tiles(hand), drawn_index, tiles(("9s",)),
        issues=issues,
    )


def next_frame(observation, *, step=1):
    return replace(observation, timestamp=observation.timestamp + step / 30,
                   frame_index=observation.frame_index + step)


def commit(tracker, observation):
    first = tracker.update(observation)
    assert not first.committed
    second = tracker.update(next_frame(observation))
    assert second.committed
    return second


def test_state_cli_auto_layout_replays_confirmation_frames(tmp_path):
    # Use production gold weights, retaining the video's original ROI and glyphs.
    # A live-config replay otherwise assumes Steam geometry for these old frames.
    import yaml
    replay = yaml.safe_load((ROOT / "configs/live.steam.yaml").read_text(encoding="utf-8"))
    replay.update(roi="configs/roi.yaml", vision="configs/vision.yaml",
                  countdown=None, read_scores=True)
    config_path = tmp_path / "gold-video-replay.yaml"
    config_path.write_text(yaml.safe_dump(replay, allow_unicode=True), encoding="utf-8")
    output = tmp_path / "auto-layout.json"
    exit_code = state_main([
        "images",
        str(ROOT / "artifacts/roi/review/frame_1590.png"),
        str(ROOT / "artifacts/roi/review/frame_1605.png"),
        "--live-config", str(config_path),
        "--output", str(output),
    ])
    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["layout_source"] == "metadata"
    assert report["active_slots"] == ["bottom", "top", "left"]
    assert report["complete_observation_count"] == 2
    assert report["event_counts"] == {"ROUND_START": 1}
    assert all(item["layout_locked"] == ["bottom", "top", "left"]
               for item in report["updates"])


@pytest.fixture
def validator():
    return StateValidator.load(ROOT / "configs/sanma.json")


def test_stable_round_origin_creates_complete_baseline(validator):
    tracker = MahjongStateTracker(validator, stable_frames=2)
    update = commit(tracker, snapshot())

    assert [event.type for event in update.events] == [GameEventType.ROUND_START]
    assert update.events[0].certainty == "observed"
    assert update.events[0].details["joined_mid_round"] is False
    assert update.state.phase == TrackingPhase.TRACKING
    assert update.state.history_complete
    assert update.state.decision_ready
    assert update.state.dealer == "bottom"
    assert update.state.self_seat == "1z"
    duplicate = tracker.update(snapshot(timestamp=2.0, frame=20))
    assert duplicate.stable and not duplicate.committed


def test_mid_round_baseline_does_not_claim_observed_start(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    update = tracker.update(snapshot(rivers=(("7p",), (), ())))

    assert [event.type for event in update.events] == [GameEventType.ROUND_START]
    assert update.events[0].certainty == "inferred"
    assert update.events[0].details["baseline"] is True
    assert update.events[0].details["joined_mid_round"] is True
    assert not update.state.history_complete


@pytest.mark.parametrize("incomplete_source", ["mid_round", "resync"])
@pytest.mark.parametrize("new_hand_origin", [True, False])
def test_new_hand_completeness_is_independent_of_previous_hand(
        validator, incomplete_source, new_hand_origin):
    tracker = MahjongStateTracker(validator, stable_frames=2)
    commit(tracker, snapshot(rivers=(("7p",), (), ())))
    if incomplete_source == "resync":
        jumped = snapshot(timestamp=2.0, frame=20, remaining=50,
                          rivers=(("7p", "8p"), ("4s",), ()))
        broken = commit(tracker, jumped)
        assert broken.state.phase == TrackingPhase.RESYNC
    assert not tracker.state.history_complete
    new_hand = snapshot(timestamp=3.0, frame=30, hand_number=2,
                        rivers=((), (), ()) if new_hand_origin else (("8p",), (), ()))
    update = commit(tracker, new_hand)
    assert update.state.history_complete is new_hand_origin
    assert update.state.decision_ready is new_hand_origin
    assert update.state.player("bottom").discard_history == (() if new_hand_origin else ("8p",))
    assert update.events[-1].type == GameEventType.ROUND_START
    assert update.events[-1].details["joined_mid_round"] is (not new_hand_origin)


def test_same_hand_stability_does_not_repair_missing_history(validator):
    tracker = MahjongStateTracker(validator, stable_frames=2)
    before = snapshot(rivers=(("7p",), (), ()))
    commit(tracker, before)
    update = tracker.update(replace(before, timestamp=3.0, frame_index=30))
    assert update.stable
    assert not update.state.history_complete
    assert not update.state.decision_ready


def _without_scores(observation):
    players = tuple(replace(p, score=None, score_available=False)
                    for p in observation.players)
    return replace(observation, players=players)


def test_unavailable_scores_do_not_block_tracking(validator):
    tracker = MahjongStateTracker(validator, stable_frames=2)
    update = commit(tracker, _without_scores(snapshot()))

    assert [event.type for event in update.events] == [GameEventType.ROUND_START]
    assert update.state.phase == TrackingPhase.TRACKING
    assert update.state.decision_ready
    assert all(player.score is None for player in update.state.players)


def test_riichi_stick_without_scores_needs_no_payment_evidence(validator):
    before = _without_scores(snapshot())
    after = replace(next_frame(before), riichi_sticks=1)
    assert validator.validate_transition(before, after) == ()


def test_transition_skips_score_diff_when_values_go_unavailable(validator):
    before = snapshot()
    after = _without_scores(next_frame(before))
    assert validator.validate_transition(before, after) == ()


def test_partially_unavailable_scores_are_flagged(validator):
    players = tuple(replace(p, score=score, score_available=False)
                    for p, score in zip(snapshot().players, (35000, None, 35000)))
    observation = replace(snapshot(), players=players)
    issues = validator.validate_observation(observation)
    assert any("Scores must be bounded" in issue for issue in issues)


def test_draw_then_discard_are_reconstructed_locally(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot())
    drawn = snapshot(timestamp=2.0, frame=2, hand=BASE_HAND + ("4s",),
                     drawn_index=13, remaining=54)
    draw_update = tracker.update(drawn)
    discarded = snapshot(timestamp=3.0, frame=3, hand=BASE_HAND,
                         remaining=54, rivers=(("4s",), (), ()))
    discard_update = tracker.update(discarded)

    assert [(event.type, event.player, event.tile) for event in draw_update.events] == [
        (GameEventType.DRAW, "bottom", "4s")]
    assert [(event.type, event.player, event.tile) for event in discard_update.events] == [
        (GameEventType.DISCARD, "bottom", "4s")]
    assert discard_update.state.last_discard == ("bottom", "4s")
    assert discard_update.state.player("bottom").discard_history == ("4s",)


def test_self_discard_then_next_seat_draw_uses_previous_draw_marker(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    before = snapshot(hand=BASE_HAND + ("4s",), drawn_index=13, remaining=54)
    tracker.update(before)
    after = snapshot(timestamp=2.0, frame=2, hand=BASE_HAND,
                     remaining=53, rivers=(("4s",), (), ()))
    update = tracker.update(after)

    assert [(event.type, event.player) for event in update.events] == [
        (GameEventType.DISCARD, "bottom"), (GameEventType.DRAW, "top")]
    assert update.state.last_discard == ("bottom", "4s")
    assert update.state.current_actor == "top"


def test_self_draw_without_gap_marker_is_proved_by_hand_growth(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot())
    update = tracker.update(snapshot(
        timestamp=2.0, frame=2, hand=BASE_HAND + ("4s",),
        drawn_index=None, remaining=54))

    assert [(event.type, event.player, event.tile) for event in update.events] == [
        (GameEventType.DRAW, "bottom", "4s")]
    assert update.events[0].certainty == "inferred"
    assert update.events[0].details == {"evidence": "self_hand_conservation"}


def test_sparse_self_tsumogiri_is_proved_without_gap_marker(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot())
    update = tracker.update(snapshot(
        timestamp=2.0, frame=2, remaining=54,
        rivers=(("4s",), (), ())))

    assert [(event.type, event.player, event.tile) for event in update.events] == [
        (GameEventType.DRAW, "bottom", "4s"),
        (GameEventType.DISCARD, "bottom", "4s"),
    ]
    assert update.events[0].details == {"evidence": "self_hand_conservation"}


def test_extra_self_hand_tile_proves_next_draw_even_without_gap_marker(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot(
        hand=BASE_HAND + ("4s",), drawn_index=None, remaining=54))
    update = tracker.update(snapshot(
        timestamp=2.0, frame=2, hand=BASE_HAND, remaining=53,
        rivers=(("4s",), (), ())))

    assert [(event.type, event.player, event.tile) for event in update.events] == [
        (GameEventType.DISCARD, "bottom", "4s"),
        (GameEventType.DRAW, "top", None),
    ]


def test_riichi_requires_matching_stick_and_score_change(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    before = snapshot(hand=BASE_HAND + ("4s",), drawn_index=13, remaining=54)
    tracker.update(before)
    after = snapshot(timestamp=2.0, frame=2, remaining=54,
                     rivers=(("4s",), (), ()), horizontal=(0, None, None),
                     scores=(34000, 35000, 35000), sticks=1)
    update = tracker.update(after)

    assert [event.type for event in update.events] == [
        GameEventType.RIICHI, GameEventType.DISCARD,
        GameEventType.SCORE_UPDATE]
    assert update.state.player("bottom").riichi


def test_pon_removes_one_river_tile_without_losing_discard_history(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    before = snapshot(rivers=((), (), ("5p",)))
    tracker.update(before)
    after = snapshot(timestamp=2.0, frame=2,
                     melds=((), (("5p", "5p", "5p"),), ()))
    update = tracker.update(after)

    assert [(event.type, event.player) for event in update.events] == [
        (GameEventType.PON, "top")]
    assert update.state.player("left").discard_history == ("5p",)


@pytest.mark.parametrize("before_river", [(), ("6p",)])
def test_three_tile_meld_requires_a_compatible_called_river_tile(
        validator, before_river):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot(rivers=((), (), before_river)))
    update = tracker.update(snapshot(
        timestamp=2.0, frame=2,
        rivers=((), (), ()), melds=((), (("5p", "5p", "5p"),), ())))

    assert [event.type for event in update.events] == [GameEventType.RESYNC]
    assert any(
        marker in issue
        for issue in update.validation_issues
        for marker in ("lacks exactly one", "incompatible with the new meld"))


def test_self_pon_must_consume_the_matching_concealed_tiles(validator):
    before_hand = ("5p",) + BASE_HAND[1:]
    wrong_after = list(before_hand)
    wrong_after.remove("5p")
    wrong_after.remove("9m")
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot(
        hand=before_hand, rivers=((), (), ("5p",))))

    update = tracker.update(snapshot(
        timestamp=2.0, frame=2, hand=tuple(wrong_after),
        rivers=((), (), ()),
        melds=((('5p', '5p', '5p'),), (), ()),
    ))

    assert [event.type for event in update.events] == [GameEventType.RESYNC]
    assert any("Self hand transition" in issue
               for issue in update.validation_issues)


def test_self_pon_with_matching_concealed_tiles_is_valid(validator):
    before_hand = ("5p",) + BASE_HAND[1:]
    after_hand = list(before_hand)
    after_hand.remove("5p")
    after_hand.remove("5p")
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot(
        hand=before_hand, rivers=((), (), ("5p",))))

    update = tracker.update(snapshot(
        timestamp=2.0, frame=2, hand=tuple(after_hand),
        rivers=((), (), ()),
        melds=((('5p', '5p', '5p'),), (), ()),
    ))

    assert [(event.type, event.player) for event in update.events] == [
        (GameEventType.PON, "bottom")]


def test_partial_meld_emits_generic_event_and_refinement_is_not_a_new_event(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot(rivers=((), (), ("5p",))))
    partial = snapshot(
        timestamp=2.0, frame=2,
        rivers=((), (), ()), melds=((), (("5p", None, "5p"),), ()))

    update = tracker.update(partial)

    assert update.observation.complete
    assert [event.type for event in update.events] == [GameEventType.MELD]
    assert update.events[0].details["subtype"] == "unknown"
    assert update.state.uncertain_fields == ("melds_top",)
    assert not update.state.decision_ready
    assert update.state.player("top").melds[0].tiles == ("5p", None, "5p")

    refined = snapshot(
        timestamp=3.0, frame=3,
        rivers=((), (), ()), melds=((), (("5p", "5p", "5p"),), ()))
    refined_update = tracker.update(refined)

    assert refined_update.committed
    assert refined_update.events == ()
    assert refined_update.state.uncertain_fields == ()


def test_opponent_discard_with_wall_change_keeps_draw_actor_unknown(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot())

    update = tracker.update(snapshot(
        timestamp=2.0, frame=2, remaining=54,
        rivers=((), ("4s",), ())))

    assert [(event.type, event.player) for event in update.events] == [
        (GameEventType.DRAW, None), (GameEventType.DISCARD, "top")]
    assert update.events[0].details == {
        "relative_order": "ambiguous_with_discard"}


def test_observed_self_draw_after_opponent_discard_has_causal_order(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot())

    update = tracker.update(snapshot(
        timestamp=2.0, frame=2, remaining=54,
        hand=BASE_HAND + ("4s",), drawn_index=13,
        rivers=((), (), ("4s",))))

    assert [(event.type, event.player) for event in update.events] == [
        (GameEventType.DISCARD, "left"), (GameEventType.DRAW, "bottom")]


def test_kan_and_replacement_draw_are_reconstructed(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot(rivers=((), (), ("8p",))))

    update = tracker.update(snapshot(
        timestamp=2.0, frame=2, remaining=54,
        rivers=((), (), ()),
        melds=((), (("8p", "8p", "8p", "8p"),), ())))

    assert [(event.type, event.player, event.tile) for event in update.events] == [
        (GameEventType.KAN, "top", "8p"),
        (GameEventType.DRAW, "top", None),
    ]


def test_kita_and_replacement_draw_are_both_recorded(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    hand_with_north = BASE_HAND[:-1] + ("4z",)
    tracker.update(snapshot(hand=hand_with_north))
    after_hand = BASE_HAND[:-1] + ("8s",)
    update = tracker.update(snapshot(timestamp=2.0, frame=2, hand=after_hand,
                                     remaining=54, kita=(1, 0, 0)))

    assert [(event.type, event.player) for event in update.events] == [
        (GameEventType.KITA, "bottom"), (GameEventType.DRAW, "bottom")]


def test_valid_round_transition_closes_and_starts_round(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot())
    update = tracker.update(snapshot(
        timestamp=2.0, frame=2, hand_number=2,
        scores=(36000, 34000, 35000)))

    assert [event.type for event in update.events] == [
        GameEventType.SCORE_UPDATE, GameEventType.ROUND_END,
        GameEventType.ROUND_START]
    assert update.state.round_id == "1z:2:0"
    assert update.state.history_complete


def test_nonlocal_jump_resyncs_without_fabricating_events(validator):
    tracker = MahjongStateTracker(validator, stable_frames=2)
    commit(tracker, snapshot())
    jumped = snapshot(timestamp=2.0, frame=20, remaining=53,
                      rivers=(("7z", "8s"), (), ()))
    tracker.update(jumped)
    update = tracker.update(next_frame(jumped))

    assert [event.type for event in update.events] == [GameEventType.RESYNC]
    assert update.state.phase == TrackingPhase.RESYNC
    assert not update.state.history_complete
    assert update.validation_issues
    assert not any(event.type in (GameEventType.DRAW, GameEventType.DISCARD)
                   for event in update.events)


def test_stable_resync_recovers_current_window_without_repairing_history(validator):
    tracker=MahjongStateTracker(validator,stable_frames=2)
    commit(tracker,snapshot())
    jumped=snapshot(timestamp=2,frame=20,remaining=52,
                    rivers=((),('7z',),('8s',)),
                    hand=BASE_HAND+('4s',),drawn_index=13)
    broken=commit(tracker,jumped)
    assert broken.state.phase==TrackingPhase.RESYNC and not broken.state.decision_ready
    first=tracker.update(next_frame(jumped,step=2))
    assert not first.stable and first.state.phase==TrackingPhase.RESYNC
    recovered=tracker.update(next_frame(jumped,step=3))
    assert recovered.stable and recovered.committed and recovered.events==()
    assert recovered.state.phase==TrackingPhase.TRACKING
    assert recovered.state.decision_ready
    assert not recovered.state.history_complete
    assert recovered.state.history==broken.state.history


def test_resync_recovery_still_requires_valid_fresh_stable_observations(validator):
    tracker=MahjongStateTracker(validator,stable_frames=2)
    commit(tracker,snapshot())
    jumped=snapshot(timestamp=2,frame=20,remaining=53,rivers=(('7z','8s'),(),()))
    commit(tracker,jumped)
    invalid=replace(next_frame(jumped,step=2),issues=('hand: unknown tile',))
    assert tracker.update(invalid).validation_issues
    assert tracker.state.phase==TrackingPhase.RESYNC
    assert not tracker.update(next_frame(jumped,step=3)).stable
    recovered=tracker.update(next_frame(jumped,step=4))
    assert recovered.state.phase==TrackingPhase.TRACKING
    assert not recovered.state.history_complete
    assert not recovered.state.decision_ready  # No current own draw or enabled button.


def test_pass_is_an_agent_action_not_a_game_event():
    assert AgentActionType.PASS.value == "PASS"
    assert "PASS" not in {event.value for event in GameEventType}


def test_incomplete_frame_resets_stability_and_never_commits(validator):
    tracker = MahjongStateTracker(validator, stable_frames=2)
    incomplete = snapshot(issues=("hand: one tile is unknown",))
    first = tracker.update(incomplete)
    second = tracker.update(next_frame(incomplete))

    assert not first.stable and not second.stable
    assert not first.committed and not second.committed
    assert tracker.state is None


def test_unexplained_same_size_hand_replacement_forces_resync(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot())
    changed_hand = ("8s",) + BASE_HAND[1:]

    update = tracker.update(snapshot(
        timestamp=2.0, frame=2, hand=changed_hand))

    assert [event.type for event in update.events] == [GameEventType.RESYNC]
    assert any("Self hand" in issue for issue in update.validation_issues)


def test_invalid_sanma_chi_is_rejected(validator):
    observation = snapshot(melds=((), (("3p", "4p", "5p"),), ()))
    issues = validator.validate_observation(observation)

    assert any("invalid meld" in issue for issue in issues)


def test_partial_meld_must_have_a_legal_completion(validator):
    observation = snapshot(melds=((), (("5p", None, "6p"),), ()))
    issues = validator.validate_observation(observation)

    assert any("invalid meld" in issue for issue in issues)


def test_invalid_seat_wind_is_rejected_before_turn_inference(validator):
    observation = snapshot()
    bad_player = replace(observation.players[2], seat="4z")
    observation = replace(
        observation,
        players=observation.players[:2] + (bad_player,))

    assert "Seat winds do not match the configured player count" in \
        validator.validate_observation(observation)


def test_observations_must_be_strictly_ordered(validator):
    tracker = MahjongStateTracker(validator, stable_frames=1)
    tracker.update(snapshot())
    with pytest.raises(ValueError, match="increasing"):
        tracker.update(snapshot(timestamp=1.0, frame=2))


def test_real_frame_combines_visible_tiles_metadata_and_confidence(validator, gold_tile_classifier):
    config = VisionConfig.load(ROOT / "configs/vision.yaml")
    reader = FrameObservationReader(
        config,
        gold_tile_classifier,
        GlyphTemplateClassifier(ROOT / "artifacts/templates/glyphs", config),
    )
    observation = reader.read(
        read_image(ROOT / "artifacts/roi/review/frame_1590.png"),
        ROIConfig.load(ROOT / "configs/roi.yaml"), SLOTS,
        timestamp=53.0, frame_index=1590)

    assert observation.complete
    assert validator.validate_observation(observation) == ()
    assert observation.round_id == "1z:2:0"
    assert observation.remaining_tiles == 21
    assert [player.score for player in observation.players] == [42000, 30000, 32000]
    assert observation.confidence["overall"] > 0
    assert observation.to_dict()["hand"][0]["confidence"] > 0


def test_four_player_observation_without_kita_is_non_applicable_not_missing():
    slots = ("bottom", "top", "left", "right")
    sequence = SimpleNamespace(tiles=(), drawn_index=None, issues=(), complete=True)
    meld = SimpleNamespace(groups=(), issues=(), complete=True)
    visible = SimpleNamespace(
        active_slots=slots, hand=sequence, dora=sequence,
        rivers={slot: sequence for slot in slots},
        melds={slot: meld for slot in slots}, kita={}, issues=(), complete=True)
    complete_number = lambda value: SimpleNamespace(
        value=value, glyphs=(), issues=(), complete=True)
    round_reading = SimpleNamespace(
        round_wind="1z", hand_number=1, glyphs=(), issues=(), complete=True)
    seats = {
        slot: SimpleNamespace(
            wind=f"{index + 1}z", dealer=index == 0,
            glyph=SimpleNamespace(confidence=0.99), issues=(), complete=True)
        for index, slot in enumerate(slots)
    }
    metadata = SimpleNamespace(
        active_slots=slots, round=round_reading,
        remaining_tiles=complete_number(69), honba=complete_number(0),
        riichi_sticks=complete_number(0),
        scores={slot: complete_number(25000) for slot in slots},
        seats=seats, issues=(), complete=True)
    reader = object.__new__(FrameObservationReader)
    reader.visible_reader = SimpleNamespace(read=lambda *_: visible)
    reader.metadata_reader = SimpleNamespace(read=lambda *_: metadata)
    roi = ROIConfig.load(ROOT / "configs/roi.yaml").for_game_mode(4)

    observation = reader.read(
        np.zeros((1156, 1920, 3), dtype=np.uint8), roi, slots,
        timestamp=1.0, frame_index=30)

    assert [player.kita_count for player in observation.players] == [0, 0, 0, 0]
    assert not any(key.startswith("kita_") for key in observation.confidence)
