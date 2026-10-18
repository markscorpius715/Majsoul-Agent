from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.perception import (GameMetadataReader, GlyphSegmenter,
                                GlyphTemplateClassifier, HandReader, HandSegmenter,
                                MeldReader, RiverReader, RiverSegmenter,
                                SeatLayoutResolver, TileRegionReader,
                                TileTemplateClassifier, VisibleTilesReader,
                                VisionConfig, extract_roi, rotate_roi,
                                save_template)
from majsoul.perception.roi_extractor import ROICrop
from majsoul.perception.schemas import SeatReading, SegmentedTile
from majsoul.perception.metadata_reader import seat_consistency_issues
from majsoul.domain import Box


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def roi_config():
    return ROIConfig.load(ROOT / "configs/roi.yaml")


@pytest.fixture(scope="module")
def vision_config():
    return VisionConfig.load(ROOT / "configs/vision.yaml")


@pytest.mark.parametrize(("path", "hand_count", "drawn_index", "river_count"), [
    ("artifacts/roi/review/frame_300.png", 14, 13, 1),
    ("artifacts/roi/review/frame_1590.png", 10, None, 10),
    ("artifacts/roi/survey/frame_8100.png", 13, None, 6),
])
def test_real_frame_hand_and_bottom_river_segmentation(
        path, hand_count, drawn_index, river_count, roi_config, vision_config):
    image = read_image(ROOT / path)
    hand = HandSegmenter(vision_config).segment(extract_roi(image, roi_config, "self_hand"))
    river = RiverSegmenter(vision_config).segment(extract_roi(image, roi_config, "river_bottom"))
    assert len(hand.tiles) == hand_count
    assert hand.drawn_index == drawn_index
    assert not hand.issues
    assert len(river.tiles) == river_count
    assert not river.issues
    assert [tile.box.x1 for tile in hand.tiles] == sorted(tile.box.x1 for tile in hand.tiles)
    assert [(tile.row, tile.column) for tile in river.tiles] == sorted(
        (tile.row, tile.column) for tile in river.tiles)


def test_real_animation_is_reported_as_occlusion(roi_config, vision_config):
    image = read_image(ROOT / "artifacts/roi/survey/frame_6300.png")
    hand = HandSegmenter(vision_config).segment(extract_roi(image, roi_config, "self_hand"))
    assert len(hand.tiles) == 7
    assert any("animation or occlusion" in issue for issue in hand.issues)


def test_real_action_overlay_is_not_appended_to_river(roi_config, vision_config):
    image = read_image(ROOT / "artifacts/perception/phase2/horizontal/frame_6750.png")
    river = RiverSegmenter(vision_config).segment(
        extract_roi(image, roi_config, "river_bottom"))
    assert len(river.tiles) == 13
    assert all(tile.row < 3 for tile in river.tiles)
    assert any("Off-grid" in issue for issue in river.issues)


def test_template_classifier_uses_real_tile_faces_and_rejects_unknown(
        tmp_path, roi_config, vision_config):
    image = read_image(ROOT / "artifacts/roi/review/frame_300.png")
    result = HandSegmenter(vision_config).segment(extract_roi(image, roi_config, "self_hand"))
    # Tiles 1 and 2 are two separately rendered copies of 5p in the real frame.
    save_template(tmp_path, "5p", "frame300_a", result.tiles[1].image)
    # The 1s template is visually distinct and supplies a real competing class.
    save_template(tmp_path, "1s", "frame300", result.tiles[4].image)
    classifier = TileTemplateClassifier(tmp_path, vision_config)
    prediction = classifier.classify(result.tiles[2])
    assert prediction.tile == "5p"
    assert prediction.confidence > 0
    noise = np.random.default_rng(7).integers(0, 256, result.tiles[2].image.shape, dtype=np.uint8)
    unknown = classifier.classify(SegmentedTile(Box(0, 0, 93, 146), noise))
    assert unknown.tile is None
    assert unknown.candidates
    assert unknown.candidates[0][1] >= unknown.candidates[-1][1]


@pytest.mark.parametrize("path", [
    "artifacts/roi/review/frame_1575.png",
    "artifacts/roi/review/frame_1605.png",
])
def test_gold_classifier_recognizes_stable_adjacent_frames(
        path, roi_config, vision_config, gold_tile_classifier):
    image = read_image(ROOT / path)
    classifier = gold_tile_classifier
    hand = HandReader(vision_config, classifier).read(
        extract_roi(image, roi_config, "self_hand"))
    river = RiverReader(vision_config, classifier).read(
        extract_roi(image, roi_config, "river_bottom"))
    # Rechecked against source pixels, not inferred from the classifier:
    # hand starts with two six-pin; bottom river's ninth tile is eight-pin.
    assert hand.labels == ("6p", "6p", "4s", "0s", "5s", "6s", "7s", "7s", "8s", "6z")
    assert river.labels == ("1m", "1p", "1s", "7p", "4p", "3z", "2z", "1m", "8p", "7z")
    assert hand.complete
    assert river.complete


def test_classifier_normalizes_real_rotated_and_small_tile(
        roi_config, vision_config):
    image = read_image(ROOT / "artifacts/roi/review/frame_1590.png")
    classifier = TileTemplateClassifier(ROOT / "artifacts/templates/tiles", vision_config)
    river = RiverSegmenter(vision_config).segment(
        extract_roi(image, roi_config, "river_bottom"))
    # A river crop is much smaller than the hand template for 1p.
    assert classifier.classify(river.tiles[1]).tile == "1p"
    source = river.tiles[3]
    rotated = SegmentedTile(
        source.box,
        cv2.rotate(source.image, cv2.ROTATE_90_CLOCKWISE),
        source.region,
        source.row,
        source.column,
        "horizontal",
    )
    prediction = classifier.classify(rotated)
    assert prediction.tile == "7p"
    assert prediction.orientation.startswith("horizontal:")


def test_all_active_rivers_are_normalized_and_mapped_to_screen(
        roi_config, vision_config, gold_tile_classifier):
    image = read_image(ROOT / "artifacts/roi/review/frame_1605.png")
    classifier = gold_tile_classifier
    reader = RiverReader(vision_config, classifier)
    expected = {
        "bottom": ("1m", "1p", "1s", "7p", "4p", "3z", "2z", "1m", "8p", "7z"),
        # Top river's final tile has three columns of two bamboo (six-sou).
        "top": ("1m", "9m", "9p", "1p", "2p", "5p", "7z", "3z", "1z", "1z", "6s"),
        "left": ("2z", "3z", "6z", "1p", "2p", "3p", "5p", "8s", "4p"),
    }
    for slot, labels in expected.items():
        crop = extract_roi(image, roi_config, f"river_{slot}")
        reading = reader.read(crop, vision_config.rotation_for_slot(slot))
        assert reading.labels == labels
        assert reading.complete
        assert all(crop.box.x1 <= item.detection.box.x1 < item.detection.box.x2 <= crop.box.x2
                   and crop.box.y1 <= item.detection.box.y1 < item.detection.box.y2 <= crop.box.y2
                   for item in reading.tiles)


@pytest.mark.parametrize("degrees", [-90, 0, 90, 180])
def test_rotated_roi_boxes_round_trip_to_screen(degrees):
    source = ROICrop("test", Box(100, 200, 104, 203),
                     np.zeros((3, 4, 3), dtype=np.uint8))
    oriented = rotate_roi(source, degrees)
    height, width = oriented.image.shape[:2]
    mapped = oriented.map_box_to_screen(Box(0, 0, width, height))
    assert mapped == source.box


def test_dora_and_kita_readers_use_real_rotated_regions(roi_config, vision_config):
    image = read_image(ROOT / "artifacts/roi/review/frame_1590.png")
    classifier = TileTemplateClassifier(ROOT / "artifacts/templates/tiles", vision_config)
    dora = TileRegionReader(vision_config, classifier, vision_config.dora).read(
        extract_roi(image, roi_config, "dora_indicators"))
    assert dora.labels == ("7s", "6z")
    assert dora.complete
    for slot, labels in {"bottom": ("4z",), "top": ("4z",), "left": ()}.items():
        kita = TileRegionReader(
            vision_config, classifier, vision_config.kita_for_slot(slot), "4z").read(
                extract_roi(image, roi_config, f"kita_{slot}"),
                vision_config.rotation_for_slot(slot))
        assert kita.labels == labels
        assert kita.complete


def test_kita_reader_rejects_animation_contamination(roi_config, vision_config):
    image = read_image(ROOT / "artifacts/roi/survey/frame_6300.png")
    classifier = TileTemplateClassifier(ROOT / "artifacts/templates/tiles", vision_config)
    kita = TileRegionReader(
        vision_config, classifier, vision_config.kita_for_slot("bottom"), "4z").read(
            extract_roi(image, roi_config, "kita_bottom"),
            vision_config.rotation_for_slot("bottom"))
    assert not kita.complete
    assert any("animation or occlusion" in issue for issue in kita.issues)


def test_real_melds_are_grouped_after_slot_normalization(roi_config, vision_config):
    image = read_image(ROOT / "artifacts/roi/review/frame_1590.png")
    classifier = TileTemplateClassifier(ROOT / "artifacts/templates/tiles", vision_config)
    expected = {
        "bottom": (("9s", "9s", "9s", "9s"),),
        "top": (("5z", "5z", "5z"),),
        "left": (),
    }
    for slot, groups in expected.items():
        reading = MeldReader(vision_config, classifier, slot).read(
            extract_roi(image, roi_config, f"melds_{slot}"))
        assert reading.labels == groups
        assert reading.complete


def test_meld_reader_rejects_action_animation(roi_config, vision_config):
    image = read_image(ROOT / "artifacts/roi/survey/frame_6300.png")
    classifier = TileTemplateClassifier(ROOT / "artifacts/templates/tiles", vision_config)
    reading = MeldReader(vision_config, classifier, "bottom").read(
        extract_roi(image, roi_config, "melds_bottom"))
    assert not reading.complete
    assert any("animation or occlusion" in issue for issue in reading.issues)


def test_complete_visible_tile_observation_uses_locked_layout(
        roi_config, vision_config, gold_tile_classifier):
    image = read_image(ROOT / "artifacts/roi/review/frame_1590.png")
    classifier = gold_tile_classifier
    reading = VisibleTilesReader(vision_config, classifier).read(
        image, roi_config, ("bottom", "top", "left"))
    assert reading.complete
    assert reading.issues == ()
    assert reading.active_slots == ("bottom", "top", "left")
    assert reading.hand.labels == (
        "6p", "6p", "4s", "0s", "5s", "6s", "7s", "7s", "8s", "6z")
    assert reading.dora.labels == ("7s", "6z")
    assert reading.melds["bottom"].labels == (("9s", "9s", "9s", "9s"),)
    assert reading.melds["top"].labels == (("5z", "5z", "5z"),)
    assert {slot: len(river.tiles) for slot, river in reading.rivers.items()} == {
        "bottom": 10, "top": 11, "left": 9}
    with pytest.raises(ValueError, match="valid player layout"):
        VisibleTilesReader(vision_config, classifier).read(
            image, roi_config, ("bottom", "top"))


@pytest.mark.parametrize(("path", "round_id", "remaining", "riichi", "scores", "seats"), [
    ("artifacts/roi/review/frame_300.png", "1z:2:0", 48, 0,
     {"bottom": 42000, "top": 30000, "left": 33000},
     {"bottom": ("2z", False), "top": ("3z", False), "left": ("1z", True)}),
    ("artifacts/roi/review/frame_1590.png", "1z:2:0", 21, 1,
     {"bottom": 42000, "top": 30000, "left": 32000},
     {"bottom": ("2z", False), "top": ("3z", False), "left": ("1z", True)}),
    ("artifacts/roi/survey/frame_2700.png", "1z:2:1", 49, 0,
     {"bottom": 40000, "top": 28000, "left": 37000},
     {"bottom": ("2z", False), "top": ("3z", False), "left": ("1z", True)}),
    ("artifacts/roi/survey/frame_6300.png", "1z:3:0", 24, 2,
     {"bottom": 23800, "top": 43200, "left": 36000},
     {"bottom": ("1z", True), "top": ("2z", False), "left": ("3z", False)}),
    ("artifacts/roi/survey/frame_8100.png", "1z:3:1", 37, 0,
     {"bottom": 37800, "top": 37200, "left": 30000},
     {"bottom": ("1z", True), "top": ("2z", False), "left": ("3z", False)}),
])
def test_real_metadata_perception_across_rounds(
        path, round_id, remaining, riichi, scores, seats, roi_config, vision_config):
    image = read_image(ROOT / path)
    classifier = GlyphTemplateClassifier(ROOT / "artifacts/templates/glyphs", vision_config)
    reading = GameMetadataReader(vision_config, classifier).read(
        image, roi_config, ("bottom", "top", "left"))
    assert reading.complete
    assert reading.round_id == round_id
    assert reading.remaining_tiles.value == remaining
    assert reading.riichi_sticks.value == riichi
    assert {slot: value.value for slot, value in reading.scores.items()} == scores
    assert {slot: (value.wind, value.dealer)
            for slot, value in reading.seats.items()} == seats


def test_real_metadata_resolves_and_locks_unique_match_layout(
        roi_config, vision_config):
    image = read_image(ROOT / "artifacts/roi/review/frame_1590.png")
    classifier = GlyphTemplateClassifier(ROOT / "artifacts/templates/glyphs", vision_config)
    resolver = SeatLayoutResolver(
        roi_config, GameMetadataReader(vision_config, classifier),
        starting_points=35000, confirmations_required=2)

    first = resolver.observe(image)
    assert first.candidates == (("bottom", "top", "left"),)
    assert first.pending_layout == ("bottom", "top", "left")
    assert first.pending_count == 1
    assert first.locked_layout is None

    second = resolver.observe(image)
    assert second.locked_layout == ("bottom", "top", "left")
    # Later hands reuse the match-level lock instead of assigning slots again.
    later = read_image(ROOT / "artifacts/roi/survey/frame_6300.png")
    assert resolver.observe(later).locked_layout == ("bottom", "top", "left")


def test_ambiguous_layout_evidence_cannot_bridge_confirmation_streak(roi_config):
    class MetadataStub:
        accepted = {("bottom", "top", "left")}

        def read(self, _image, _roi, active_slots):
            layout = tuple(active_slots)
            if layout not in self.accepted:
                return SimpleNamespace(active_slots=layout, complete=False)
            seats = {
                slot: SimpleNamespace(wind=wind, dealer=index == 0)
                for index, (slot, wind) in enumerate(
                    zip(layout, ("1z", "2z", "3z")))
            }
            scores = {slot: SimpleNamespace(value=35000) for slot in layout}
            return SimpleNamespace(
                active_slots=layout,
                complete=True,
                seats=seats,
                scores=scores,
                round=SimpleNamespace(hand_number=1),
                remaining_tiles=SimpleNamespace(value=55),
                honba=SimpleNamespace(value=0),
                riichi_sticks=SimpleNamespace(value=0),
            )

    reader = MetadataStub()
    resolver = SeatLayoutResolver(
        roi_config, reader, starting_points=35000, confirmations_required=2)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    assert resolver.observe(image).pending_count == 1

    reader.accepted = {tuple(layout) for layout in roi_config.layout_variants}
    ambiguous = resolver.observe(image)
    assert len(ambiguous.candidates) == 3
    assert ambiguous.pending_layout is None
    assert ambiguous.pending_count == 0
    assert ambiguous.locked_layout is None

    reader.accepted = {("bottom", "top", "left")}
    assert resolver.observe(image).locked_layout is None
    assert resolver.observe(image).locked_layout == ("bottom", "top", "left")


@pytest.mark.parametrize("path", [
    "artifacts/roi/river-review/frame_4230.png",
    "artifacts/roi/river-review/frame_6870.png",
])
def test_metadata_reader_rejects_transition_and_animation_frames(
        path, roi_config, vision_config):
    image = read_image(ROOT / path)
    classifier = GlyphTemplateClassifier(ROOT / "artifacts/templates/glyphs", vision_config)
    reading = GameMetadataReader(vision_config, classifier).read(
        image, roi_config, ("bottom", "top", "left"))
    assert not reading.complete
    assert not reading.round.complete
    assert any(not score.complete for score in reading.scores.values())


def test_layout_locks_with_scores_unavailable(roi_config, vision_config):
    image = read_image(ROOT / "artifacts/roi/review/frame_1590.png")
    classifier = GlyphTemplateClassifier(ROOT / "artifacts/templates/glyphs", vision_config)
    reader = GameMetadataReader(vision_config, classifier, read_scores=False)
    reading = reader.read(image, roi_config, ("bottom", "top", "left"))
    assert not reading.scores_available
    assert all(value.value is None for value in reading.scores.values())
    assert all(not value.issues for value in reading.scores.values())
    assert reading.complete

    resolver = SeatLayoutResolver(
        roi_config, reader, starting_points=35000, confirmations_required=2)
    first = resolver.observe(image)
    assert first.candidates == (("bottom", "top", "left"),)
    second = resolver.observe(image)
    assert second.locked_layout == ("bottom", "top", "left")


def test_glyph_segmenter_splits_touching_rotated_score_digits(
        roi_config, vision_config):
    image = read_image(ROOT / "artifacts/roi/review/frame_1590.png")
    crop = rotate_roi(extract_roi(image, roi_config, "score_left"), -90)
    glyphs = GlyphSegmenter(vision_config).segment(crop, "score", split_wide=True)
    classifier = GlyphTemplateClassifier(ROOT / "artifacts/templates/glyphs", vision_config)
    labels = tuple(classifier.classify(
        glyph.mask, glyph.box, allowed_labels=set("0123456789")).label
                   for glyph in glyphs)
    assert labels == tuple("32000")
    assert all(crop.box.x1 <= glyph.box.x1 < glyph.box.x2 <= crop.box.x2
               and crop.box.y1 <= glyph.box.y1 < glyph.box.y2 <= crop.box.y2
               for glyph in glyphs)


def test_single_semantic_glyph_merges_disconnected_strokes(vision_config):
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    crop = ROICrop("seat_top", Box(100, 200, 180, 260), image)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[10:52, 12:27] = 255
    mask[8:50, 42:61] = 255

    glyph = GlyphSegmenter(vision_config).segment_merged(
        crop, mask, vision_config.glyph_styles["seat"])

    assert glyph is not None
    assert glyph.box == Box(112, 208, 161, 252)
    assert cv2.connectedComponents(glyph.mask)[0] == 3


def test_known_duplicate_seat_winds_are_rejected_before_all_seats_are_complete():
    seats = {
        "bottom": SeatReading("bottom", "2z", False, None, 0.0),
        "top": SeatReading("top", None, False, None, 0.0,
                           ("Seat wind glyph is unknown",)),
        "left": SeatReading("left", "1z", True, None, 0.3),
        "right": SeatReading("right", "2z", False, None, 0.0),
    }

    issues = seat_consistency_issues(seats, 4)

    assert "Active layout contains duplicate known seat winds" in issues


def test_vision_config_and_roi_extraction_reject_bad_inputs(tmp_path, roi_config, vision_config):
    invalid = tmp_path / "vision.yaml"
    invalid.write_text("schema_version: 99\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        VisionConfig.load(invalid)
    with pytest.raises(KeyError, match="Unknown ROI"):
        extract_roi(np.zeros((1156, 1920, 3), dtype=np.uint8), roi_config, "not_a_region")
    with pytest.raises(ValueError, match="non-empty"):
        extract_roi(np.empty((0, 0, 3), dtype=np.uint8), roi_config, "self_hand")
