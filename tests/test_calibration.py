import cv2
import numpy as np
import pytest
from pathlib import Path

from majsoul.calibration.debug import export_debug, export_projection_trials
from majsoul.calibration.frames import extract_frame, read_image, write_image
from majsoul.calibration.roi_config import ROIConfig, default_layout, required_regions
from majsoul.calibration.roi_editor import edit_rois, select_region
from majsoul.calibration.roi_gui import (describe_region, fit_image, mirror_horizontal,
                                         normalize_canvas_rect, opposite_side_region)
from majsoul.calibration.seat_layout import SeatLayoutTracker


def config():
    slots, variants, self_slot = default_layout(3)
    return ROIConfig(reference_size=[200, 100], player_count=3,
                     screen_slots=slots, layout_variants=variants,
                     self_slot=self_slot, kita=True,
                     regions={"self_hand": [0.1, 0.6, 0.9, 1.0]})


@pytest.mark.parametrize("rect", [[0, 0, 0, 1], [-0.1, 0, 1, 1],
                                     [0, 0, float('nan'), 1], [0, 0, 1.1, 1]])
def test_reject_invalid_roi(rect):
    item = config()
    item.regions["self_hand"] = rect
    with pytest.raises(ValueError):
        item.validate()


def test_dynamic_rois_and_unknown():
    slots, _, _ = default_layout(3)
    assert "river_right" in required_regions(slots, False)
    assert "kita_bottom" not in required_regions(slots, False)
    assert "drawn_tile" not in required_regions(slots, True)
    assert "dealer_bottom" not in required_regions(slots, True)
    item = config()
    assert "river_bottom" in item.validate()
    with pytest.raises(ValueError, match="Uncalibrated"):
        item.validate(complete=True)


def test_roundtrip_scaling_and_crops(tmp_path):
    item = config()
    path = tmp_path / "roi.yaml"
    item.save(path)
    loaded = ROIConfig.load(path)
    assert loaded.pixel_boxes((400, 200))["self_hand"] == (40, 120, 360, 200)
    # A materially different recording range is never selected silently.
    with pytest.raises(ValueError, match="projection all"):
        loaded.pixel_boxes((400, 300))
    image = np.arange(400 * 200 * 3, dtype=np.uint8).reshape(200, 400, 3)
    report = export_debug(image, loaded, tmp_path / "debug")
    crop = read_image(tmp_path / "debug/crops/self_hand.png")
    np.testing.assert_array_equal(crop, image[120:200, 40:360])
    assert report["uncalibrated"]
    assert (tmp_path / "debug/roi_debug.png").exists()


def test_small_aspect_drift_uses_independent_normalized_xy_scaling():
    item = config()
    # 202x100 differs by 1%, below the configurable 3% automatic limit.
    assert item.assess_target_size((202, 100)).status == "normalized_xy"
    assert item.pixel_boxes((202, 100))["self_hand"] == (20, 60, 182, 100)


def test_large_aspect_drift_exports_distinct_projection_trials(tmp_path):
    item = config()
    image = np.zeros((100, 240, 3), dtype=np.uint8)
    report = export_projection_trials(image, item, tmp_path / "trials")

    assert report["target_size_assessment"]["status"] == "review_required"
    assert report["requires_human_or_recognition_review"] is True
    assert len(report["trials"]) > 1
    for trial in report["trials"]:
        assert (Path(trial["output"]) / "roi_debug.png").is_file()
    assert (tmp_path / "trials/projection_trials.json").is_file()


def test_debug_exports_only_regions_for_locked_layout(tmp_path):
    item = config()
    item.regions.update({
        "river_bottom": [0.2, 0.2, 0.3, 0.3],
        "river_top": [0.3, 0.2, 0.4, 0.3],
        "river_left": [0.1, 0.2, 0.2, 0.3],
        "river_right": [0.8, 0.2, 0.9, 0.3],
    })
    report = export_debug(np.zeros((100, 200, 3), dtype=np.uint8), item,
                          tmp_path / "active", active_slots=["bottom", "top", "left"])
    assert report["active_layout"] == ["bottom", "top", "left"]
    assert "river_right" in report["inactive_regions"]
    assert "river_right" not in report["pixel_boxes"]
    assert not (tmp_path / "active/crops/river_right.png").exists()


def test_parent_hand_and_meld_roi_may_overlap():
    item = config()
    item.regions["melds_bottom"] = [0.7, 0.6, 1.0, 1.0]
    item.validate()
    boxes = item.pixel_boxes((200, 100))
    assert boxes["self_hand"] == (20, 60, 180, 100)
    assert boxes["melds_bottom"] == (140, 60, 200, 100)


def test_editor_unknown_and_custom_name(monkeypatch):
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    selections = iter([(10, 60, 180, 40), (0, 0, 0, 0), (10, 10, 20, 20)])
    answers = iter(["timer", ""])
    item = edit_rois(np.zeros((100, 200, 3), dtype=np.uint8), player_count=3,
                     kita=True, names=["self_hand", "melds_bottom"],
                     selector=lambda *a, **kw: next(selections), ask=lambda _: next(answers))
    assert item.regions["self_hand"] == [0.05, 0.6, 0.95, 1.0]
    assert item.regions["melds_bottom"] is None
    assert item.regions["timer"] == [0.05, 0.1, 0.15, 0.3]


def test_unicode_images_and_video_exact_frame(tmp_path):
    image = np.full((48, 64, 3), 100, dtype=np.uint8)
    path = tmp_path / "参考.png"
    write_image(path, image)
    np.testing.assert_array_equal(read_image(path), image)
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for value in (30, 100, 200):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()
    frame, metadata = extract_frame(video, 1)
    assert abs(float(frame.mean()) - 100) < 2
    assert metadata["frame_index"] == 1
    assert metadata["seconds"] == 0.1
    with pytest.raises(ValueError, match="ended"):
        extract_frame(video, 5)


def test_preview_selection_maps_to_original_resolution():
    image = np.zeros((1200, 1920, 3), dtype=np.uint8)
    def select(title, preview, **kwargs):
        assert preview.shape[:2] == (600, 960)
        return (120, 300, 600, 300)
    rect = select_region("self_hand", image, select, (960, 600))
    item = config()
    item.reference_size = [1920, 1200]
    item.regions = {"self_hand": rect}
    assert item.pixel_boxes((1920, 1200))["self_hand"] == (240, 600, 1440, 1200)
    with pytest.raises(ValueError, match="preview_size"):
        select_region("self_hand", image, select, (0, 600))


def test_gui_geometry_maps_drag_to_normalized_image():
    scale, offset, shown = fit_image((200, 100), (100, 100))
    assert shown == (100, 50)
    assert offset == (0, 25)
    rect = normalize_canvas_rect((10, 30), (90, 70), (200, 100), scale, offset)
    assert rect == pytest.approx([0.1, 0.1, 0.9, 0.9])


def test_gui_geometry_clamps_and_rejects_tiny_drag():
    assert normalize_canvas_rect((-10, -20), (110, 70), (100, 50), 1, (0, 0)) == [0, 0, 1, 1]
    assert normalize_canvas_rect((10, 10), (11, 11), (100, 50), 1, (0, 0)) is None
    with pytest.raises(ValueError, match="positive"):
        fit_image((100, 50), (0, 100))
    assert "screen slot left" in describe_region("melds_left")


def test_gui_mirrors_left_and_right_screen_slots():
    slots, _, _ = default_layout(3)
    assert opposite_side_region("melds_left", slots) == "melds_right"
    assert opposite_side_region("river_right", slots) == "river_left"
    assert opposite_side_region("self_hand", slots) is None
    assert mirror_horizontal([0.1, 0.2, 0.3, 0.4]) == pytest.approx([0.7, 0.2, 0.9, 0.4])


def test_detected_slots_select_one_sanma_layout():
    item = config()
    assert item.layout_for_active_slots(["left", "bottom", "top"]) == ["bottom", "top", "left"]
    active = item.active_region_names(["bottom", "left", "right"])
    assert "self_hand" in active
    assert "river_left" not in active  # The sparse test config contains only self_hand.
    item.regions.update({"river_left": None, "river_right": None, "river_top": None})
    active = item.active_region_names(["bottom", "left", "right"])
    assert "river_left" in active
    assert "river_right" in active
    assert "river_top" not in active
    with pytest.raises(ValueError, match="valid player layout"):
        item.layout_for_active_slots(["bottom", "top"])


def test_calibrated_geometry_can_be_reused_with_explicit_four_player_mode():
    item = config().for_game_mode(4)
    assert item.player_count == 4
    assert item.layout_variants == [["bottom", "top", "left", "right"]]
    assert item.kita is False
    assert item.layout_for_active_slots(
        ["right", "left", "bottom", "top"]) == ["bottom", "top", "left", "right"]


def test_schema_v1_migrates_numeric_players_without_guessing_right(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("""reference_size: [200, 100]
player_count: 3
seat_order: [self, right, left]
kita: true
regions:
  self_hand: [0.1, 0.6, 0.9, 1.0]
  river_0: [0.4, 0.5, 0.6, 0.8]
  river_1: [0.4, 0.1, 0.6, 0.3]
  river_2: [0.2, 0.3, 0.4, 0.5]
  drawn_tile: [0.8, 0.8, 0.9, 1.0]
  dealer_0: [0.4, 0.4, 0.5, 0.5]
schema_version: 1
""", encoding="utf-8")
    item = ROIConfig.load(path)
    assert item.schema_version == 2
    assert item.regions["river_bottom"] == [0.4, 0.5, 0.6, 0.8]
    assert item.regions["river_top"] == [0.4, 0.1, 0.6, 0.3]
    assert item.regions["river_left"] == [0.2, 0.3, 0.4, 0.5]
    assert item.regions["river_right"] is None
    assert "drawn_tile" not in item.regions
    assert "dealer_0" not in item.regions


def test_seat_layout_locks_for_match_and_only_reset_allows_change():
    item = config()
    tracker = SeatLayoutTracker(item)
    assert tracker.observe(["bottom", "top", "left"]) is None
    assert tracker.observe(["left", "bottom", "top"]) == ("bottom", "top", "left")
    # Hand/round transitions do not call reset_match, so the lock is reused.
    assert tracker.observe(["bottom", "top", "left"]) == ("bottom", "top", "left")
    with pytest.raises(ValueError, match="cannot change within one match"):
        tracker.observe(["bottom", "left", "right"])
    tracker.reset_match()
    assert tracker.observe(["bottom", "left", "right"]) is None
    assert tracker.observe(["right", "left", "bottom"]) == ("bottom", "left", "right")


def test_seat_layout_requires_lock_before_selecting_regions():
    tracker = SeatLayoutTracker(config(), confirmations_required=3)
    with pytest.raises(ValueError, match="has not been locked"):
        tracker.active_region_names()
    with pytest.raises(ValueError, match="integer >= 2"):
        SeatLayoutTracker(config(), confirmations_required=1)


def test_unknown_layout_evidence_breaks_confirmation_streak():
    tracker = SeatLayoutTracker(config(), confirmations_required=2)
    expected = ("bottom", "top", "left")
    assert tracker.observe(expected) is None
    assert tracker.pending_layout == expected
    assert tracker.pending_count == 1

    assert tracker.observe_unknown() is None
    assert tracker.pending_layout is None
    assert tracker.pending_count == 0
    assert tracker.observe(expected) is None
    assert tracker.observe(expected) == expected

    # Once match-level state is locked, transient missing evidence cannot clear it.
    assert tracker.observe_unknown() == expected
    assert tracker.locked_layout == expected
