import cv2
from dataclasses import replace
import numpy as np
import pytest

from majsoul.calibration.frames import write_image
from majsoul.calibration.roi_config import ROIConfig, default_layout
from majsoul.domain import Box
from majsoul.live.buttons import ActionButtonReader, ButtonConfig


DARK = (30, 30, 30)
BRIGHT = (170, 250, 170)
INK = (20, 20, 20)
BUTTON_A = (110, 120, 180, 170)
BUTTON_B = (210, 120, 280, 170)


def roi_config():
    slots, variants, self_slot = default_layout(3)
    return ROIConfig(reference_size=[400, 200], player_count=3,
                     screen_slots=slots, layout_variants=variants,
                     self_slot=self_slot, kita=False,
                     regions={"action_buttons": [0.25, 0.5, 0.75, 1.0]})


def draw_button(image, rect, shift=(0, 0)):
    x1, y1, x2, y2 = rect
    dx, dy = shift
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), BRIGHT, -1)
    cv2.rectangle(image, (x1 + 18 + dx, y1 + 12 + dy), (x1 + 30 + dx, y1 + 38 + dy), INK, -1)
    cv2.rectangle(image, (x1 + 34 + dx, y1 + 24 + dy), (x1 + 52 + dx, y1 + 34 + dy), INK, -1)


def draw_noise_button(image, rect, seed=7):
    x1, y1, x2, y2 = rect
    rng = np.random.default_rng(seed)
    noise = rng.integers(170, 251, size=(y2 - y1, x2 - x1), dtype=np.uint8)
    image[y1:y2, x1:x2] = cv2.merge([noise, noise, noise])


def write_templates(root, image):
    x1, y1, x2, y2 = BUTTON_A
    write_image(root / "ron" / "sample.png", image[y1:y2, x1:x2])
    other = np.full((50, 70, 3), BRIGHT, dtype=np.uint8)
    cv2.rectangle(other, (10, 10), (59, 19), INK, -1)
    cv2.rectangle(other, (10, 30), (59, 39), INK, -1)
    write_image(root / "pass" / "sample.png", other)


def test_config_from_dict_defaults():
    config = ButtonConfig.from_dict({})
    assert config == ButtonConfig()
    assert config.hsv_lower == (0, 0, 150)
    assert config.hsv_upper == (179, 90, 255)
    assert config.normalized_size == (48, 48)
    assert config.maximum_distance == 0.25
    assert config.minimum_margin == 0.05


def test_config_from_dict_overrides():
    config = ButtonConfig.from_dict({
        "hsv_lower": [35, 40, 120], "hsv_upper": [90, 255, 255],
        "width_fraction": [0.1, 0.8], "height_fraction": [0.2, 0.9],
        "minimum_fill_ratio": 0.6, "normalized_size": [64, 64],
        "maximum_distance": 0.3, "minimum_margin": 0.1,
    })
    assert config.hsv_lower == (35, 40, 120)
    assert config.hsv_upper == (90, 255, 255)
    assert config.width_fraction == (0.1, 0.8)
    assert config.height_fraction == (0.2, 0.9)
    assert config.minimum_fill_ratio == 0.6
    assert config.normalized_size == (64, 64)
    assert config.maximum_distance == 0.3
    assert config.minimum_margin == 0.1


@pytest.mark.parametrize("data", [
    "buttons",
    {"hsv_lower": [0, 0]},
    {"hsv_lower": [0, 0, 300]},
    {"hsv_lower": [0, 100, 0], "hsv_upper": [179, 50, 255]},
    {"width_fraction": [0.9, 0.1]},
    {"height_fraction": [0.0, 0.5]},
    {"minimum_fill_ratio": 0},
    {"minimum_fill_ratio": 1.5},
    {"normalized_size": [48]},
    {"normalized_size": [0, 48]},
    {"maximum_distance": 0},
    {"maximum_distance": 2.5},
    {"minimum_margin": -0.1},
    {"minimum_margin": 2},
])
def test_config_from_dict_rejects_invalid(data):
    with pytest.raises(ValueError):
        ButtonConfig.from_dict(data)


def test_read_detects_known_button_and_reports_unknown(tmp_path):
    image = np.full((200, 400, 3), DARK, dtype=np.uint8)
    draw_button(image, BUTTON_A)
    draw_noise_button(image, BUTTON_B)
    write_templates(tmp_path / "templates", image)
    reader = ActionButtonReader(ButtonConfig(), tmp_path / "templates")
    assert reader.known_labels == ("pass", "ron")
    seen = []
    buttons = reader.read(image, roi_config(),
                          unknown_sink=lambda *args: seen.append(args))
    assert buttons == {"ron": Box(*BUTTON_A)}
    assert len(seen) == 1
    label, candidate, box = seen[0]
    assert label is None
    assert box == Box(*BUTTON_B)
    assert candidate.shape == (50, 70, 3)
    prediction = reader.last_predictions[box]
    assert not prediction['accepted'] and prediction['hint'] is None
    assert not prediction['clipped']
    distances = prediction['ranked_distances']
    assert {label for label,_ in distances} == {'ron','pass'}
    assert [distance for _,distance in distances] == sorted(distance for _,distance in distances)
    assert prediction['distance'] == distances[0][1]
    assert prediction['margin'] == pytest.approx(distances[1][1]-distances[0][1])


def test_read_without_candidates(tmp_path):
    scene = np.full((200, 400, 3), DARK, dtype=np.uint8)
    draw_button(scene, BUTTON_A)
    write_templates(tmp_path / "templates", scene)
    reader = ActionButtonReader(ButtonConfig(), tmp_path / "templates")
    dark = np.full((200, 400, 3), DARK, dtype=np.uint8)
    seen = []
    assert reader.read(dark, roi_config(),
                       unknown_sink=lambda *args: seen.append(args)) == {}
    assert seen == []


def test_read_with_empty_template_library(tmp_path):
    reader = ActionButtonReader(ButtonConfig(), tmp_path / "missing")
    assert reader.known_labels == ()
    image = np.full((200, 400, 3), DARK, dtype=np.uint8)
    draw_button(image, BUTTON_A)
    seen = []
    assert reader.read(image, roi_config(),
                       unknown_sink=lambda *args: seen.append(args)) == {}
    assert seen == []


def test_duplicate_label_keeps_closest_candidate(tmp_path):
    image = np.full((200, 400, 3), DARK, dtype=np.uint8)
    draw_button(image, BUTTON_A)
    draw_button(image, BUTTON_B, shift=(1, 1))
    write_templates(tmp_path / "templates", image)
    reader = ActionButtonReader(ButtonConfig(), tmp_path / "templates")
    seen = []
    buttons = reader.read(image, roi_config(),
                          unknown_sink=lambda *args: seen.append(args))
    assert buttons == {"ron": Box(*BUTTON_A)}
    assert seen == []


def test_even_exact_template_match_cannot_authorize_roi_clipped_button(tmp_path):
    image=np.full((200,400,3),DARK,dtype=np.uint8)
    draw_button(image,BUTTON_A)
    cropped_roi=replace(roi_config(),regions={'action_buttons':[.25,.65,.75,1.]})
    # An old truncated template can be an exact match. Geometry must still veto.
    write_image(tmp_path/'templates'/'ron'/'old-clipped.png',image[130:170,110:180])
    reader=ActionButtonReader(ButtonConfig(),tmp_path/'templates')
    unknowns=[]
    assert reader.read(image,cropped_roi,unknown_sink=lambda *args:unknowns.append(args))=={}
    assert len(unknowns)==1
    prediction=reader.last_predictions[unknowns[0][2]]
    assert prediction['distance']==pytest.approx(0,abs=1e-6)
    assert prediction['clipped'] and not prediction['accepted']
    assert prediction['rejection']=='ROI_boundary_clipping'
    assert reader.last_occlusion_boxes
