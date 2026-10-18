import random
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from majsoul.perception.tile_augment import (PRESETS, TileAugment, TileAugmentConfig,
                                             _apply_gamma, apply_roi_shift)

GEOMETRY_OPS = {"roi_shift", "edge_crop"}
CLARITY_OPS = {"downsample", "gaussian_blur", "motion_blur", "sharpen"}
COLOR_OPS = {"color_jitter", "gamma"}


def noise_image(width, height, seed=0):
    gen = np.random.default_rng(seed)
    arr = gen.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def run_calls(aug, img, n, seed=0):
    rng = random.Random(seed)
    return [aug(img, rng=rng)[1] for _ in range(n)]


def test_skip_probability_is_20_percent():
    # noise forced to 1.0 makes every augmented call produce an op,
    # so an empty ops list happens exactly on the 20% skip branch.
    cfg = replace(TileAugmentConfig(), noise_probability=1.0)
    ops_list = run_calls(TileAugment(cfg), noise_image(64, 96), 20000)
    p_empty = sum(1 for ops in ops_list if not ops) / len(ops_list)
    assert p_empty == pytest.approx(0.2, abs=0.01)


def test_conditional_branch_probabilities():
    # augment_probability=1.0 makes the conditional probabilities directly observable.
    cfg = replace(TileAugmentConfig(), augment_probability=1.0)
    ops_list = run_calls(TileAugment(cfg), noise_image(64, 96), 20000)
    n = len(ops_list)

    geom = {"roi_shift": 0, "edge_crop": 0, "none": 0}
    clarity = {"downsample": 0, "gaussian_blur": 0, "motion_blur": 0, "sharpen": 0, "none": 0}
    color = {"color_jitter": 0, "gamma": 0, "none": 0}
    noise = 0
    for ops in ops_list:
        names = [op["op"] for op in ops]
        assert len(set(names) & GEOMETRY_OPS) <= 1
        assert len(set(names) & CLARITY_OPS) <= 1
        assert len(set(names) & COLOR_OPS) <= 1
        g = set(names) & GEOMETRY_OPS
        geom[g.pop() if g else "none"] += 1
        c = set(names) & CLARITY_OPS
        clarity[c.pop() if c else "none"] += 1
        col = set(names) & COLOR_OPS
        color[col.pop() if col else "none"] += 1
        noise += "noise" in names

    assert geom["roi_shift"] / n == pytest.approx(0.30, abs=0.02)
    assert geom["edge_crop"] / n == pytest.approx(0.15, abs=0.02)
    assert geom["none"] / n == pytest.approx(0.55, abs=0.02)
    assert clarity["downsample"] / n == pytest.approx(0.25, abs=0.02)
    assert clarity["gaussian_blur"] / n == pytest.approx(0.10, abs=0.02)
    assert clarity["motion_blur"] / n == pytest.approx(0.03, abs=0.02)
    assert clarity["sharpen"] / n == pytest.approx(0.05, abs=0.02)
    assert clarity["none"] / n == pytest.approx(0.57, abs=0.02)
    assert color["color_jitter"] / n == pytest.approx(0.50, abs=0.02)
    assert color["gamma"] / n == pytest.approx(0.15, abs=0.02)
    assert color["none"] / n == pytest.approx(0.35, abs=0.02)
    assert noise / n == pytest.approx(0.10, abs=0.02)


def test_preset_b_disables_ops_without_rescaling():
    cfg = PRESETS["B"]
    for field_name in ("roi_shift_probability", "edge_crop_probability",
                       "small_rotation_probability", "gaussian_blur_probability",
                       "motion_blur_probability", "sharpen_probability",
                       "gamma_probability", "noise_probability",
                       "right_angle_probability"):
        assert getattr(cfg, field_name) == 0.0
    assert cfg.downsample_probability == 0.25
    assert cfg.color_jitter_probability == 0.50

    forced = replace(cfg, augment_probability=1.0)
    ops_list = run_calls(TileAugment(forced), noise_image(64, 96), 20000)
    n = len(ops_list)
    seen = {op["op"] for ops in ops_list for op in ops}
    assert seen <= {"downsample", "color_jitter"}
    downsample = sum(1 for ops in ops_list if any(op["op"] == "downsample" for op in ops))
    jitter = sum(1 for ops in ops_list if any(op["op"] == "color_jitter" for op in ops))
    assert downsample / n == pytest.approx(0.25, abs=0.02)
    assert jitter / n == pytest.approx(0.50, abs=0.02)


def test_with_disabled_validates_names():
    with pytest.raises(ValueError):
        TileAugmentConfig().with_disabled(frozenset({"not_an_op"}))


def test_reproducibility():
    aug = TileAugment(TileAugmentConfig())
    img = noise_image(50, 80)
    out1, ops1 = aug(img, rng=random.Random(42))
    out2, ops2 = aug(img, rng=random.Random(42))
    assert out1.tobytes() == out2.tobytes()
    assert ops1 == ops2

    noisy = TileAugment(replace(TileAugmentConfig(), noise_probability=1.0))
    out_a, _ = noisy(img, rng=random.Random(1))
    out_b, _ = noisy(img, rng=random.Random(2))
    assert out_a.tobytes() != out_b.tobytes()


def test_output_size_preserved_without_right_angle():
    cfg = replace(TileAugmentConfig(), augment_probability=1.0)
    aug = TileAugment(cfg)
    rng = random.Random(0)
    img = noise_image(53, 91)
    for _ in range(200):
        out, _ = aug(img, rng=rng)
        assert out.size == img.size
        assert out.mode == "RGB"


def test_preset_f_right_angle_swaps_axes():
    cfg = replace(PRESETS["F"], augment_probability=1.0)
    assert cfg.right_angle_probability == 1.0
    aug = TileAugment(cfg)
    rng = random.Random(0)
    img = noise_image(53, 91)
    swapped = 0
    for _ in range(200):
        out, ops = aug(img, rng=rng)
        rot = [op for op in ops if op["op"] == "right_angle"]
        assert len(rot) == 1
        if rot[0]["degrees"] in (90, 270):
            assert out.size == (img.height, img.width)
            swapped += 1
        else:
            assert out.size == img.size
    assert swapped > 0


def test_pixel_scale_relative_to_short_side():
    cfg = replace(TileAugmentConfig(), augment_probability=1.0,
                  roi_shift_probability=0.0, edge_crop_probability=0.0,
                  small_rotation_probability=0.0, downsample_probability=0.0,
                  gaussian_blur_probability=1.0, motion_blur_probability=0.0,
                  sharpen_probability=0.0, color_jitter_probability=0.0,
                  gamma_probability=0.0, noise_probability=0.0)
    aug = TileAugment(cfg)
    _, ops32 = aug(noise_image(32, 64), rng=random.Random(7))
    _, ops128 = aug(noise_image(128, 256), rng=random.Random(7))
    assert ops32[0]["op"] == ops128[0]["op"] == "gaussian_blur"
    assert ops128[0]["sigma_px"] == pytest.approx(4.0 * ops32[0]["sigma_px"])


def make_canvas_with_stripe():
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    canvas[:, 60, :] = 255
    canvas_img = Image.fromarray(canvas, "RGB")
    crop = canvas_img.crop((10, 10, 90, 90))
    return canvas_img, crop


def stripe_column(img):
    return int(np.asarray(img).mean(axis=(0, 2)).argmax())


def test_roi_shift_with_canvas_moves_content():
    canvas, crop = make_canvas_with_stripe()
    assert stripe_column(crop) == 50
    shifted, op = apply_roi_shift(crop, 5.0, 0.0, canvas=canvas)
    assert op["mode"] == "canvas" and not op["clamped"]
    assert stripe_column(shifted) == 45
    shifted, op = apply_roi_shift(crop, -5.0, 0.0, canvas=canvas)
    assert stripe_column(shifted) == 55


def test_roi_shift_clamped_at_canvas_edge():
    canvas, crop = make_canvas_with_stripe()
    shifted, op = apply_roi_shift(crop, 100.0, 0.0, canvas=canvas)
    assert op["clamped"] is True
    assert shifted.size == crop.size
    assert stripe_column(shifted) == 40  # window clamped to x=20


def test_roi_shift_in_crop_approx():
    _, crop = make_canvas_with_stripe()
    shifted, op = apply_roi_shift(crop, 5.0, 0.0, canvas=None)
    assert op["mode"] == "in_crop_approx"
    assert shifted.size == crop.size
    assert stripe_column(shifted) == 45


def test_edge_crop_axis_loss_bounded():
    cfg = replace(TileAugmentConfig(), augment_probability=1.0,
                  roi_shift_probability=0.0, edge_crop_probability=1.0,
                  small_rotation_probability=0.0, downsample_probability=0.0,
                  gaussian_blur_probability=0.0, motion_blur_probability=0.0,
                  sharpen_probability=0.0, color_jitter_probability=0.0,
                  gamma_probability=0.0, noise_probability=0.0)
    aug = TileAugment(cfg)
    rng = random.Random(0)
    img = noise_image(80, 120)
    for _ in range(1000):
        out, ops = aug(img, rng=rng)
        (op,) = ops
        assert op["op"] == "edge_crop"
        assert (op["left_px"] + op["right_px"]) / img.width <= 0.04 + 1e-9
        assert (op["top_px"] + op["bottom_px"]) / img.height <= 0.04 + 1e-9
        assert out.size == img.size


def test_gamma_identity_at_one():
    img = noise_image(40, 64)
    out, op = _apply_gamma(img, 1.0)
    assert op == {"op": "gamma", "gamma": 1.0}
    assert out.tobytes() == img.tobytes()


def test_zero_augment_probability_never_augments():
    cfg = replace(TileAugmentConfig(), augment_probability=0.0)
    aug = TileAugment(cfg)
    rng = random.Random(0)
    img = noise_image(40, 64)
    for _ in range(100):
        out, ops = aug(img, rng=rng)
        assert ops == []
        assert out.tobytes() == img.tobytes()


def test_disabled_config_never_augments():
    aug = TileAugment(TileAugmentConfig(enabled=False))
    rng = random.Random(0)
    img = noise_image(40, 64)
    for _ in range(100):
        out, ops = aug(img, rng=rng)
        assert ops == []
        assert out.tobytes() == img.tobytes()


def test_preset_a_is_deterministic_passthrough():
    aug = TileAugment(PRESETS["A"])
    out, ops = aug(noise_image(40, 64), rng=random.Random(0))
    assert ops == []
