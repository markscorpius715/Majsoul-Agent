"""Online random augmentation for tile crops (PIL RGB, upright or horizontal).

All geometric/degradation ops run at the native ROI pixel scale; the caller
performs the 224x224 resize and Normalize afterwards. Pixel-scale-dependent
strengths are expressed against s = min(W, H) and normalized to s = 32.

Reproducibility: every stochastic decision draws from the caller-supplied
``random.Random`` in a fixed branch order (augment gate -> geometry group ->
small rotation -> clarity group -> color group -> noise -> right angle).
Dense per-pixel noise is the only op needing many samples; it derives a
``numpy`` Generator via ``np.random.default_rng(rng.getrandbits(128))`` so a
given ``random.Random`` seed still fully determines the output.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

# Fixed sampling ranges from the augmentation spec; only probabilities and the
# strength caps exposed on TileAugmentConfig are configurable.
_DOWNSAMPLE_RATIO_RANGE = (0.65, 0.95)
_GAUSSIAN_SIGMA_RANGE = (0.3, 0.6)   # multiplied by s/32 -> pixels
_MOTION_BETA_RANGE = (0.2, 0.4)
_MOTION_BASE_LENGTH = 3.0            # kernel length in px at s = 32
_SHARPEN_ALPHA_RANGE = (0.1, 0.3)
_SHARPEN_SIGMA = 0.5                 # multiplied by s/32 -> pixels
_JITTER_BRIGHTNESS_RANGE = (0.85, 1.15)
_JITTER_CONTRAST_RANGE = (0.85, 1.15)
_JITTER_SATURATION_RANGE = (0.90, 1.10)
_GAMMA_RANGE = (0.90, 1.10)
_NOISE_SIGMA_RANGE = (0.003, 0.012)  # on [0, 1] RGB

_OP_PROBABILITY_FIELDS = {
    "roi_shift": "roi_shift_probability",
    "edge_crop": "edge_crop_probability",
    "small_rotation": "small_rotation_probability",
    "downsample": "downsample_probability",
    "gaussian_blur": "gaussian_blur_probability",
    "motion_blur": "motion_blur_probability",
    "sharpen": "sharpen_probability",
    "color_jitter": "color_jitter_probability",
    "gamma": "gamma_probability",
    "noise": "noise_probability",
    "right_angle": "right_angle_probability",
}


@dataclass(frozen=True)
class TileAugmentConfig:
    enabled: bool = True
    augment_probability: float = 0.8          # per load: 20% skip all random augmentation
    # A. geometry group (mutually exclusive, conditional on entering the branch)
    roi_shift_probability: float = 0.30
    edge_crop_probability: float = 0.15       # remaining 0.55: no geometry op
    roi_shift_max_dx_fraction: float = 0.04   # dx/W ~ U(-0.04, 0.04)
    roi_shift_max_dy_fraction: float = 0.03   # dy/H ~ U(-0.03, 0.03)
    edge_crop_max_fraction: float = 0.02      # per edge U(0, 0.02) of W or H; per-axis loss <= 0.04
    # B. small-angle rotation (independent Bernoulli)
    small_rotation_probability: float = 0.20
    small_rotation_max_degrees: float = 3.0
    # C. clarity group (mutually exclusive)
    downsample_probability: float = 0.25
    gaussian_blur_probability: float = 0.10
    motion_blur_probability: float = 0.03
    sharpen_probability: float = 0.05         # remaining 0.57: untouched
    # D. color group (mutually exclusive)
    color_jitter_probability: float = 0.50
    gamma_probability: float = 0.15           # remaining 0.35: untouched
    # E. noise (independent Bernoulli)
    noise_probability: float = 0.10
    # F. right-angle rotation (separate experiment, off by default)
    right_angle_probability: float = 0.0
    right_angle_distribution: tuple = (0.4, 0.2, 0.2, 0.2)  # P(0, 90, 180, 270 deg)

    def with_disabled(self, ops: frozenset[str]) -> TileAugmentConfig:
        """Zero the listed op probabilities; the freed mass goes to the
        group's "none" outcome (other ops are not rescaled)."""
        unknown = ops - _OP_PROBABILITY_FIELDS.keys()
        if unknown:
            raise ValueError(f"Unknown ops for with_disabled: {sorted(unknown)}")
        return replace(self, **{_OP_PROBABILITY_FIELDS[op]: 0.0 for op in ops})


_D = TileAugmentConfig()
PRESETS = {
    "A": replace(_D, enabled=False),
    "B": _D.with_disabled(frozenset({
        "roi_shift", "edge_crop", "small_rotation", "gaussian_blur",
        "motion_blur", "sharpen", "gamma", "noise", "right_angle",
    })),
    "C": _D.with_disabled(frozenset({
        "small_rotation", "gaussian_blur", "motion_blur", "sharpen",
        "gamma", "noise", "right_angle",
    })),
    "D": _D,
    "F": replace(_D, right_angle_probability=1.0),
}


def _to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)


def _to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(arr, 0.0, 255.0)).astype(np.uint8), "RGB")


def _gaussian_ksize(sigma_px: float) -> int:
    return max(3, 2 * math.ceil(3.0 * sigma_px) + 1)


def motion_blur_length(short_side: int) -> int:
    """Kernel length in px: round-half-even of 3*s/32, clamped to >= 3, then
    forced odd by adding 1 when even (odd kernels keep the blur centered)."""
    length = max(3, round(_MOTION_BASE_LENGTH * short_side / 32.0))
    if length % 2 == 0:
        length += 1
    return length


def apply_roi_shift(img: Image.Image, dx_px: float, dy_px: float,
                    canvas: Optional[Image.Image] = None) -> tuple[Image.Image, dict]:
    """Shift the W x H sampling window by (dx_px, dy_px) pixels.

    With ``canvas`` (the crop centered on a larger frame) the window is taken
    from the canvas; offsets pushing it outside are clamped (clamped=True).
    Positive dx moves the window right, so features shift left in the output.
    Without a canvas, the crop is shifted in place with edge-replication
    padding (mode="in_crop_approx").
    """
    w, h = img.size
    if canvas is not None:
        cw, ch = canvas.size
        base_x = (cw - w) / 2.0
        base_y = (ch - h) / 2.0
        wx = int(round(base_x + dx_px))
        wy = int(round(base_y + dy_px))
        cx = min(max(wx, 0), cw - w)
        cy = min(max(wy, 0), ch - h)
        op = {"op": "roi_shift", "dx_px": float(dx_px), "dy_px": float(dy_px),
              "mode": "canvas", "clamped": (cx, cy) != (wx, wy)}
        return canvas.crop((cx, cy, cx + w, cy + h)), op
    pad_x = int(math.ceil(abs(dx_px))) + 1
    pad_y = int(math.ceil(abs(dy_px))) + 1
    arr = np.asarray(img)
    padded = cv2.copyMakeBorder(arr, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
    ox = pad_x + int(round(dx_px))
    oy = pad_y + int(round(dy_px))
    shifted = padded[oy:oy + h, ox:ox + w]
    op = {"op": "roi_shift", "dx_px": float(dx_px), "dy_px": float(dy_px),
          "mode": "in_crop_approx"}
    return Image.fromarray(shifted, "RGB"), op


def apply_edge_crop(img: Image.Image, left_px: int, top_px: int,
                    right_px: int, bottom_px: int) -> tuple[Image.Image, dict]:
    w, h = img.size
    if w - left_px - right_px < 1 or h - top_px - bottom_px < 1:
        raise ValueError("edge crop would remove the whole image")
    arr = _to_float(img)
    cropped = arr[top_px:h - bottom_px, left_px:w - right_px]
    restored = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    op = {"op": "edge_crop", "left_px": int(left_px), "top_px": int(top_px),
          "right_px": int(right_px), "bottom_px": int(bottom_px)}
    return _to_pil(restored), op


def apply_small_rotation(img: Image.Image, degrees: float) -> tuple[Image.Image, dict]:
    w, h = img.size
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), degrees, 1.0)
    rotated = cv2.warpAffine(_to_float(img), matrix, (w, h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return _to_pil(rotated), {"op": "small_rotation", "degrees": float(degrees)}


def apply_downsample(img: Image.Image, ratio: float) -> tuple[Image.Image, dict]:
    w, h = img.size
    dw = max(1, round(w * ratio))
    dh = max(1, round(h * ratio))
    small = cv2.resize(_to_float(img), (dw, dh), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    op = {"op": "downsample", "ratio": float(ratio),
          "width_ratio": dw / w, "height_ratio": dh / h}
    return _to_pil(restored), op


def apply_gaussian_blur(img: Image.Image, sigma_px: float) -> tuple[Image.Image, dict]:
    ksize = _gaussian_ksize(sigma_px)
    blurred = cv2.GaussianBlur(_to_float(img), (ksize, ksize), sigmaX=sigma_px, sigmaY=sigma_px)
    return _to_pil(blurred), {"op": "gaussian_blur", "sigma_px": float(sigma_px), "ksize": ksize}


def apply_motion_blur(img: Image.Image, length: int, theta_deg: float,
                      beta: float) -> tuple[Image.Image, dict]:
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D(((length - 1) / 2.0, (length - 1) / 2.0), theta_deg, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length), flags=cv2.INTER_LINEAR)
    total = float(kernel.sum())
    if total > 0:
        kernel /= total
    arr = _to_float(img)
    blurred = cv2.filter2D(arr, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    mixed = (1.0 - beta) * arr + beta * blurred
    op = {"op": "motion_blur", "length": int(length),
          "theta_deg": float(theta_deg), "beta": float(beta)}
    return _to_pil(mixed), op


def apply_sharpen(img: Image.Image, alpha: float, sigma_px: float) -> tuple[Image.Image, dict]:
    ksize = _gaussian_ksize(sigma_px)
    arr = _to_float(img)
    blurred = cv2.GaussianBlur(arr, (ksize, ksize), sigmaX=sigma_px, sigmaY=sigma_px)
    sharpened = arr + alpha * (arr - blurred)
    op = {"op": "sharpen", "alpha": float(alpha), "sigma_px": float(sigma_px), "ksize": ksize}
    return _to_pil(sharpened), op


def apply_color_jitter(img: Image.Image, brightness: float, contrast: float,
                       saturation: float) -> tuple[Image.Image, dict]:
    """Fixed order brightness -> contrast -> saturation (no hue). Brightness
    multiplies; contrast scales around the global channel mean; saturation
    interpolates against the Rec.601 luma image."""
    arr = _to_float(img)
    arr = arr * brightness
    mean = float(arr.mean())
    arr = (arr - mean) * contrast + mean
    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    arr = gray[..., None] + saturation * (arr - gray[..., None])
    op = {"op": "color_jitter", "brightness": float(brightness),
          "contrast": float(contrast), "saturation": float(saturation)}
    return _to_pil(arr), op


def apply_gamma(img: Image.Image, gamma: float) -> tuple[Image.Image, dict]:
    """I' = I^gamma on I in [0, 1]; gamma < 1 brightens."""
    arr = _to_float(img) / 255.0
    arr = np.power(arr, gamma) * 255.0
    return _to_pil(arr), {"op": "gamma", "gamma": float(gamma)}


_apply_gamma = apply_gamma  # internal alias used by tests


def apply_noise(img: Image.Image, sigma: float, rng) -> tuple[Image.Image, dict]:
    """Zero-mean Gaussian noise on [0, 1] RGB, clipped back to [0, 1].

    The per-pixel stream comes from np.random.default_rng seeded with
    rng.getrandbits(128); this derivation keeps the whole op reproducible
    from the caller-supplied random.Random state alone.
    """
    arr = _to_float(img) / 255.0
    gen = np.random.default_rng(rng.getrandbits(128))
    arr = arr + gen.standard_normal(arr.shape, dtype=np.float32) * sigma
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    return _to_pil(arr), {"op": "noise", "sigma": float(sigma)}


def apply_right_angle_rotation(img: Image.Image, degrees: int) -> tuple[Image.Image, dict]:
    transpose = {0: None, 90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}
    if degrees not in transpose:
        raise ValueError(f"Unsupported right-angle rotation: {degrees}")
    out = img.copy() if transpose[degrees] is None else img.transpose(transpose[degrees])
    return out, {"op": "right_angle", "degrees": int(degrees)}


class TileAugment:
    def __init__(self, config: TileAugmentConfig, canvas_lookup: Optional[Callable[[str], Optional[Image.Image]]] = None):
        self.config = config
        self.canvas_lookup = canvas_lookup

    def __call__(self, img: Image.Image, *, sample_id=None, rng: random.Random = None) -> tuple[Image.Image, list[dict]]:
        rng = rng if rng is not None else random
        cfg = self.config
        if not cfg.enabled:
            return img.copy(), []
        if rng.uniform(0.0, 1.0) >= cfg.augment_probability:
            return img.copy(), []

        ops: list[dict] = []
        out = img

        u = rng.uniform(0.0, 1.0)
        if u < cfg.roi_shift_probability:
            dx = rng.uniform(-cfg.roi_shift_max_dx_fraction, cfg.roi_shift_max_dx_fraction) * out.width
            dy = rng.uniform(-cfg.roi_shift_max_dy_fraction, cfg.roi_shift_max_dy_fraction) * out.height
            canvas = self.canvas_lookup(sample_id) if (sample_id is not None and self.canvas_lookup) else None
            out, op = apply_roi_shift(out, dx, dy, canvas=canvas)
            ops.append(op)
        elif u < cfg.roi_shift_probability + cfg.edge_crop_probability:
            # floor keeps per-axis loss <= 2 * edge_crop_max_fraction after int conversion
            left = int(rng.uniform(0.0, cfg.edge_crop_max_fraction) * out.width)
            top = int(rng.uniform(0.0, cfg.edge_crop_max_fraction) * out.height)
            right = int(rng.uniform(0.0, cfg.edge_crop_max_fraction) * out.width)
            bottom = int(rng.uniform(0.0, cfg.edge_crop_max_fraction) * out.height)
            out, op = apply_edge_crop(out, left, top, right, bottom)
            ops.append(op)

        if rng.uniform(0.0, 1.0) < cfg.small_rotation_probability:
            degrees = rng.uniform(-cfg.small_rotation_max_degrees, cfg.small_rotation_max_degrees)
            out, op = apply_small_rotation(out, degrees)
            ops.append(op)

        short = min(out.width, out.height)
        u = rng.uniform(0.0, 1.0)
        threshold = cfg.downsample_probability
        if u < threshold:
            ratio = rng.uniform(*_DOWNSAMPLE_RATIO_RANGE)
            out, op = apply_downsample(out, ratio)
            ops.append(op)
        else:
            threshold += cfg.gaussian_blur_probability
            if u < threshold:
                sigma = rng.uniform(*_GAUSSIAN_SIGMA_RANGE) * short / 32.0
                out, op = apply_gaussian_blur(out, sigma)
                ops.append(op)
            else:
                threshold += cfg.motion_blur_probability
                if u < threshold:
                    length = motion_blur_length(short)
                    theta = rng.uniform(0.0, 180.0)
                    beta = rng.uniform(*_MOTION_BETA_RANGE)
                    out, op = apply_motion_blur(out, length, theta, beta)
                    ops.append(op)
                else:
                    threshold += cfg.sharpen_probability
                    if u < threshold:
                        alpha = rng.uniform(*_SHARPEN_ALPHA_RANGE)
                        sigma = _SHARPEN_SIGMA * short / 32.0
                        out, op = apply_sharpen(out, alpha, sigma)
                        ops.append(op)

        u = rng.uniform(0.0, 1.0)
        if u < cfg.color_jitter_probability:
            brightness = rng.uniform(*_JITTER_BRIGHTNESS_RANGE)
            contrast = rng.uniform(*_JITTER_CONTRAST_RANGE)
            saturation = rng.uniform(*_JITTER_SATURATION_RANGE)
            out, op = apply_color_jitter(out, brightness, contrast, saturation)
            ops.append(op)
        elif u < cfg.color_jitter_probability + cfg.gamma_probability:
            gamma = rng.uniform(*_GAMMA_RANGE)
            out, op = apply_gamma(out, gamma)
            ops.append(op)

        if rng.uniform(0.0, 1.0) < cfg.noise_probability:
            sigma = rng.uniform(*_NOISE_SIGMA_RANGE)
            out, op = apply_noise(out, sigma, rng)
            ops.append(op)

        if cfg.right_angle_probability > 0.0 and rng.uniform(0.0, 1.0) < cfg.right_angle_probability:
            u = rng.uniform(0.0, 1.0)
            cumulative = 0.0
            degrees = 0
            for degrees, mass in zip((0, 90, 180, 270), cfg.right_angle_distribution):
                cumulative += mass
                if u < cumulative:
                    break
            out, op = apply_right_angle_rotation(out, degrees)
            ops.append(op)

        return out, ops
