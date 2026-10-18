"""Diagnostic degraded-set evaluation for tile classifier checkpoints.

Applies a fixed, deterministic degradation pipeline to a split of the gold
dataset and evaluates a trained checkpoint. This is a *diagnostic* tool: the
degraded sets reuse augmentation-style corruptions and must not replace clean
validation or an independent-source test.

Usage:

    D:/pScript/MiniConda/envs/bottle_yolo/python.exe tools/tile_augment_degraded_eval.py \
        --checkpoint artifacts/tile-augment-20261002/runs/D_s42/checkpoint.pt \
        --split-manifest artifacts/tile-augment-20261002/split_manifest.json \
        --split val --degradation moderate --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import tile_model_select as tms  # noqa: E402
import tile_augment_study as tas  # noqa: E402
from majsoul.perception import tile_augment as ta  # noqa: E402


def sample_rng(sample_id: str, seed: int) -> "object":
    import random
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def make_degradation(name: str):
    """Return degrade(img, sample_id) -> PIL image with fixed per-sample draws."""

    def degrade(img, sample_id):
        rng = sample_rng(sample_id, make_degradation.seed)
        s = min(img.size)
        if name == "none":
            pass
        elif name == "mild":
            img, _ = ta.apply_downsample(img, 0.85)
            img, _ = ta.apply_color_jitter(img, brightness=0.95, contrast=0.95, saturation=1.0)
            img, _ = ta.apply_noise(img, 0.005, rng)
        elif name == "moderate":
            img, _ = ta.apply_downsample(img, 0.70)
            img, _ = ta.apply_gaussian_blur(img, 0.45 * s / 32)
            img, _ = ta.apply_color_jitter(img, brightness=0.90, contrast=0.90, saturation=0.95)
            img, _ = ta.apply_noise(img, 0.008, rng)
        elif name == "strong":
            img, _ = ta.apply_downsample(img, 0.65)
            img, _ = ta.apply_motion_blur(img, ta.motion_blur_length(s), 45.0, 0.4)
            img, _ = ta.apply_color_jitter(img, brightness=0.85, contrast=0.85, saturation=0.90)
            img, _ = ta.apply_gamma(img, 1.10)
            img, _ = ta.apply_noise(img, 0.012, rng)
        else:
            raise ValueError(f"unknown degradation: {name}")
        return img

    return degrade


make_degradation.seed = 42


class DegradedTransform:
    def __init__(self, degrade):
        self.degrade = degrade
        self.post = tas.EvalTransform()

    def __call__(self, img, sample_id=None):
        return self.post(self.degrade(img, sample_id))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--degradation", default="moderate",
                        choices=["none", "mild", "moderate", "strong"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    make_degradation.seed = args.seed

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    samples = [s for s in manifest["samples"] if s["split"] == args.split]

    dataset = tas.ManifestTileDataset(samples, DegradedTransform(make_degradation(args.degradation)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = tms.build_model("convnext_tiny", len(tms.TILE_LABELS))
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    n_classes = len(tms.TILE_LABELS)
    import numpy as np
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.inference_mode():
        for images, targets, _sids in loader:
            images = images.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(images)
            preds = logits.float().argmax(dim=1).cpu()
            for t, p in zip(targets.tolist(), preds.tolist()):
                cm[t, p] += 1

    total = int(cm.sum())
    correct = int(np.trace(cm))
    per_class = {}
    for i, label in enumerate(tms.TILE_LABELS):
        support = int(cm[i].sum())
        per_class[label] = {
            "recall": round(int(cm[i, i]) / support, 4) if support else None,
            "support": support,
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "degradation": args.degradation,
        "seed": args.seed,
        "support": total,
        "overall_acc": round(correct / total, 5) if total else None,
        "macro_f1": round(tas.macro_f1_from_cm(cm), 5),
        "per_class": per_class,
        "note": "diagnostic degraded set; not a substitute for clean val/test",
    }
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(f"degraded[{args.degradation}] split={args.split} acc={result['overall_acc']} "
          f"macro_f1={result['macro_f1']} n={total}")


if __name__ == "__main__":
    main()
