"""Contact-sheet preview for tile augmentation endpoints.

Usage:
    .venv/Scripts/python tools/tile_augment_preview.py \
        --split-manifest artifacts/tile-augment-20261002/split_manifest.json \
        --out-dir artifacts/tile-augment-20261002/preview [--seed 42]
    .venv/Scripts/python tools/tile_augment_preview.py \
        --samples "datasets/verified/**/*.png" --out-dir .../preview

Rows are grouped by label (parent directory name); each row shows endpoint
applications of every single op (shared implementations from
majsoul.perception.tile_augment) plus three seeded random combinations.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from majsoul.perception.tile_augment import (PRESETS, TileAugment, apply_color_jitter,
                                             apply_downsample, apply_edge_crop, apply_gamma,
                                             apply_gaussian_blur, apply_motion_blur,
                                             apply_noise, apply_roi_shift, apply_sharpen,
                                             apply_small_rotation, motion_blur_length)

CELL_W = 150
CELL_H = 200
HEADER_H = 44
TITLE_H = 24
MAX_SHEET_W = 4000
MAX_PER_LABEL = 6


@dataclass
class Sample:
    path: Path
    label: str
    region: str | None
    canvas_path: Path | None


def size_bucket(img: Image.Image) -> str:
    s = min(img.size)
    if s < 64:
        return "<64"
    if s <= 128:
        return "64-128"
    return ">128"


def load_manifest_samples(manifest_path: Path, canvas_dir: Path | None = None) -> list[Sample]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "samples" in data:
            entries = data["samples"]
        elif "entries" in data:
            entries = data["entries"]
        elif "splits" in data:
            entries = [e for split in data["splits"].values() for e in split]
        else:
            entries = []
    else:
        entries = data
    samples = []
    for entry in entries:
        path = Path(entry.get("path") or entry.get("image"))
        if not path.is_absolute():
            # manifest paths are project-root relative; fall back to
            # manifest-relative for standalone manifests
            if not path.exists():
                path = manifest_path.parent / path
        label = entry.get("label") or path.parent.name
        canvas = entry.get("canvas") or entry.get("canvas_path")
        canvas_path = None
        if canvas:
            canvas_path = Path(canvas)
            if not canvas_path.is_absolute():
                canvas_path = manifest_path.parent / canvas_path
        elif canvas_dir and entry.get("sample_id"):
            candidate = canvas_dir / f"{entry['sample_id']}.png"
            if candidate.exists():
                canvas_path = candidate
        samples.append(Sample(path=path, label=label,
                              region=entry.get("region"), canvas_path=canvas_path))
    return samples


def load_glob_samples(patterns: list[str]) -> list[Sample]:
    samples = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            matches = sorted(p.rglob("*.png"))
        else:
            matches = sorted(Path(m) for m in glob.glob(pattern, recursive=True))
        for match in matches:
            samples.append(Sample(path=match, label=match.parent.name,
                                  region=None, canvas_path=None))
    return samples


def stratified_select(samples: list[Sample], max_per_label: int) -> list[Sample]:
    """Per label, prefer covering distinct (region, size bucket) combos."""
    by_label: dict[str, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    selected = []
    for label in sorted(by_label):
        candidates = sorted(by_label[label], key=lambda s: str(s.path))
        combos: dict[tuple, Sample] = {}
        for sample in candidates:
            with Image.open(sample.path) as img:
                key = (sample.region, size_bucket(img))
            combos.setdefault(key, sample)
        picked = [combos[key] for key in sorted(combos, key=str)][:max_per_label]
        if not picked:
            picked = candidates[:1]
        selected.extend(picked)
    return selected


def op_title(op: dict) -> str:
    name = op["op"]
    skip = {"op"}
    parts = []
    for key, value in op.items():
        if key in skip:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.3g}")
        else:
            parts.append(f"{key}={value}")
    return name + ("\n" + " ".join(parts) if parts else "")


def endpoint_cells(img: Image.Image, canvas: Image.Image | None, seed: int) -> list[tuple[str, Image.Image]]:
    w, h = img.size
    s = min(w, h)
    cells: list[tuple[str, Image.Image]] = [("original", img)]

    max_dx = PRESETS["D"].roi_shift_max_dx_fraction * w
    for sign, tag in ((1.0, "+max"), (-1.0, "-max")):
        out, op = apply_roi_shift(img, sign * max_dx, 0.0, canvas=canvas)
        mode = op["mode"] if op["mode"] == "canvas" else "approx"
        cells.append((f"roi_shift {tag}\ndx={op['dx_px']:.2f} {mode}", out))

    out, op = apply_edge_crop(img, int(0.02 * w), int(0.02 * h), int(0.02 * w), int(0.02 * h))
    cells.append(("edge_crop max\n"
                  f"l{op['left_px']} t{op['top_px']} r{op['right_px']} b{op['bottom_px']}", out))

    for deg in (3.0, -3.0):
        out, op = apply_small_rotation(img, deg)
        cells.append((f"small_rotation\n{deg:+.1f} deg", out))

    out, op = apply_downsample(img, 0.65)
    cells.append((f"downsample\nr={op['ratio']}", out))

    sigma = 0.6 * s / 32.0
    out, op = apply_gaussian_blur(img, sigma)
    cells.append((f"gaussian_blur\nsigma={op['sigma_px']:.3g} k={op['ksize']}", out))

    length = motion_blur_length(s)
    out, op = apply_motion_blur(img, length, 45.0, 0.4)
    cells.append((f"motion_blur\nL={op['length']} th={op['theta_deg']:g} b={op['beta']}", out))

    out, op = apply_sharpen(img, 0.3, 0.5 * s / 32.0)
    cells.append((f"sharpen\na={op['alpha']} sigma={op['sigma_px']:.3g}", out))

    for b in (0.85, 1.15):
        out, op = apply_color_jitter(img, b, 1.0, 1.0)
        cells.append((f"color_jitter\nbrightness={b}", out))

    for g in (0.90, 1.10):
        out, op = apply_gamma(img, g)
        cells.append((f"gamma\n{g}", out))

    out, op = apply_noise(img, 0.012, random.Random(seed))
    cells.append((f"noise\nsigma={op['sigma']}", out))

    combo_cfg = replace(PRESETS["D"], augment_probability=1.0)
    aug = TileAugment(combo_cfg,
                      canvas_lookup=(lambda _sid: canvas) if canvas is not None else None)
    for i in range(3):
        out, ops = aug(img, sample_id="preview", rng=random.Random(seed * 1000 + i))
        title = f"random #{i + 1}\n" + ("; ".join(op["op"] for op in ops) or "none")
        cells.append((title, out))
    return cells


def fit_cell(img: Image.Image) -> Image.Image:
    cell = Image.new("RGB", (CELL_W, CELL_H), (255, 255, 255))
    thumb = img.copy()
    thumb.thumbnail((CELL_W - 8, CELL_H - 8), Image.BILINEAR)
    cell.paste(thumb, ((CELL_W - thumb.width) // 2, (CELL_H - thumb.height) // 2))
    return cell


def render_sheets(rows: list[dict], out_dir: Path, seed: int) -> list[dict]:
    font = ImageFont.load_default()
    sheets = []
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    for label in sorted(by_label):
        label_rows = by_label[label]
        cells_per_page = max(1, MAX_SHEET_W // CELL_W)
        n_cells = len(label_rows[0]["cells"])
        pages = max(1, -(-n_cells // cells_per_page))
        for page in range(pages):
            lo = page * cells_per_page
            hi = min(n_cells, lo + cells_per_page)
            width = (hi - lo) * CELL_W
            height = TITLE_H + len(label_rows) * (HEADER_H + CELL_H)
            sheet = Image.new("RGB", (width, height), (230, 230, 230))
            draw = ImageDraw.Draw(sheet)
            title = f"label={label} page {page + 1}/{pages}"
            draw.text((4, 6), title, fill=(0, 0, 0), font=font)
            for r, row in enumerate(label_rows):
                y0 = TITLE_H + r * (HEADER_H + CELL_H)
                for c, (cell_title, cell_img) in enumerate(row["cells"][lo:hi]):
                    x0 = c * CELL_W
                    for line_no, line in enumerate(cell_title.split("\n")[:3]):
                        draw.text((x0 + 4, y0 + 4 + line_no * 12), line,
                                  fill=(0, 0, 0), font=font)
                    sheet.paste(fit_cell(cell_img), (x0, y0 + HEADER_H))
            safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
            name = f"sheet_{safe_label}_p{page + 1}.png" if pages > 1 else f"sheet_{safe_label}.png"
            sheet.save(out_dir / name)
            sheets.append({
                "file": name,
                "labels": sorted({r["label"] for r in label_rows}),
                "regions": sorted({r["region"] for r in label_rows if r["region"]}),
                "size_buckets": sorted({r["bucket"] for r in label_rows}),
                "samples": [str(r["path"]) for r in label_rows],
                "cells": [title for title, _ in label_rows[0]["cells"][lo:hi]],
            })
    return sheets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--samples", nargs="*", default=None,
                        help="PNG files, directories, or glob patterns")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--canvas-dir", type=Path, default=None,
                        help="directory of <sample_id>.png shift canvases")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-label", type=int, default=MAX_PER_LABEL)
    args = parser.parse_args()

    if args.split_manifest:
        samples = load_manifest_samples(args.split_manifest, args.canvas_dir)
    elif args.samples:
        samples = load_glob_samples(args.samples)
    else:
        parser.error("one of --split-manifest or --samples is required")
    if not samples:
        parser.error("no samples found")

    selected = stratified_select(samples, args.max_per_label)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, sample in enumerate(selected):
        img = Image.open(sample.path).convert("RGB")
        canvas = None
        if sample.canvas_path and sample.canvas_path.exists():
            canvas = Image.open(sample.canvas_path).convert("RGB")
        rows.append({
            "label": sample.label,
            "region": sample.region,
            "bucket": size_bucket(img),
            "path": sample.path,
            "cells": endpoint_cells(img, canvas, args.seed + index),
        })

    sheets = render_sheets(rows, args.out_dir, args.seed)
    index = {"seed": args.seed, "n_samples": len(selected), "sheets": sheets}
    (args.out_dir / "preview_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(sheets)} sheet(s) for {len(selected)} sample(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
