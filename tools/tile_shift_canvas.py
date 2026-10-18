"""Build the extended-canvas cache used by ROI-shift augmentation.

For every sample of the split manifest whose ``block_id`` is not null (i.e.
samples that exist in the source video), this tool re-decodes ``first_frame``
of the video, expands the recorded ``screen_box`` by a symmetric margin, and
saves the result as ``OUT/<sample_id>.png`` with the original crop strictly
centered.  Correctness is verified by re-computing the raw manifest's
``pixel_sha256`` over the centered W x H window of the rotated canvas.

Frames are decoded once per unique ``first_frame`` (samples are grouped and
sorted by frame, each group reuses a single seek + read).

Run with an environment that has cv2, e.g.:

    D:/pScript/MiniConda/envs/bottle_yolo/python.exe tools/tile_shift_canvas.py \
        --split-manifest artifacts/tile-augment-20261002/split_manifest.json \
        --raw-manifest datasets/raw/captured/captured_yonma_20260923_005221/tile_crops/manifest.json \
        --video "屏幕录制 2026-09-23 005221.mp4" \
        --out-dir artifacts/tile-augment-20261002/canvases --margin 0.08
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2

# Physical screen slot -> rotation that brings the crop upright.
ROTATION_BY_SLOT = {"bottom": 0, "top": 180, "left": -90, "right": 90}
SLOTLESS_REGIONS = {"self_hand": "bottom", "dora_indicators": "bottom"}

# Candidate offsets (px) around the floored center window when parity makes
# the symmetric margins off by one; the hash check decides which is right.
CENTER_OFFSET_CANDIDATES = [
    (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
]


def slot_of(region: str) -> str:
    if region in SLOTLESS_REGIONS:
        return SLOTLESS_REGIONS[region]
    slot = region.rsplit("_", 1)[-1]
    if slot not in ROTATION_BY_SLOT:
        raise ValueError(f"unknown region slot: {region}")
    return slot


def rotate_frame(arr, rotation_deg: int):
    if rotation_deg == 0:
        return arr
    if rotation_deg == 180:
        return cv2.rotate(arr, cv2.ROTATE_180)
    if rotation_deg == 90:
        return cv2.rotate(arr, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == -90:
        return cv2.rotate(arr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported rotation: {rotation_deg}")


def pixel_sha256(arr) -> str:
    return hashlib.sha256(f"{arr.dtype}:{arr.shape}".encode() + arr.tobytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--margin", type=float, default=0.08,
                        help="symmetric outward margin as a fraction of each box side")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke: process only the first N eligible samples")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    raw = json.loads(Path(args.raw_manifest).read_text(encoding="utf-8"))
    raw_by_id = {s["id"]: s for s in raw["samples"]}
    margin = args.margin

    meta: dict[str, dict] = {}
    failures: list[dict] = []

    def fail(sample_id: str, reason: str):
        meta[sample_id] = {"verified": False, "reason": reason}
        failures.append({"sample_id": sample_id, "reason": reason})

    # Eligible samples: block_id non-null (present in the video). Grouped by
    # first_frame so each unique frame is decoded once.
    groups: dict[int, list] = defaultdict(list)
    eligible = [s for s in split["samples"] if s.get("block_id") is not None]

    def sort_key(sample):
        entry = raw_by_id.get(sample["sample_id"])
        frame = entry["first_frame"] if entry else 1 << 60
        return (frame, sample["sample_id"])

    eligible.sort(key=sort_key)
    if args.limit:
        eligible = eligible[: args.limit]

    for sample in eligible:
        sample_id = sample["sample_id"]
        entry = raw_by_id.get(sample_id)
        if entry is None:
            fail(sample_id, "sample_id not found in raw manifest")
            continue
        region = sample.get("region") or entry["region"]
        try:
            rotation = ROTATION_BY_SLOT[slot_of(region)]
        except ValueError as exc:
            fail(sample_id, str(exc))
            continue
        x1, y1, x2, y2 = entry["screen_box"]
        box_w, box_h = x2 - x1, y2 - y1
        expected = (box_w, box_h) if rotation in (0, 180) else (box_h, box_w)
        if expected != (sample["width"], sample["height"]):
            fail(sample_id, f"size mismatch: rotated box {expected} != manifest "
                            f"({sample['width']}, {sample['height']})")
            continue
        groups[entry["first_frame"]].append((sample, entry, rotation))

    total = sum(len(g) for g in groups.values()) + len(failures)
    print(f"eligible={len(eligible)} grouped={total - len(failures)} "
          f"unique_frames={len(groups)} pre_failed={len(failures)}", flush=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")

    processed = 0
    for frame_idx in sorted(groups):
        group = groups[frame_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            for sample, _, _ in group:
                fail(sample["sample_id"], f"failed to decode frame {frame_idx}")
                processed += 1
            continue
        # One replicate-padded frame per group; the pad must cover the largest
        # expansion needed by any sample on this frame.
        pad_frame = max(
            math.ceil(max(e["screen_box"][2] - e["screen_box"][0],
                          e["screen_box"][3] - e["screen_box"][1]) * margin) + 2
            for _, e, _ in group
        )
        padded = cv2.copyMakeBorder(frame, pad_frame, pad_frame, pad_frame, pad_frame,
                                    cv2.BORDER_REPLICATE)

        for sample, entry, rotation in group:
            sample_id = sample["sample_id"]
            x1, y1, x2, y2 = entry["screen_box"]
            box_w, box_h = x2 - x1, y2 - y1
            ex, ey = round(box_w * margin), round(box_h * margin)
            cx1, cy1 = x1 + pad_frame - ex, y1 + pad_frame - ey
            canvas = padded[cy1:cy1 + box_h + 2 * ey, cx1:cx1 + box_w + 2 * ex]
            canvas = rotate_frame(canvas, rotation)
            # Margins in the rotated (crop) coordinate system.
            margin_x, margin_y = (ex, ey) if rotation in (0, 180) else (ey, ex)

            w, h = sample["width"], sample["height"]
            ch, cw = canvas.shape[:2]
            base_x, base_y = (cw - w) // 2, (ch - h) // 2
            target = entry["pixel_sha256"]
            found = None
            for dx, dy in CENTER_OFFSET_CANDIDATES:
                ox, oy = base_x + dx, base_y + dy
                if ox < 0 or oy < 0 or ox + w > cw or oy + h > ch:
                    continue
                if pixel_sha256(canvas[oy:oy + h, ox:ox + w]) == target:
                    found = (dx, dy)
                    break
            if found is None:
                fail(sample_id, "center window pixel_sha256 mismatch")
            else:
                cv2.imwrite(str(out_dir / f"{sample_id}.png"), canvas)
                meta[sample_id] = {
                    "verified": True,
                    "margin_x_px": margin_x,
                    "margin_y_px": margin_y,
                    "rotation_deg": rotation,
                    "crop_offset": list(found),
                }
            processed += 1
            if processed % 500 == 0:
                print(f"progress {processed}/{total} "
                      f"verified={sum(1 for m in meta.values() if m['verified'])} "
                      f"failed={len(failures)}", flush=True)
    cap.release()

    verified = sum(1 for m in meta.values() if m["verified"])
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    report = {
        "split_manifest": args.split_manifest,
        "raw_manifest": args.raw_manifest,
        "video": args.video,
        "out_dir": str(out_dir),
        "margin": margin,
        "total": total,
        "verified": verified,
        "failed": len(failures),
        "failures": failures,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"done: total={total} verified={verified} failed={len(failures)}", flush=True)
    for item in failures[:20]:
        print(f"  FAIL {item['sample_id']}: {item['reason']}", flush=True)
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more (see build_report.json)", flush=True)


if __name__ == "__main__":
    main()
