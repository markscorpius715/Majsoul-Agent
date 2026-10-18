from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..calibration.frames import read_image, write_image


def create_review_sheet(
        cases_root: str | Path, case_ids: list[str] | tuple[str, ...],
        output: str | Path, *, columns: int = 4,
        cell_size: tuple[int, int] = (240, 220)) -> dict:
    """Create a compact, labeled sheet without changing review status."""
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("case_ids must be a non-empty unique sequence")
    if columns < 1 or min(cell_size) < 80:
        raise ValueError("Invalid review sheet layout")
    root = Path(cases_root)
    cell_width, cell_height = cell_size
    header_height = 54
    rows = math.ceil(len(case_ids) / columns)
    canvas = np.full((rows * cell_height, columns * cell_width, 3), 245,
                     dtype=np.uint8)
    entries = []
    for index, case_id in enumerate(case_ids):
        case = root / case_id
        metadata_path = case / "metadata.json"
        crop_path = case / "crop.png"
        if not metadata_path.is_file() or not crop_path.is_file():
            raise FileNotFoundError(f"Review case is incomplete: {case_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        crop = read_image(crop_path)
        available_width = cell_width - 16
        available_height = cell_height - header_height - 10
        scale = min(available_width / crop.shape[1],
                    available_height / crop.shape[0])
        resized = cv2.resize(
            crop, (max(1, round(crop.shape[1] * scale)),
                   max(1, round(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
        row, column = divmod(index, columns)
        x0, y0 = column * cell_width, row * cell_height
        x = x0 + (cell_width - resized.shape[1]) // 2
        y = y0 + header_height + (available_height - resized.shape[0]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.rectangle(canvas, (x0, y0),
                      (x0 + cell_width - 1, y0 + cell_height - 1),
                      (120, 120, 120), 1)
        cv2.putText(canvas, case_id, (x0 + 8, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1,
                    cv2.LINE_AA)
        subtitle = f"{metadata['kind']} {metadata['region']}"
        cv2.putText(canvas, subtitle[:32], (x0 + 8, y0 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (60, 60, 60), 1,
                    cv2.LINE_AA)
        entries.append({
            "case_id": case_id,
            "kind": metadata["kind"],
            "region": metadata["region"],
            "status": metadata["status"],
            "source": metadata.get("source"),
        })
    target = Path(output)
    write_image(target, canvas)
    report = {"output": str(target), "columns": columns, "entries": entries}
    target.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def create_crop_corpus_sheet(
        manifest: str | Path, output: str | Path, *,
        max_per_region: int = 8,
        cell_size: tuple[int, int] = (180, 190)) -> dict:
    """Build a deterministic time-spread audit sheet from an unverified corpus."""
    if max_per_region < 1 or min(cell_size) < 80:
        raise ValueError("Invalid crop corpus sheet layout")
    manifest_path = Path(manifest)
    data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if data.get("schema_version") != 1 or not isinstance(data.get("samples"), list):
        raise ValueError("Unsupported crop corpus manifest")
    grouped: dict[str, list[dict]] = {}
    for sample in data["samples"]:
        grouped.setdefault(sample["region"], []).append(sample)
    if not grouped:
        raise ValueError("Crop corpus has no samples")
    for samples in grouped.values():
        samples.sort(key=lambda item: (item["first_frame"], item["id"]))

    selected = {}
    for region, samples in sorted(grouped.items()):
        if len(samples) <= max_per_region:
            selected[region] = samples
            continue
        indexes = np.linspace(0, len(samples) - 1, max_per_region)
        selected[region] = [samples[round(index)] for index in indexes]

    cell_width, cell_height = cell_size
    header_height = 50
    regions = list(selected)
    canvas = np.full(
        (len(regions) * cell_height, max_per_region * cell_width, 3),
        245, dtype=np.uint8)
    entries = []
    root = manifest_path.parent
    for row, region in enumerate(regions):
        for column, sample in enumerate(selected[region]):
            crop = read_image(root / sample["path"])
            available_width = cell_width - 12
            available_height = cell_height - header_height - 8
            scale = min(available_width / crop.shape[1],
                        available_height / crop.shape[0])
            resized = cv2.resize(
                crop, (max(1, round(crop.shape[1] * scale)),
                       max(1, round(crop.shape[0] * scale))),
                interpolation=(cv2.INTER_AREA if scale < 1
                               else cv2.INTER_NEAREST))
            x0, y0 = column * cell_width, row * cell_height
            x = x0 + (cell_width - resized.shape[1]) // 2
            y = y0 + header_height + (available_height - resized.shape[0]) // 2
            canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            cv2.rectangle(
                canvas, (x0, y0),
                (x0 + cell_width - 1, y0 + cell_height - 1),
                (120, 120, 120), 1)
            cv2.putText(
                canvas, region[:24], (x0 + 6, y0 + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1,
                cv2.LINE_AA)
            cv2.putText(
                canvas, f"f{sample['first_frame']} {sample['orientation']}",
                (x0 + 6, y0 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (60, 60, 60), 1, cv2.LINE_AA)
            entries.append({
                "id": sample["id"], "region": region,
                "frame": sample["first_frame"], "path": sample["path"],
            })
    target = Path(output)
    write_image(target, canvas)
    report = {
        "output": str(target), "manifest": str(manifest_path),
        "max_per_region": max_per_region, "region_count": len(regions),
        "entries": entries,
    }
    target.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
