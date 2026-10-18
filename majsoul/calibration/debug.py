import json
from pathlib import Path

import cv2
import numpy as np

from .frames import write_image


def export_debug(image, config, output, active_slots=None, *, projection="auto"):
    missing = config.validate()
    height, width = image.shape[:2]
    selected_projection = config.projection((width, height), projection)
    boxes = selected_projection.boxes
    active_layout = None
    inactive_regions = []
    if active_slots is not None:
        active_layout = config.layout_for_active_slots(active_slots)
        included = set(config.active_region_names(active_layout))
        inactive_regions = [name for name in boxes if name not in included]
        boxes = {name: box for name, box in boxes.items() if name in included}
        missing = [name for name in missing if name in included]
    output = Path(output)
    overlay = image.copy()
    for name, (x1, y1, x2, y2) in boxes.items():
        write_image(output / "crops" / f"{name}.png", image[y1:y2, x1:x2])
        cv2.rectangle(overlay, (x1, y1), (x2-1, y2-1), (0, 220, 0), 1)
        if name in config.region_polygons:
            rw, rh = config.reference_size
            vertices = np.array([[round(x*rw*selected_projection.scale_x+selected_projection.offset_x),
                                  round(y*rh*selected_projection.scale_y+selected_projection.offset_y)]
                                 for x, y in config.region_polygons[name]], dtype=np.int32)
            cv2.polylines(overlay, [vertices], True, (0, 140, 255), 2)
        cv2.putText(overlay, name, (x1, max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 220, 0), 1, cv2.LINE_AA)
    write_image(output / "roi_debug.png", overlay)
    report = {"resolution": [width, height], "reference_resolution": config.reference_size,
              "target_size_assessment": config.assess_target_size((width, height)).to_dict(),
              "projection": selected_projection.to_dict(),
              "active_layout": active_layout, "inactive_regions": inactive_regions,
              "uncalibrated": missing, "pixel_boxes": boxes,
              "extra_unknown": [n for n, r in config.regions.items() if r is None and n not in missing]}
    (output / "roi_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def export_projection_trials(image, config, output, active_slots=None):
    """Export every plausible recording-range projection for manual/recognition tests."""
    output = Path(output)
    height, width = image.shape[:2]
    trials = []
    for candidate in config.projection_candidates((width, height)):
        trial_output = output / candidate.name
        report = export_debug(
            image, config, trial_output, active_slots,
            projection=candidate.name)
        trials.append({
            "name": candidate.name,
            "output": str(trial_output),
            "projection": report["projection"],
            "crop_count": len(report["pixel_boxes"]),
        })
    summary = {
        "resolution": [width, height],
        "target_size_assessment": config.assess_target_size((width, height)).to_dict(),
        "trials": trials,
        "selected": None,
        "requires_human_or_recognition_review": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "projection_trials.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary
