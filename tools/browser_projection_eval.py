"""Compare ROI projection candidates on browser-captured frames.

For every projection candidate (stretch / fit_height_* / fit_width_*) this tool
derives an equivalent ROIConfig whose reference size matches the target frame
(``auto`` then resolves to the same pixel boxes), runs the full visible-tiles
perception pipeline, and reports recognition counts and confidence per region.
The derived config of the chosen projection can be promoted to a versioned
browser layout config afterwards.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import PROJECTION_NAMES, ROIConfig
from majsoul.perception.config import VisionConfig
from majsoul.perception.tile_classifier import TileTemplateClassifier
from majsoul.perception.visible_tiles import VisibleTilesReader


def derived_config(config: ROIConfig, size, projection: str) -> ROIConfig:
    width, height = size
    projected = config.projection(size, projection)
    regions = {}
    for name, rect in config.regions.items():
        box = projected.boxes.get(name)
        if rect is None or box is None:
            regions[name] = None
            continue
        x1, y1, x2, y2 = box
        regions[name] = [x1 / width, y1 / height, x2 / width, y2 / height]
    derived = dataclasses.replace(
        config, reference_size=[width, height], regions=regions)
    derived.validate()
    return derived


def _sequence_summary(reading) -> dict:
    labels = [tile.prediction.tile for tile in reading.tiles]
    confidences = [round(tile.prediction.confidence, 4) for tile in reading.tiles]
    return {
        "labels": labels,
        "confidences": confidences,
        "tiles": len(labels),
        "unknown": sum(1 for label in labels if label is None),
        "mean_confidence": (round(sum(confidences) / len(confidences), 4)
                            if confidences else None),
        "issues": list(reading.issues),
    }


def _meld_summary(reading) -> dict:
    flat = [tile for group in reading.groups for tile in group]
    labels = [tile.prediction.tile for tile in flat]
    confidences = [round(tile.prediction.confidence, 4) for tile in flat]
    return {
        "groups": [[tile.prediction.tile for tile in group]
                   for group in reading.groups],
        "tiles": len(labels),
        "unknown": sum(1 for label in labels if label is None),
        "mean_confidence": (round(sum(confidences) / len(confidences), 4)
                            if confidences else None),
        "issues": list(reading.issues),
    }


def evaluate(image_path: str, roi_path: str, vision_path: str, templates: str,
             active_slots, projections, output: str) -> dict:
    image = read_image(image_path)
    height, width = image.shape[:2]
    config = ROIConfig.load(roi_path)
    vision = VisionConfig.load(vision_path)
    classifier = TileTemplateClassifier(templates, vision)
    if not classifier.labels:
        raise SystemExit(f"No labeled tile templates found in {templates}")
    reader = VisibleTilesReader(vision, classifier)

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"image": image_path, "size": [width, height],
              "active_slots": list(active_slots), "projections": {}}
    for name in projections:
        derived = derived_config(config, (width, height), name)
        proj_dir = out_dir / name
        proj_dir.mkdir(parents=True, exist_ok=True)
        derived.save(proj_dir / "roi_derived.yaml")
        reading = reader.read(image, derived, active_slots)
        regions = {"self_hand": _sequence_summary(reading.hand),
                   "dora_indicators": _sequence_summary(reading.dora)}
        for slot, river in reading.rivers.items():
            regions[f"river_{slot}"] = _sequence_summary(river)
        for slot, meld in reading.melds.items():
            regions[f"melds_{slot}"] = _meld_summary(meld)
        for slot, kita in reading.kita.items():
            regions[f"kita_{slot}"] = _sequence_summary(kita)
        total = sum(r["tiles"] for r in regions.values())
        unknown = sum(r["unknown"] for r in regions.values())
        scored = [r["mean_confidence"] for r in regions.values()
                  if r["mean_confidence"] is not None]
        report["projections"][name] = {
            "regions": regions, "tiles": total, "unknown": unknown,
            "mean_confidence": (round(sum(scored) / len(scored), 4)
                                if scored else None),
            "config": str(proj_dir / "roi_derived.yaml"),
        }
        print(f"{name}: tiles={total} unknown={unknown} "
              f"mean_conf={report['projections'][name]['mean_confidence']}")
    summary = out_dir / "projection_eval.json"
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print(f"wrote {summary}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--roi-config", default="configs/roi.yaml")
    parser.add_argument("--vision-config", default="configs/vision.yaml")
    parser.add_argument("--templates", default="artifacts/templates/tiles")
    parser.add_argument("--active-slots", nargs="+",
                        default=["bottom", "top", "left"])
    parser.add_argument("--projections", nargs="+", default=list(PROJECTION_NAMES),
                        choices=list(PROJECTION_NAMES))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    evaluate(args.image, args.roi_config, args.vision_config, args.templates,
             args.active_slots, args.projections, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
