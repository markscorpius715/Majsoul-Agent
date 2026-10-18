from __future__ import annotations

import json
from pathlib import Path

import cv2

from ..calibration.frames import write_image
from .schemas import (GameMetadataReading, GlyphPrediction, MeldReading,
                      SegmentationResult, TileSequenceReading)


def export_segmentation(image, results: dict[str, SegmentationResult], output):
    output = Path(output)
    overlay = image.copy()
    report = {"regions": {}, "issues": []}
    for region, result in results.items():
        items = []
        for index, tile in enumerate(result.tiles):
            x1, y1, x2, y2 = map(round, (tile.box.x1, tile.box.y1, tile.box.x2, tile.box.y2))
            color = (0, 170, 255) if index == result.drawn_index else (0, 220, 0)
            cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), color, 2)
            label = f"{region}[{index}] r{tile.row}c{tile.column} {tile.orientation}"
            cv2.putText(overlay, label, (x1, max(14, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, color, 1, cv2.LINE_AA)
            write_image(output / "crops" / region / f"{index:02d}.png", tile.image)
            items.append({"index": index, "box": [x1, y1, x2, y2], "row": tile.row,
                          "column": tile.column, "orientation": tile.orientation,
                          "drawn": index == result.drawn_index})
        report["regions"][region] = items
        report["issues"].extend(f"{region}: {issue}" for issue in result.issues)
    write_image(output / "segmentation_debug.png", overlay)
    (output / "segmentation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def export_recognition(image, readings: dict[str, TileSequenceReading | MeldReading], output,
                       *, active_slots=None):
    output = Path(output)
    overlay = image.copy()
    report = {"regions": {}, "issues": [], "complete": True}
    if active_slots is not None:
        report["active_slots"] = list(active_slots)
    for region, reading in readings.items():
        items = []
        if isinstance(reading, MeldReading):
            sequence = [(recognized, group_index)
                        for group_index, group in enumerate(reading.groups)
                        for recognized in group]
            drawn_index = None
        else:
            sequence = [(recognized, None) for recognized in reading.tiles]
            drawn_index = reading.drawn_index
        for index, (recognized, group_index) in enumerate(sequence):
            detection = recognized.detection
            prediction = recognized.prediction
            x1, y1, x2, y2 = map(
                round, (detection.box.x1, detection.box.y1,
                        detection.box.x2, detection.box.y2))
            accepted = prediction.tile is not None
            color = (0, 200, 0) if accepted else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), color, 2)
            tile_label = prediction.tile or "unknown"
            label = f"{tile_label} {prediction.confidence:.2f}"
            cv2.putText(overlay, label, (x1, max(14, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            items.append({
                "index": index,
                "box": [x1, y1, x2, y2],
                "row": detection.row,
                "column": detection.column,
                "tile": prediction.tile,
                "confidence": prediction.confidence,
                "distance": prediction.distance,
                "margin": prediction.margin,
                "candidates": [{"tile": tile, "score": score}
                               for tile, score in prediction.candidates],
                "orientation": prediction.orientation,
                "drawn": index == drawn_index,
            })
            if group_index is not None:
                items[-1]["group"] = group_index
        report["regions"][region] = items
        report["issues"].extend(f"{region}: {issue}" for issue in reading.issues)
        report["complete"] = report["complete"] and reading.complete
    write_image(output / "recognition_debug.png", overlay)
    (output / "recognition_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _glyph_payload(prediction: GlyphPrediction | None):
    if prediction is None:
        return None
    return {
        "label": prediction.label,
        "confidence": prediction.confidence,
        "distance": prediction.distance,
        "margin": prediction.margin,
        "box": list(map(round, (prediction.box.x1, prediction.box.y1,
                                prediction.box.x2, prediction.box.y2))),
    }


def export_metadata(image, reading: GameMetadataReading, output):
    output = Path(output)
    overlay = image.copy()
    predictions = [*reading.round.glyphs, *reading.remaining_tiles.glyphs,
                   *reading.honba.glyphs, *reading.riichi_sticks.glyphs]
    predictions.extend(glyph for score in reading.scores.values() for glyph in score.glyphs)
    predictions.extend(seat.glyph for seat in reading.seats.values() if seat.glyph is not None)
    for prediction in predictions:
        x1, y1, x2, y2 = map(
            round, (prediction.box.x1, prediction.box.y1,
                    prediction.box.x2, prediction.box.y2))
        accepted = prediction.label is not None
        color = (0, 200, 0) if accepted else (0, 0, 255)
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        cv2.putText(overlay, prediction.label or "unknown", (x1, max(14, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def numeric_payload(value):
        return {"value": value.value,
                "glyphs": [_glyph_payload(item) for item in value.glyphs],
                "issues": list(value.issues), "complete": value.complete}

    report = {
        "active_slots": list(reading.active_slots),
        "round_id": reading.round_id,
        "round": {
            "wind": reading.round.round_wind,
            "hand_number": reading.round.hand_number,
            "glyphs": [_glyph_payload(item) for item in reading.round.glyphs],
            "issues": list(reading.round.issues),
            "complete": reading.round.complete,
        },
        "remaining_tiles": numeric_payload(reading.remaining_tiles),
        "honba": numeric_payload(reading.honba),
        "riichi_sticks": numeric_payload(reading.riichi_sticks),
        "scores": {slot: numeric_payload(value) for slot, value in reading.scores.items()},
        "seats": {
            slot: {
                "wind": value.wind,
                "dealer": value.dealer,
                "red_fill_ratio": value.red_fill_ratio,
                "glyph": _glyph_payload(value.glyph),
                "issues": list(value.issues),
                "complete": value.complete,
            }
            for slot, value in reading.seats.items()
        },
        "issues": list(reading.issues),
        "complete": reading.complete,
    }
    write_image(output / "metadata_debug.png", overlay)
    (output / "metadata_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
