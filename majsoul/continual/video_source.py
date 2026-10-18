from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import cv2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_captured_video(
        video: str | Path, *, source_id: str, manifest_path: str | Path,
        player_count: int, mode: str | None = None,
        sha256: bool = True) -> dict:
    """Register an immutable local video as review-only captured source data."""
    path = Path(video)
    if not path.is_file():
        raise ValueError(f"Video does not exist: {path}")
    if (not isinstance(source_id, str) or not source_id.strip() or
            any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in source_id.lower())):
        raise ValueError("source_id must use letters, numbers, underscores or hyphens")
    if player_count not in (3, 4):
        raise ValueError("player_count must be 3 or 4")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if (width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0 or
            frame_count <= 0):
        raise ValueError("Video metadata is incomplete")

    stat = path.stat()
    manifest = {
        "schema_version": 1,
        "source_id": source_id.strip(),
        "source_type": "captured_video",
        "domain": "mahjong_soul",
        "path": path.name,
        "bytes": stat.st_size,
        "sha256": _sha256(path) if sha256 else None,
        "video": {
            "resolution": [width, height],
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps,
            "player_count": player_count,
            "mode": mode,
        },
        "provenance": {
            "provided_by": "project_user",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "authorization": "user_authorized_for_related_content_recognition_training",
        },
        "status": "registered_unlabeled_review_only",
        "admission": {
            "training_allowed": False,
            "reason": "Samples require region extraction, deduplication and human verification",
            "allowed_scope": [
                "shared_tile_faces", "shared_glyphs", "rivers", "melds",
                "right_screen_slot", "generic_ui_states",
            ],
            "excluded_scope": [
                "three_player_scores", "kita", "sanma_event_rules",
                "automatic_pseudo_labels",
            ],
        },
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.replace(target)
    return manifest
