"""Bounded lobby-only idle input, independent of play and dataset collection.

The caller supplies a reviewed empty point and a reference lobby screenshot.
Never assume a lobby template alone proves that point is unobstructed.
"""
import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np

from ..calibration.frames import read_image, write_image
from ..calibration.roi_config import ROIConfig
from .capture import FocusUnavailable, LiveWindow, roi_for_frame
from .collector import CollectConfig, menu_score
from .config import LiveConfig


@dataclass
class LobbyGuard:
    reference: np.ndarray
    point: tuple[float, float]
    ui: CollectConfig
    roi: ROIConfig
    projection: str
    patch_fraction: float = .03
    maximum_error: float = .05

    def __post_init__(self):
        if (len(self.point) != 2 or any(not math.isfinite(v) or not 0 < v < 1 for v in self.point)
                or not 0 < self.patch_fraction < .5 or not 0 < self.maximum_error < 1):
            raise ValueError("Invalid reviewed point or patch bounds")
        self.template = read_image(self.ui.menu_template)
        if not self.accepts(self.reference):
            raise ValueError("Reference must be a compatible, recognized lobby")

    def _patch(self, image):
        height, width = image.shape[:2]
        x, y = self.point
        r = self.patch_fraction / 2
        if not (r < x < 1-r and r < y < 1-r):
            raise ValueError("Reviewed patch must stay inside the window")
        patch = image[round((y-r)*height):round((y+r)*height),
                      round((x-r)*width):round((x+r)*width)]
        return cv2.resize(patch, (48, 48)).astype(np.float32) / 255

    def accepts(self, image):
        height, width = image.shape[:2]
        try:
            roi_for_frame(self.roi, (width, height), self.projection)
        except ValueError:
            return False
        if menu_score(image, self.template, self.ui.menu_region,
                      self.ui.calibration_size) < self.ui.menu_threshold:
            return False
        return float(np.abs(self._patch(image)-self._patch(self.reference)).mean()) <= self.maximum_error

    def pulse(self, window, *, sleep=time.sleep):
        before = window.capture()
        if not self.accepts(before.image):
            return "skipped_unrecognized_or_obstructed", before
        sleep(.15)
        fresh = window.capture()
        if fresh.window != before.window or not self.accepts(fresh.image):
            return "skipped_changed", fresh
        width, height = fresh.image.shape[1], fresh.image.shape[0]
        try:
            window.click((self.point[0]*width, self.point[1]*height), expected=fresh)
        except FocusUnavailable:
            return "skipped_focus_unavailable", fresh
        return "input_sent", fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/live.steam.yaml")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--point", nargs=2, type=float, required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--interval", type=float, default=180)
    parser.add_argument("--duration", type=float, default=14400)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v <= 0 for v in (args.interval, args.duration)):
        parser.error("interval and duration must be finite and positive")
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    lease = (directory / ".lease").open("a+b")
    lease.write(b"0")
    lease.flush()
    lease.seek(0)
    import msvcrt
    msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
    config = LiveConfig.load(args.config)
    guard = LobbyGuard(read_image(args.reference), tuple(args.point),
                       CollectConfig.load(config.lifecycle_ui), ROIConfig.load(config.roi),
                       config.projection)
    window = LiveWindow(LiveWindow.find(config.window_title_regex, config.window_class_regex))
    started, due, pulses = time.monotonic(), time.monotonic(), 0
    status = {"pid": os.getpid(), "started_at": time.time(), "input_count": 0,
              "duration_seconds": args.duration, "interval_seconds": args.interval,
              "reference": str(Path(args.reference).resolve()), "point": args.point,
              "status": "running", "purpose": "lobby_keepalive_only"}
    def record():
        status["updated_at"] = time.time()
        temporary = directory / "status.tmp"
        temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "status.json")
    record()
    try:
        while time.monotonic()-started < args.duration and not (directory / "STOP").exists():
            if time.monotonic() >= due:
                outcome, frame = guard.pulse(window)
                pulses += outcome == "input_sent"
                status.update(last_outcome=outcome, input_count=pulses, last_checked_at=time.time())
                write_image(directory / "last-check.png", frame.image)
                with (directory / "events.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"time": time.time(), "outcome": outcome, "input_count": pulses})+"\n")
                due = time.monotonic()+args.interval
            record()
            time.sleep(min(1, max(.01, due-time.monotonic())))
        status["status"] = "stopped"
    except (Exception, KeyboardInterrupt) as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        record()
        lease.close()


if __name__ == "__main__":
    main()
