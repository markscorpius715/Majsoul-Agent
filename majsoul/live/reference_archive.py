"""Bounded JPEG pre-roll and event-triggered context; never infer game events."""
from collections import deque
import json
from pathlib import Path

import cv2


class ReferenceArchive:
    def __init__(self, root: Path, *, before_seconds=3.0, after_seconds=3.0,
                 sample_seconds=0.5, maximum_frames=32, jpeg_quality=85):
        if min(before_seconds, after_seconds, sample_seconds) <= 0 or maximum_frames < 2:
            raise ValueError("Invalid reference context bounds")
        self.root = root
        self.before = before_seconds
        self.after = after_seconds
        self.interval = sample_seconds
        self.quality = jpeg_quality
        self.maximum_frames = maximum_frames
        self.buffer = deque(maxlen=maximum_frames)
        self.active = None
        self.sequence = 0

    def observe(self, frame, index, match, round_index, *, force=False):
        now = frame.captured_at
        if self.buffer and (self.buffer[-1]["match_index"], self.buffer[-1]["round_index"]) != (match, round_index):
            self.close("scope_changed")
            self.buffer.clear()
        if self.buffer and self.buffer[-1]["frame_index"] == index:
            return
        if not force and self.buffer and now-self.buffer[-1]["captured_at"] < self.interval:
            return
        ok, data = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise OSError("Cannot encode reference context")
        sample = {"captured_at": now, "frame_index": index, "match_index": match,
                  "round_index": round_index, "jpeg": data.tobytes()}
        self.buffer.append(sample)
        while self.buffer and now-self.buffer[0]["captured_at"] > self.before:
            self.buffer.popleft()
        if self.active:
            if now <= self.active["until"]:
                self._save(sample)
            else:
                self.close("post_window_complete")

    def trigger(self, tags, frame, index, match, round_index):
        self.observe(frame, index, match, round_index, force=True)
        now = frame.captured_at
        if self.active:
            # A burst is one bounded episode, rather than duplicate cases per poll.
            self.active["triggers"].append({"captured_at": now, "frame_index": index,
                                            "tags": sorted(set(tags))})
            self._manifest()
            return
        self.sequence += 1
        directory = self.root / f"{self.sequence:06d}"
        directory.mkdir(parents=True, exist_ok=False)
        self.active = {"schema_version": 1, "directory": directory,
                       "trigger_at": now, "until": now+self.after,
                       "match_index": match, "round_index": round_index,
                       "before_seconds": self.before, "after_seconds": self.after,
                       "sample_seconds": self.interval, "labels_verified": False,
                       "triggers": [{"captured_at": now, "frame_index": index,
                                     "tags": sorted(set(tags))}], "frames": [],
                       "complete": False, "status": "collecting_post_context"}
        for sample in self.buffer:
            if sample["match_index"] == match and sample["round_index"] == round_index:
                self._save(sample)
        self._manifest()

    def _save(self, sample):
        case = self.active
        if len(case["frames"]) >= self.maximum_frames:
            case["frames_limited"] = True
            return
        if any(item["frame_index"] == sample["frame_index"] for item in case["frames"]):
            return
        name = f'{sample["frame_index"]:08d}.jpg'
        (case["directory"] / name).write_bytes(sample["jpeg"])
        case["frames"].append({key: value for key, value in sample.items() if key != "jpeg"}
                              | {"file": name, "offset_seconds": sample["captured_at"]-case["trigger_at"]})
        self._manifest()

    def _manifest(self):
        case = self.active
        frames = case["frames"]
        gaps = [b["captured_at"]-a["captured_at"] for a, b in zip(frames, frames[1:])]
        payload = {key: value for key, value in case.items() if key != "directory"}
        payload["maximum_gap_seconds"] = max(gaps, default=None)
        payload["available_before_seconds"] = max(0, -frames[0]["offset_seconds"]) if frames else 0
        payload["available_after_seconds"] = max(0, frames[-1]["offset_seconds"]) if frames else 0
        payload["notice"] = "Observed reference context, not a complete replay or human-verified labels."
        target = case["directory"] / "clip.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def close(self, reason="session_stopped"):
        if self.active:
            self.active["status"] = reason
            self.active["complete"] = (reason == "post_window_complete"
                                       and not self.active.get("frames_limited", False))
            self._manifest()
            self.active = None
