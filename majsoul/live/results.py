"""Persist the first scorecard and its delivery intent before dismissing it."""

from hashlib import sha256
import json
from pathlib import Path
from threading import Lock
import time


class ResultDelivery:
    def __init__(self, directory, *, mode, match_index, notify, log):
        self.directory = Path(directory)
        self.mode, self.match_index = mode, match_index
        self.notify, self.log = notify, log
        self.image = None
        self.submitted = False
        self.record = None
        self.lock = Lock()

    def _write(self, **changes):
        with self.lock:
            self.record.update(changes)
            path = self.directory / f"match-{self.match_index:04d}-result-delivery.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.record, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
            temporary.replace(path)

    def capture(self, frame, save):
        if self.image is None:
            self.image = Path(save(frame))
            digest = sha256(self.image.read_bytes()).hexdigest()
            self.record = {
                "schema_version": 1, "session_id": self.directory.name,
                "match_index": self.match_index, "mode": self.mode,
                "image": str(self.image.resolve()), "image_sha256": digest,
                "captured_at": frame.captured_at, "saved_at": time.time(),
                "window": list(frame.window), "returned_lobby": False,
            }
            self._write(notification_status_snapshot="pending")
            self.log("match_result_saved", file=str(self.image), image_sha256=digest)
        if not self.submitted:
            message = (f"[{self.mode}] session {self.directory.name} number {self.match_index} match"
                       "final result screenshot saved; processing confirmation, return to lobby not yet verified")
            receipt = self.notify(message, self.image, metadata={
                "purpose": "match_result", "session_id": self.directory.name,
                "match_index": self.match_index, "mode": self.mode,
                "image_sha256": self.record["image_sha256"],
            })
            self.submitted = True
            self._write(notification_status_snapshot="queued",
                        notification_id=getattr(receipt, "notification_id", None),
                        delivery_ledger=getattr(receipt, "notification_ledger", None))
            if hasattr(receipt, "add_done_callback"):
                def completed(future):
                    try:
                        delivered = bool(future.result())
                    except Exception:
                        delivered = False
                    self._write(notification_status_snapshot="delivered" if delivered else "not_delivered")
                receipt.add_done_callback(completed)
            else:
                self._write(notification_status_snapshot="delivered" if receipt else "not_delivered")
        return self.image

    def lobby_verified(self, frame):
        if self.record is not None:
            self._write(returned_lobby=True, lobby_verified_at=frame.captured_at)
