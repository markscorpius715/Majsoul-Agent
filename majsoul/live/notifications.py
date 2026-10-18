"""Bounded notification worker with a durable, inspectable delivery ledger."""

from concurrent.futures import Future
from hashlib import sha256
import json
from pathlib import Path
import math
from itertools import count
from queue import Empty, Full, PriorityQueue
from threading import Event, Thread
import time


def unknown_button_images(directory):
    directory = Path(directory)
    canonical = (directory / 'metadata.json').is_file()
    names = ('crop.png', 'full_frame.png') if canonical else ('crop.png', 'frame.png')
    paths = [directory/name for name in names if (directory/name).is_file()]
    if canonical and len(paths) != len(names):
        raise ValueError('Canonical review case lacks crop or full frame evidence')
    return paths


class NotificationQueue:
    def __init__(self, transport, directory: Path, *, capacity=32, logger=None,
                 max_attempts=1, retry_seconds=1.0):
        if type(max_attempts) is not int or max_attempts < 1 or retry_seconds <= 0:
            raise ValueError("Invalid notification retry bounds")
        self.transport = transport
        self.directory = Path(directory)
        self.logger = logger or (lambda *args, **kwargs: None)
        self.queue = PriorityQueue(maxsize=capacity)
        self._sequence = count()
        self.stop = Event()
        self.jobs = {}
        self.worker = None
        self.max_attempts = max_attempts
        self.retry_seconds = retry_seconds
        if type(capacity) is not int or capacity < 1 or not math.isfinite(retry_seconds):
            raise ValueError("Invalid notification queue bounds")
        self._lease = None

    def _acquire(self):
        if self._lease is not None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        stream = (self.directory / ".queue.lock").open("a+b")
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            import os
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError("Notification directory is owned by a live worker") from exc
        self._lease = stream

    def _release(self):
        if self._lease is not None:
            self._lease.close()
            self._lease = None

    @staticmethod
    def _identity(method, payload, metadata, attachments):
        return sha256(json.dumps([method, payload, metadata, attachments], ensure_ascii=False,
                                 sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _attachments(method, payload):
        paths = []
        if method == "notify_text" and len(payload) > 1 and payload[1] is not None:
            paths = [Path(payload[1])]
        elif method == "notify_unknown_button":
            paths = unknown_button_images(payload[0])
        return [{"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}
                for path in paths]

    def _write(self, job):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (job["id"] + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        temporary.replace(path)

    def submit(self, method, *args, metadata=None):
        if self.stop.is_set():
            raise RuntimeError("Notification queue is closed")
        if method not in {"notify_text", "notify_unknown_button"}:
            raise ValueError("Unsupported notification method")
        if ((method == "notify_text" and (not 1 <= len(args) <= 2 or not isinstance(args[0], str))) or
                (method == "notify_unknown_button" and (len(args) != 2 or not isinstance(args[1], str)))):
            raise ValueError("Invalid notification arguments")
        payload = [str(arg) if isinstance(arg, Path) else arg for arg in args]
        # Freeze absolute paths and byte identities before the screen is closed.
        if method == "notify_unknown_button":
            payload[0] = str(Path(payload[0]).resolve())
        elif len(payload) > 1 and payload[1] is not None:
            payload[1] = str(Path(payload[1]).resolve())
        metadata = dict(metadata or {})
        attachments = self._attachments(method, payload)
        if (metadata.get("purpose") == "match_result" and
                (len(attachments) != 1 or metadata.get("image_sha256") != attachments[0]["sha256"])):
            raise ValueError("Result notification must bind its exact saved image")
        digest = self._identity(method, payload, metadata, attachments)
        if digest in self.jobs:
            return self.jobs[digest]
        future = Future()
        future.notification_id = digest
        future.notification_ledger = str((self.directory / (digest + ".json")).resolve())
        self._acquire()
        previous = self.directory / (digest + ".json")
        if previous.exists():
            job = json.loads(previous.read_text(encoding="utf-8"))
            self.jobs[digest] = future
            # Failed jobs need an explicit bounded retry, never a restart loop.
            future.set_result(job.get("status") == "delivered")
            return future
        job = {"schema_version": 2, "id": digest, "method": method, "args": payload,
               "metadata": metadata, "attachments": attachments,
               "target": getattr(getattr(self.transport, "config", None), "chat_id", None),
               "delivery": {}, "status": "queued", "created_at": time.time(), "attempts": 0}
        self._write(job)
        return self._enqueue(job, future)

    def _enqueue(self, job, future):
        digest = job["id"]
        future.notification_id = digest
        future.notification_ledger = str((self.directory / (digest + ".json")).resolve())
        self.jobs[digest] = future
        try:
            priority = 0 if job.get("metadata", {}).get("purpose") == "match_result" else 1
            self.queue.put_nowait((priority, next(self._sequence), job, future))
        except Full:
            job["status"] = "deferred"
            job["error"] = "Notification queue is full; retained on disk"
            self._write(job)
            future.set_result(False)
            self.logger("notification_deferred", notification_id=digest)
            return future
        if self.worker is None:
            self.worker = Thread(target=self._run, name="majsoul-notifications", daemon=True)
            self.worker.start()
        return future

    def recovery_report(self):
        """Read-only inventory. Legacy or interrupted in-flight jobs need review."""
        entries = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(job, dict) or not isinstance(job.get("metadata", {}), dict):
                    raise ValueError("Invalid notification record")
                parts = job.get("delivery", {})
                if not isinstance(parts, dict) or any(not isinstance(p, dict) for p in parts.values()):
                    raise ValueError("Invalid partial-delivery record")
                eligible = (job.get("schema_version") == 2 and
                            job.get("method") == "notify_text" and
                            job.get("metadata", {}).get("purpose") == "match_result" and
                            job.get("status") in {"failed", "deferred", "queued", "retry_wait"} and
                            not any(part.get("status") in {"sending", "unknown"} for part in parts.values()) and
                            path.stem == job.get("id") == self._identity(
                                job["method"], job["args"], job["metadata"], job["attachments"]))
                if eligible and self._attachments(job["method"], job["args"]) != job["attachments"]:
                    raise ValueError("Attachment identity changed")
                if eligible and (len(job["attachments"]) != 1 or
                                 job["metadata"].get("image_sha256") != job["attachments"][0]["sha256"]):
                    raise ValueError("Result image binding is incomplete")
                target = getattr(getattr(self.transport, "config", None), "chat_id", None)
                if eligible and job.get("target") != target:
                    raise ValueError("Notification destination changed")
                reason = ("ready_for_bounded_retry" if eligible else
                          "already_delivered" if job.get("status") == "delivered" else
                          "interrupted_send_requires_review" if job.get("status") in {"sending", "delivery_unknown"} or
                          any(part.get("status") in {"sending", "unknown"} for part in parts.values()) else
                          "legacy_record_requires_review" if job.get("schema_version") != 2 else
                          "not_an_eligible_result_job")
                entries.append({"id": job.get("id"), "status": job.get("status"), "reason": reason,
                                "eligible": eligible, "path": str(path),
                                "metadata": job.get("metadata", {}), "attachments": job.get("attachments", []),
                                "delivery": parts,
                                "attempts": job.get("attempts", 0)})
            except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
                entries.append({"path": str(path), "eligible": False, "error": str(exc)})
        return entries

    def recover_results(self, *, limit):
        if type(limit) is not int or not 1 <= limit <= self.queue.maxsize:
            raise ValueError("Recovery limit must fit the bounded queue")
        if self.stop.is_set():
            raise RuntimeError("Notification queue is closed")
        self._acquire()
        futures = []
        for entry in self.recovery_report():
            if not entry["eligible"] or entry["id"] in self.jobs:
                continue
            job = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
            job["status"] = "queued"
            job["recovery_count"] = job.get("recovery_count", 0) + 1
            job["recovered_at"] = time.time()
            self._write(job)
            futures.append(self._enqueue(job, Future()))
            if len(futures) >= limit:
                break
        return futures

    def _run(self):
        try:
            self._drain()
        finally:
            self._release()

    def _drain(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                _, _, job, future = self.queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                args = job["args"]
                if job["method"] == "notify_unknown_button":
                    args = [Path(args[0]), *args[1:]]
                elif len(args) > 1 and args[1] is not None:
                    args = [args[0], Path(args[1])]
                delivered = False
                # A path that now points at different bytes must never be sent.
                if self._attachments(job["method"], job["args"]) != job["attachments"]:
                    raise ValueError("Notification attachment identity changed")
                for attempt in range(self.max_attempts):
                    if self._attachments(job["method"], job["args"]) != job["attachments"]:
                        raise ValueError("Notification attachment identity changed before retry")
                    job["status"] = "sending"
                    job["attempts"] += 1
                    self._write(job)
                    try:
                        if hasattr(self.transport, "deliver_notification"):
                            delivered = bool(self.transport.deliver_notification(
                                job, lambda: self._write(job)))
                        else:
                            delivered = bool(getattr(self.transport, job["method"])(*args))
                    except Exception as exc:
                        job["error"] = str(exc)
                    ambiguous = any(p.get("status") == "unknown" for p in job["delivery"].values())
                    if delivered or ambiguous or self.stop.is_set() or attempt+1 == self.max_attempts:
                        break
                    job["status"] = "retry_wait"
                    self._write(job)
                    self.stop.wait(self.retry_seconds)
                job["status"] = ("delivered" if delivered else "delivery_unknown" if ambiguous else "failed")
                if delivered:
                    job.pop("error", None)
                job["completed_at"] = time.time()
                self._write(job)
                self.logger("notification_result", notification_id=job["id"],
                            status=job["status"])
                future.set_result(delivered)
            except Exception as exc:
                job["status"] = "failed"
                job["error"] = str(exc)
                try:
                    self._write(job)
                finally:
                    future.set_result(False)
            finally:
                self.queue.task_done()

    def close(self, timeout=3):
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=timeout)
            if self.worker.is_alive() and hasattr(self.transport, "cancel"):
                self.transport.cancel()
        else:
            self._release()
        pending = sum(not future.done() for future in self.jobs.values())
        if pending:
            self.logger("notifications_pending", count=pending,
                        directory=str(self.directory))
        return pending
