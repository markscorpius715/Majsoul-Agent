"""Feishu skill-helper transport with acknowledged parts and stoppable children."""

from __future__ import annotations

import json
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import time
from threading import Event, Lock

from .config import NotifyConfig
from .notifications import unknown_button_images


class NotificationInterrupted(RuntimeError):
    """The helper ended without an acknowledgment; remote delivery is unknown."""


class FeishuNotifier:
    """Send text and evidence through the existing skill helper; failures log."""

    def __init__(self, config: NotifyConfig, *, runner=None, logger=None):
        self.config = config
        self.runner = runner
        self.logger = logger or (lambda message: None)
        self._delivered_text = set()
        self._delivered_images = set()
        self._closed = Event()
        self._process_lock = Lock()
        self._process = None

    def cancel(self):
        self._closed.set()
        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                try:
                    self._process.kill()
                except OSError:
                    pass  # It may have exited between poll and kill.

    def _run(self, command: list[str]) -> subprocess.CompletedProcess:
        environment = os.environ.copy()
        # Route each helper explicitly; inherited proxy variables must not
        # defeat configured direct mode or leave a retired proxy in service.
        for name in tuple(environment):
            if name.lower() in ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'):
                environment.pop(name)
        if self.config.proxy:
            environment.update(HTTP_PROXY=self.config.proxy, HTTPS_PROXY=self.config.proxy)
        if self.runner is not None:
            result = self.runner(
                command, capture_output=True, text=True, encoding="utf-8",
                errors="replace", check=False, timeout=self.config.timeout_seconds,
                env=environment)
        else:
            with self._process_lock:
                if self._closed.is_set():
                    raise NotificationInterrupted("Notification worker was stopped")
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace", env=environment,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                self._process = process
            try:
                try:
                    stdout, stderr = process.communicate(timeout=self.config.timeout_seconds)
                except subprocess.TimeoutExpired as exc:
                    process.kill()
                    process.communicate()
                    raise NotificationInterrupted("Feishu helper timed out without an acknowledgment") from exc
                if self._closed.is_set():
                    raise NotificationInterrupted("Feishu helper interrupted before acknowledgment")
                result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            finally:
                with self._process_lock:
                    self._process = None
        if result.returncode != 0:
            detail = (result.stderr or result.stdout
                      or "unknown helper error").strip()
            raise RuntimeError(f"Feishu helper failed: {detail}")
        return result

    def notify_text(self, message: str, image: Path | None = None) -> bool:
        return self.deliver_notification(
            {"method": "notify_text", "args": [message, image], "delivery": {}}, lambda: None)

    def notify_unknown_button(self, case_dir: Path, message: str) -> bool:
        return self.deliver_notification(
            {"method": "notify_unknown_button", "args": [case_dir, message], "delivery": {}},
            lambda: None)

    def deliver_notification(self, job, checkpoint) -> bool:
        """Checkpoint each acknowledged part, so image retries do not resend text.

        The skill helper exposes success, but no platform message identifier.
        A process lost during a send has no exactly-once delivery guarantee.
        """
        if not self.config.enabled:
            return False
        try:
            if "target" in job and job["target"] != self.config.chat_id:
                raise RuntimeError("Notification destination differs from the queued destination")
            method, args = job["method"], job["args"]
            if method == "notify_text":
                message = args[0]
                paths = [Path(args[1])] if len(args) > 1 and args[1] is not None else []
            elif method == "notify_unknown_button":
                message = args[1]
                paths = unknown_button_images(args[0])
            else:
                raise ValueError("Unsupported notification method")
            steps = [("text", sha256(message.encode("utf-8")).hexdigest(), message)]
            for path in paths:
                steps.append(("image", sha256(path.read_bytes()).hexdigest(), str(path)))
            helper = str(self.config.feishu_helper)
            target = (["--chat-id", self.config.chat_id]
                      if self.config.chat_id else [])
            preflight = self._run(
                [self.config.node, helper, "check-config", *target])
            readiness = json.loads(preflight.stdout)
            if not readiness.get("ready"):
                raise RuntimeError("Feishu notification target is not ready")
            delivery = job.setdefault("delivery", {})
            if delivery and job.get("delivery_target") != self.config.chat_id:
                raise RuntimeError("Notification destination differs from the recorded destination")
            job["delivery_target"] = self.config.chat_id
            for kind, digest, value in steps:
                key = f"{kind}:{digest}"
                receipt = delivery.get(key, {})
                memory = self._delivered_text if kind == "text" else self._delivered_images
                context = job.get("metadata", {})
                scoped = (self.config.chat_id, context.get("session_id"),
                          context.get("match_index"), digest)
                if receipt.get("status") == "delivered":
                    continue
                if receipt.get("status") in {"sending", "unknown"}:
                    raise RuntimeError("Interrupted notification part requires delivery review")
                if scoped not in memory:
                    delivery[key] = {"kind": kind, "sha256": digest, "status": "sending",
                                     "started_at": time.time(), "platform_message_id": None}
                    checkpoint()
                    command = ([self.config.node, helper, "text", "--message", value, *target]
                               if kind == "text" else [self.config.node, helper, "image", value, *target])
                    try:
                        self._run(command)
                    except Exception as exc:
                        delivery[key]["status"] = "unknown" if isinstance(exc, NotificationInterrupted) else "failed"
                        checkpoint()
                        raise
                    memory.add(scoped)
                delivery[key] = {"kind": kind, "sha256": digest, "status": "delivered",
                                 "acknowledged_at": time.time(), "platform_message_id": None,
                                 "evidence": "skill_helper_exit_success"}
                checkpoint()
            return True
        except Exception as exc:
            job["error"] = str(exc)
            self.logger(f"Feishu notify failed: {exc}")
            return False
