from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from uuid import uuid4
import zipfile

from .store import format_review_message


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _review_package(case: Path, metadata: dict) -> Path:
    target = case / "review-package.zip"
    temporary = case / f".review-package.{uuid4().hex}.tmp"
    files = [name for name in metadata.get("files", ())
             if (case / name).is_file()]
    if "metadata.json" not in files:
        files.append("metadata.json")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(set(files)):
            archive.write(case / name, arcname=name)
    temporary.replace(target)
    return target


def _run_helper(command: list[str], runner) -> subprocess.CompletedProcess:
    result = runner(
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown helper error").strip()
        raise RuntimeError(f"Feishu helper failed: {detail}")
    return result


def notify_review_case(
        cases_root: str | Path, case_id: str, *, helper: str | Path,
        chat_id: str | None = None, node_executable: str = "node",
        runner=None) -> dict:
    """Send one review package with the configured skill helper and audit success."""
    if not re.fullmatch(r"case_\d{6}", case_id):
        raise ValueError("Invalid review case id")
    helper_path = Path(helper)
    if not helper_path.is_file():
        raise ValueError(f"Feishu helper does not exist: {helper_path}")
    case = Path(cases_root) / case_id
    metadata_path = case / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Review case is incomplete: {case_id}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    for required in ("crop.png", "full_frame.png"):
        if not (case / required).is_file():
            raise FileNotFoundError(f"Review case is missing {required}: {case_id}")

    message = format_review_message(metadata)
    message_hash = sha256(message.encode("utf-8")).hexdigest()
    target_fingerprint = (sha256(chat_id.encode("utf-8")).hexdigest()[:12]
                          if chat_id else "configured_default")
    for delivery in metadata.get("notifications", ()):
        if (delivery.get("channel") == "feishu" and
                delivery.get("target_fingerprint") == target_fingerprint and
                delivery.get("message_sha256") == message_hash and
                delivery.get("status") == "sent"):
            return {**delivery, "already_sent": True}

    run = runner or subprocess.run
    target_args = ["--chat-id", chat_id] if chat_id else []
    preflight = _run_helper(
        [node_executable, str(helper_path), "check-config", *target_args], run)
    try:
        readiness = json.loads(preflight.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Feishu helper returned invalid preflight JSON") from exc
    if not readiness.get("ready"):
        if (readiness.get("credentialsReady") and
                not readiness.get("targetReady")):
            raise RuntimeError("Feishu notification target is not configured")
        raise RuntimeError("Feishu notification credentials are not configured")

    prior_delivery = any(
        delivery.get("channel") == "feishu" and
        delivery.get("target_fingerprint") == target_fingerprint and
        delivery.get("status") == "sent"
        for delivery in metadata.get("notifications", ()))
    package = _review_package(case, metadata)
    commands = [
        [node_executable, str(helper_path), "text", "--message", message,
         *target_args],
    ]
    artifacts = []
    if not prior_delivery:
        commands.extend([
            [node_executable, str(helper_path), "image", str(case / "crop.png"),
             *target_args],
            [node_executable, str(helper_path), "image", str(case / "full_frame.png"),
             *target_args],
            [node_executable, str(helper_path), "file", str(package), *target_args],
        ])
        artifacts = ["crop.png", "full_frame.png", package.name]
    for command in commands:
        _run_helper(command, run)

    delivery = {
        "channel": "feishu",
        "status": "sent",
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "target_fingerprint": target_fingerprint,
        "message_sha256": message_hash,
        "artifacts": artifacts,
        "already_sent": False,
    }
    metadata.setdefault("notifications", []).append(
        {key: value for key, value in delivery.items()
         if key != "already_sent"})
    _atomic_json(metadata_path, metadata)
    return delivery
