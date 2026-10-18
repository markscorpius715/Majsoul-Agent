from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from typing import Callable
from uuid import uuid4

from ..actions import AgentActionType
from ..calibration.frames import read_image, write_image
from ..tiles import tile_index
from .detector import difference_hash, hamming_distance
from .schemas import ReviewCase, ReviewKind, ReviewStatus, UnknownEvidence


_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.:-]+$")
_INDEXED_REGION = re.compile(r"\[\d+\]$")
_DATASET_GROUP = {
    ReviewKind.TILE: "tiles",
    ReviewKind.ACTION: "actions",
    ReviewKind.MELD: "melds",
    ReviewKind.UI_STATE: "ui_states",
    ReviewKind.METADATA: "metadata",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.replace(path)


def validate_label(kind: ReviewKind, label: str) -> str:
    label = label.strip()
    if not label or not _SAFE_LABEL.fullmatch(label):
        raise ValueError("Label must contain only letters, digits, _, ., :, or -")
    if kind == ReviewKind.TILE:
        tile_index(label)
        return label
    if kind == ReviewKind.ACTION:
        return AgentActionType(label.upper()).value
    if kind == ReviewKind.MELD:
        normalized = label.upper()
        if normalized not in {"MELD", "CHI", "PON", "KAN"}:
            raise ValueError("Meld label must be MELD, CHI, PON, or KAN")
        return normalized
    return label


def _dedup_region(region: str) -> str:
    return _INDEXED_REGION.sub("[]", region)


class ReviewCaseStore:
    def __init__(self, cases_root: str | Path, dataset_root: str | Path):
        self.cases_root = Path(cases_root)
        self.dataset_root = Path(dataset_root)

    def _next_case_id(self) -> str:
        self.cases_root.mkdir(parents=True, exist_ok=True)
        numbers = []
        for path in self.cases_root.glob("case_*"):
            match = re.fullmatch(r"case_(\d{6})", path.name)
            if match:
                numbers.append(int(match.group(1)))
        return f"case_{(max(numbers, default=0) + 1):06d}"

    def create(self, evidence: UnknownEvidence, *,
               dedup_hamming_distance: int = 6,
               dedup_match: Callable | None = None) -> ReviewCase:
        if evidence.timestamp < 0 or evidence.frame_index < 0:
            raise ValueError("Review evidence time and frame must be non-negative")
        player_count = evidence.metadata.get("player_count")
        mode = evidence.metadata.get("mode")
        if player_count not in (3, 4) or not isinstance(mode, str) or not mode.strip():
            raise ValueError(
                "Review evidence must record player_count and mode provenance")
        visual_hash = difference_hash(evidence.crop)
        dedup_region = _dedup_region(evidence.region)
        duplicate = self._find_duplicate(
            evidence.kind, dedup_region, visual_hash, dedup_hamming_distance, dedup_match)
        if duplicate is not None:
            metadata = json.loads(
                (duplicate / "metadata.json").read_text(encoding="utf-8-sig"))
            return ReviewCase(metadata["case_id"], str(duplicate),
                              ReviewStatus(metadata["status"]),
                              format_review_message(metadata), created=False)
        case_id = self._next_case_id()
        target = self.cases_root / case_id
        temporary = self.cases_root / f".{case_id}.{uuid4().hex}.tmp"
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            write_image(temporary / "full_frame.png", evidence.full_frame)
            write_image(temporary / "crop.png", evidence.crop)
            files = ["full_frame.png", "crop.png", "state.json", "metadata.json"]
            if evidence.previous_frame is not None:
                write_image(temporary / "previous_frame.png", evidence.previous_frame)
                files.append("previous_frame.png")
            if evidence.next_frame is not None:
                write_image(temporary / "next_frame.png", evidence.next_frame)
                files.append("next_frame.png")
            _atomic_json(temporary / "state.json", evidence.state)
            crop_hash = sha256((temporary / "crop.png").read_bytes()).hexdigest()
            metadata = {
                "schema_version": 1,
                "case_id": case_id,
                "status": ReviewStatus.PENDING.value,
                "kind": evidence.kind.value,
                "region": evidence.region,
                "dedup_region": dedup_region,
                "timestamp": evidence.timestamp,
                "frame_index": evidence.frame_index,
                "source": evidence.source,
                "player_count": evidence.metadata.get("player_count"),
                "mode": evidence.metadata.get("mode"),
                "model_version": evidence.model_version,
                "reasons": [reason.value for reason in evidence.reasons],
                "candidates": [candidate.to_dict() for candidate in evidence.candidates],
                "crop_sha256": crop_hash,
                "visual_hash": f"{visual_hash:016x}",
                "created_at": _utc_now(),
                "files": sorted(files),
                "context": evidence.metadata,
            }
            _atomic_json(temporary / "metadata.json", metadata)
            temporary.replace(target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        message = format_review_message(metadata)
        return ReviewCase(case_id, str(target), ReviewStatus.PENDING, message)

    def _find_duplicate(self, kind: ReviewKind, dedup_region: str, visual_hash: int,
                        maximum_distance: int, dedup_match: Callable | None = None) -> Path | None:
        if not self.cases_root.exists():
            return None
        for metadata_path in sorted(self.cases_root.glob("case_*/metadata.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
                old_dedup_region = metadata.get("dedup_region")
                if not old_dedup_region:
                    old_dedup_region = _dedup_region(metadata["region"])
                    metadata["dedup_region"] = old_dedup_region
                    _atomic_json(metadata_path, metadata)
                if metadata.get("visual_hash"):
                    old_hash = int(metadata["visual_hash"], 16)
                else:
                    old_hash = difference_hash(read_image(metadata_path.parent / "crop.png"))
                    metadata["visual_hash"] = f"{old_hash:016x}"
                    _atomic_json(metadata_path, metadata)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if (metadata.get("kind") == kind.value and
                    old_dedup_region == dedup_region and
                    hamming_distance(old_hash, visual_hash) <= maximum_distance):
                if dedup_match is not None:
                    try:
                        if not dedup_match(read_image(metadata_path.parent / "crop.png")):
                            continue
                    except (OSError, ValueError):
                        continue
                return metadata_path.parent
        return None

    def label(self, case_id: str, label: str, *, verified_by: str) -> dict:
        if not re.fullmatch(r"case_\d{6}", case_id):
            raise ValueError("Invalid review case id")
        if not verified_by.strip():
            raise ValueError("verified_by is required")
        case = self.cases_root / case_id
        metadata_path = case / "metadata.json"
        if not metadata_path.is_file() or not (case / "crop.png").is_file():
            raise FileNotFoundError(f"Review case is incomplete: {case_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata.get("status") != ReviewStatus.PENDING.value:
            raise ValueError("Only pending review cases can be labeled")
        kind = ReviewKind(metadata["kind"])
        if (metadata.get("player_count") not in (3, 4) or
                not isinstance(metadata.get("mode"), str) or
                not metadata["mode"].strip()):
            raise ValueError(
                "Review case lacks player_count/mode provenance; backfill it before labeling")
        normalized = validate_label(kind, label)
        relative = Path("verified") / _DATASET_GROUP[kind] / normalized / f"{case_id}.png"
        destination = self.dataset_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Verified sample already exists: {destination}")
        shutil.copy2(case / "crop.png", destination)
        verified_at = _utc_now()
        suggestions = metadata.get("reviewer_suggestions") or []
        if suggestions:
            latest = suggestions[-1]
            latest["status"] = (
                "confirmed_by_human"
                if normalized in (latest.get("labels") or [])
                else "corrected_by_human")
            latest["human_label"] = normalized
            latest["confirmed_at"] = verified_at
        metadata.update({
            "status": ReviewStatus.VERIFIED.value,
            "label": normalized,
            "verified_by": verified_by.strip(),
            "verified_at": verified_at,
            "dataset_path": relative.as_posix(),
        })
        _atomic_json(metadata_path, metadata)
        entry = {
            key: metadata.get(key) for key in (
                "case_id", "source", "player_count", "mode", "kind",
                "label", "verified_by",
                "verified_at", "model_version", "candidates",
                "reviewer_suggestions", "crop_sha256", "dataset_path")
        }
        entry["split"] = "unassigned"
        self._append_unique(self.dataset_root / "manifests" / "verified_samples.json",
                            entry, "case_id")
        queue_entry = {
            "case_id": case_id, "status": "pending", "label": normalized,
            "kind": kind.value, "player_count": metadata.get("player_count"),
            "mode": metadata.get("mode"),
            "dataset_path": relative.as_posix(),
        }
        self._append_unique(self.dataset_root / "manifests" / "training_queue.json",
                            queue_entry, "case_id")
        return metadata

    def suggest(self, case_id: str, labels: list[str] | tuple[str, ...], *,
                suggested_by: str, note: str | None = None) -> dict:
        """Record ordered reviewer candidates without creating a ground-truth label."""
        if not re.fullmatch(r"case_\d{6}", case_id):
            raise ValueError("Invalid review case id")
        if not suggested_by.strip():
            raise ValueError("suggested_by is required")
        metadata_path = self.cases_root / case_id / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Review case is incomplete: {case_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata.get("status") != ReviewStatus.PENDING.value:
            raise ValueError("Suggestions can only be added to pending review cases")
        kind = ReviewKind(metadata["kind"])
        normalized = []
        for label in labels:
            candidate = validate_label(kind, label)
            if candidate not in normalized:
                normalized.append(candidate)
        if not normalized:
            raise ValueError("At least one suggestion label is required")
        suggestion = {
            "labels": normalized,
            "suggested_by": suggested_by.strip(),
            "suggested_at": _utc_now(),
            "status": "awaiting_human_confirmation",
        }
        if note is not None and note.strip():
            suggestion["note"] = note.strip()
        history = metadata.setdefault("reviewer_suggestions", [])
        if history and history[-1].get("labels") == normalized:
            return metadata
        history.append(suggestion)
        _atomic_json(metadata_path, metadata)
        return metadata

    def backfill_source_context(
            self, source: str, *, player_count: int, mode: str) -> dict:
        """Migrate legacy cases without guessing context from visual content."""
        if not source.strip() or player_count not in (3, 4) or not mode.strip():
            raise ValueError("source, player_count, and mode are required")
        updated_case_ids = []
        for metadata_path in sorted(self.cases_root.glob("case_*/metadata.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            if metadata.get("source") != source:
                continue
            existing = (metadata.get("player_count"), metadata.get("mode"))
            if existing != (None, None) and existing != (player_count, mode):
                raise ValueError(
                    f"Conflicting source context for {metadata['case_id']}: {existing}")
            if existing == (player_count, mode):
                continue
            metadata["player_count"] = player_count
            metadata["mode"] = mode.strip()
            _atomic_json(metadata_path, metadata)
            updated_case_ids.append(metadata["case_id"])
        for name in ("verified_samples.json", "training_queue.json"):
            path = self.dataset_root / "manifests" / name
            if not path.exists():
                continue
            values = json.loads(path.read_text(encoding="utf-8-sig"))
            changed = False
            for entry in values:
                if (entry.get("case_id") in updated_case_ids or
                        entry.get("source") == source):
                    entry["player_count"] = player_count
                    entry["mode"] = mode.strip()
                    changed = True
            if changed:
                _atomic_json(path, values)
        return {"source": source, "player_count": player_count,
                "mode": mode.strip(), "updated_case_ids": updated_case_ids}

    def reject(self, case_id: str, *, rejected_by: str, reason: str) -> dict:
        """Reject an unusable crop without admitting it to any dataset split."""
        if not re.fullmatch(r"case_\d{6}", case_id):
            raise ValueError("Invalid review case id")
        if not rejected_by.strip():
            raise ValueError("rejected_by is required")
        if not reason.strip():
            raise ValueError("rejection reason is required")
        metadata_path = self.cases_root / case_id / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Review case is incomplete: {case_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata.get("status") != ReviewStatus.PENDING.value:
            raise ValueError("Only pending review cases can be rejected")
        metadata.update({
            "status": ReviewStatus.REJECTED.value,
            "rejected_by": rejected_by.strip(),
            "rejected_at": _utc_now(),
            "rejection_reason": reason.strip(),
        })
        _atomic_json(metadata_path, metadata)
        return metadata

    @staticmethod
    def _append_unique(path: Path, entry: dict, identity: str) -> None:
        values = (json.loads(path.read_text(encoding="utf-8-sig"))
                  if path.exists() else [])
        if any(item.get(identity) == entry[identity] for item in values):
            raise ValueError(f"Duplicate manifest entry: {entry[identity]}")
        values.append(entry)
        _atomic_json(path, values)


def format_review_message(metadata: dict) -> str:
    status = ReviewStatus(metadata.get("status", ReviewStatus.PENDING.value))
    if status == ReviewStatus.VERIFIED:
        return "\n".join([
            "Mahjong Agent encountered again a visual state that has been verified but the current model still cannot recognize",
            f"Case: {metadata['case_id']}",
            f"Type: {metadata['kind']} / {metadata['region']}",
            f"Verified label: {metadata.get('label', 'unknown')}",
            "No need to re-label; keep SAFE_PAUSE and wait for batch training and the regression gate.",
        ])
    if status == ReviewStatus.REJECTED:
        return "\n".join([
            "Mahjong Agent encountered again a visual sample that has been rejected",
            f"Case: {metadata['case_id']}",
            f"Type: {metadata['kind']} / {metadata['region']}",
            f"Rejection reason: {metadata.get('rejection_reason', 'unusable sample')}",
            "No need to re-label; a new case requires new, usable contextual evidence.",
        ])
    suggestions = metadata.get("reviewer_suggestions") or []
    suggestion_lines = []
    if suggestions:
        latest = suggestions[-1]
        suggestion_lines = [
            "AI visual review candidate (not yet human ground truth):",
            *[f"{index}. {label}" for index, label in
              enumerate(latest.get("labels") or (), 1)],
        ]
        if latest.get("note"):
            suggestion_lines.append(f"rationale: {latest['note']}")
    candidates = metadata.get("candidates") or []
    candidate_lines = [f"{index}. {item['label']}  {item['score']:.3f}"
                       for index, item in enumerate(candidates, 1)]
    if not candidate_lines:
        candidate_lines = ["(no reliable candidate)"]
    reasons = ", ".join(metadata.get("reasons") or ["unknown"])
    return "\n".join([
        "Mahjong Agent encountered a persistently unknown visual state",
        f"Case: {metadata['case_id']}",
        f"Type: {metadata['kind']} / {metadata['region']}",
        f"Frame: {metadata['frame_index']}  Timestamp: {metadata['timestamp']:.3f}s",
        f"Reasons: {reasons}",
        *suggestion_lines,
        "Model raw candidate:",
        *candidate_lines,
        "Please open the local manual labeling interface and confirm using the crop, full frame, and previous/next frames.",
        "Launch: python review_labeler.py --kind tile",
        "Alternative command format:",
        f"/label {metadata['case_id']} <label>",
        f"If the crop is not a valid sample: /reject {metadata['case_id']} <reason>",
    ])
