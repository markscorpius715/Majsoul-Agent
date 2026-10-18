"""Scalable human-review store for immutable captured crop corpora."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import heapq
from pathlib import Path
import re
import shutil
import sqlite3
from uuid import uuid4

from .schemas import ReviewStatus
from .store import validate_label
from .schemas import ReviewKind


_SAFE_CORPUS_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_SAMPLE_ID = re.compile(r"^crop_[A-Za-z0-9]+$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CropReviewRecord:
    """One raw crop plus its separate, mutable human-review state."""

    sample_id: str
    path: Path
    metadata: dict
    status: str = ReviewStatus.PENDING.value
    label: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    rejection_reason: str | None = None
    dataset_path: str | None = None
    prediction_label: str | None = None
    prediction_confidence: float | None = None
    prediction_model: str | None = None
    predicted_at: str | None = None
    prediction_status: str | None = None
    prediction_human_label: str | None = None


class CorpusReviewStore:
    """Review a large raw corpus without mutating its source manifest.

    SQLite is the canonical annotation and training-queue audit for captured
    corpora.  A missing row means ``pending``.  This keeps per-label writes
    transactional and fast even when the immutable JSON source manifest holds
    tens of thousands of samples.
    """

    def __init__(self, manifest_path: str | Path, dataset_root: str | Path,
                 *, database_path: str | Path | None = None):
        self.manifest_path = Path(manifest_path)
        self.corpus_root = self.manifest_path.parent
        self.dataset_root = Path(dataset_root)
        self.manifest = json.loads(
            self.manifest_path.read_text(encoding="utf-8-sig"))
        self.corpus_id = str(self.manifest.get("corpus_id", ""))
        if not _SAFE_CORPUS_ID.fullmatch(self.corpus_id):
            raise ValueError("Crop corpus has an unsafe or missing corpus_id")
        if self.manifest.get("label_status") != "unverified":
            raise ValueError("Crop corpus must remain an unverified raw source")
        if self.manifest.get("training_allowed") is not False:
            raise ValueError("Raw crop corpus must not be directly training-eligible")
        samples = self.manifest.get("samples")
        if not isinstance(samples, list):
            raise ValueError("Crop corpus samples must be a list")
        self.samples: dict[str, dict] = {}
        self.sample_order: list[str] = []
        for sample in samples:
            sample_id = sample.get("id") if isinstance(sample, dict) else None
            if not isinstance(sample_id, str) or not _SAFE_SAMPLE_ID.fullmatch(sample_id):
                raise ValueError(f"Unsafe crop sample id: {sample_id!r}")
            if sample_id in self.samples:
                raise ValueError(f"Duplicate crop sample id: {sample_id}")
            relative = Path(str(sample.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe crop path for {sample_id}")
            if sample.get("player_count") not in (3, 4) or not sample.get("mode"):
                raise ValueError(f"Crop sample lacks game-mode provenance: {sample_id}")
            self.samples[sample_id] = sample
            self.sample_order.append(sample_id)

        default_database = (
            self.dataset_root / "manifests" / "corpus_reviews" /
            f"{self.corpus_id}.sqlite3")
        self.database_path = (
            Path(database_path) if database_path is not None else default_database)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS corpus_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    sample_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('verified', 'rejected')),
                    label TEXT,
                    reviewed_by TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    rejection_reason TEXT,
                    dataset_path TEXT,
                    crop_sha256 TEXT NOT NULL,
                    source TEXT NOT NULL,
                    player_count INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    region TEXT NOT NULL,
                    training_status TEXT CHECK(
                        training_status IS NULL OR training_status = 'pending'),
                    CHECK (
                        (status = 'verified' AND label IS NOT NULL AND
                         dataset_path IS NOT NULL AND training_status = 'pending' AND
                         rejection_reason IS NULL) OR
                        (status = 'rejected' AND label IS NULL AND
                         dataset_path IS NULL AND training_status IS NULL AND
                         rejection_reason IS NOT NULL)
                    )
                );
                CREATE INDEX IF NOT EXISTS reviews_status_index
                    ON reviews(status);
                CREATE TABLE IF NOT EXISTS predictions (
                    sample_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    predicted_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN (
                        'awaiting_human_confirmation', 'confirmed_by_human',
                        'corrected_by_human', 'rejected_by_human')),
                    human_label TEXT
                );
                CREATE INDEX IF NOT EXISTS predictions_label_index
                    ON predictions(label);
                CREATE TABLE IF NOT EXISTS review_corrections (
                    correction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id TEXT NOT NULL,
                    previous_status TEXT NOT NULL,
                    previous_label TEXT,
                    previous_dataset_path TEXT,
                    new_status TEXT NOT NULL,
                    new_label TEXT,
                    new_dataset_path TEXT,
                    corrected_by TEXT NOT NULL,
                    corrected_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS review_corrections_sample_index
                    ON review_corrections(sample_id, correction_id);
                CREATE TABLE IF NOT EXISTS ai_review_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN (
                        'keep', 'correct', 'uncertain')),
                    reason TEXT NOT NULL,
                    previous_prediction TEXT,
                    new_prediction TEXT,
                    UNIQUE(run_id, sample_id)
                );
                CREATE INDEX IF NOT EXISTS ai_review_sample_index
                    ON ai_review_audit(sample_id, audit_id);
            """)
            existing = dict(connection.execute(
                "SELECT key, value FROM corpus_metadata"))
            identity = {
                "schema_version": "1",
                "corpus_id": self.corpus_id,
                "source_manifest": str(self.manifest_path.resolve()),
                "source_sample_count": str(len(self.samples)),
            }
            if existing:
                for key, value in identity.items():
                    if existing.get(key) != value:
                        raise ValueError(
                            f"Corpus review database identity mismatch: {key}")
            else:
                connection.executemany(
                    "INSERT INTO corpus_metadata(key, value) VALUES (?, ?)",
                    identity.items())

    def _reviews(self) -> dict[str, sqlite3.Row]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM reviews").fetchall()
        unknown = [row["sample_id"] for row in rows
                   if row["sample_id"] not in self.samples]
        if unknown:
            raise ValueError(
                f"Review database references unknown crop: {unknown[0]}")
        return {row["sample_id"]: row for row in rows}

    def _predictions(self) -> dict[str, sqlite3.Row]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM predictions").fetchall()
        unknown = [row["sample_id"] for row in rows
                   if row["sample_id"] not in self.samples]
        if unknown:
            raise ValueError(
                f"Prediction database references unknown crop: {unknown[0]}")
        return {row["sample_id"]: row for row in rows}

    def counts(self) -> Counter:
        with self._connect() as connection:
            reviewed = Counter(dict(connection.execute(
                "SELECT status, COUNT(*) FROM reviews GROUP BY status")))
        reviewed[ReviewStatus.PENDING.value] = (
            len(self.samples) - reviewed[ReviewStatus.VERIFIED.value] -
            reviewed[ReviewStatus.REJECTED.value])
        return reviewed

    def prediction_counts(self) -> Counter:
        with self._connect() as connection:
            labels = Counter(dict(connection.execute(
                "SELECT label, COUNT(*) FROM predictions GROUP BY label")))
            total = connection.execute(
                "SELECT COUNT(*) FROM predictions").fetchone()[0]
        labels["classified"] = total
        labels["unclassified"] = len(self.samples) - total
        return labels

    def verified_label_counts(self) -> Counter:
        """Return counts keyed by the final human-verified tile label."""
        with self._connect() as connection:
            return Counter(dict(connection.execute("""
                SELECT label, COUNT(*)
                FROM reviews
                WHERE status = 'verified'
                GROUP BY label
            """)))

    @staticmethod
    def position_bucket(sample: dict) -> tuple[str, str, str]:
        """Build a stable physical-position bucket from corpus metadata."""
        return tuple(str(sample.get(field) or "unknown")
                     for field in ("region", "position", "orientation"))

    def _verified_position_counts(self) -> Counter:
        counts: Counter = Counter()
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT sample_id, label
                FROM reviews
                WHERE status = 'verified'
            """).fetchall()
        for row in rows:
            counts[(row["label"], self.position_bucket(
                self.samples[row["sample_id"]]))] += 1
        return counts

    def _balance_pending_by_position(
            self, records: list[CropReviewRecord]) -> list[CropReviewRecord]:
        """Interleave least-covered label/position groups first.

        Existing verified coverage seeds the scheduler.  Each emitted candidate
        increments its projected group count, so long queues cannot monopolize
        the front of the review list.
        """
        position_counts = self._verified_position_counts()
        label_counts = self.verified_label_counts()
        groups: dict[tuple[str, tuple[str, str, str]], deque] = defaultdict(deque)
        first_seen: dict[tuple[str, tuple[str, str, str]], int] = {}
        for index, record in enumerate(records):
            label = record.prediction_label or "unclassified"
            key = (label, self.position_bucket(record.metadata))
            groups[key].append(record)
            first_seen.setdefault(key, index)

        heap = []
        projected_position = Counter(position_counts)
        projected_label = Counter(label_counts)
        for key in groups:
            label, bucket = key
            heapq.heappush(heap, (
                projected_position[(label, bucket)], projected_label[label],
                first_seen[key], label, bucket))

        ordered = []
        while heap:
            _position_total, _label_total, stable_index, label, bucket = (
                heapq.heappop(heap))
            key = (label, bucket)
            ordered.append(groups[key].popleft())
            projected_position[key] += 1
            projected_label[label] += 1
            if groups[key]:
                heapq.heappush(heap, (
                    projected_position[key], projected_label[label],
                    stable_index, label, bucket))
        return ordered

    def _order_pending_by_frequency(
            self, records: list[CropReviewRecord]) -> list[CropReviewRecord]:
        """Sort high-frequency crops first and balance positions within ties."""
        position_balanced = self._balance_pending_by_position(records)

        def frequency(record: CropReviewRecord) -> float:
            value = record.metadata.get("occurrence_count", 1)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return 0
            return max(0, float(value))

        return sorted(position_balanced, key=frequency, reverse=True)

    def regions(self) -> tuple[str, ...]:
        return tuple(sorted({str(sample.get("region", ""))
                            for sample in self.samples.values()}))

    def scan(self, *, status: str = "pending", region: str = "all",
             issue_filter: str = "all", query: str = "",
             classification: str = "all",
             predicted_label: str = "all",
             ai_reviewed_only: bool = False,
             order: str = "manifest") -> list[CropReviewRecord]:
        if status not in {"all", *(item.value for item in ReviewStatus)}:
            raise ValueError(f"Unsupported status filter: {status}")
        if region != "all" and region not in self.regions():
            raise ValueError(f"Unsupported region filter: {region}")
        if issue_filter not in {"all", "clean", "issues"}:
            raise ValueError(f"Unsupported issue filter: {issue_filter}")
        if classification not in {"all", "classified", "unclassified"}:
            raise ValueError(f"Unsupported classification filter: {classification}")
        pending_orders = {"balanced_positions", "frequency_balanced_positions"}
        reviewed_orders = {"reviewed_at_desc"}
        if order not in {"manifest", *pending_orders, *reviewed_orders}:
            raise ValueError(f"Unsupported review order: {order}")
        if order in pending_orders and status != ReviewStatus.PENDING.value:
            raise ValueError("Position balancing is only available for pending reviews")
        if order in reviewed_orders and status not in {
                ReviewStatus.VERIFIED.value, ReviewStatus.REJECTED.value}:
            raise ValueError(
                "Reviewed-time ordering is only available for reviewed samples")
        if predicted_label != "all" and predicted_label not in {
                "reject", *(f"{rank}{suit}" for suit in "mps"
                            for rank in "0123456789" if rank != "0" or suit in "mps"),
                *(f"{rank}z" for rank in "1234567")}:
            raise ValueError(f"Unsupported predicted label: {predicted_label}")
        query = query.strip().lower()
        reviews = self._reviews()
        predictions = self._predictions()
        with self._connect() as connection:
            ai_ids = ({row[0] for row in connection.execute(
                "SELECT DISTINCT sample_id FROM ai_review_audit")}
                if ai_reviewed_only else None)
        records: list[CropReviewRecord] = []
        for sample_id in self.sample_order:
            if ai_ids is not None and sample_id not in ai_ids:
                continue
            sample = self.samples[sample_id]
            review = reviews.get(sample_id)
            prediction = predictions.get(sample_id)
            current_status = (
                review["status"] if review is not None
                else ReviewStatus.PENDING.value)
            if status != "all" and current_status != status:
                continue
            if classification == "classified" and prediction is None:
                continue
            if classification == "unclassified" and prediction is not None:
                continue
            if (predicted_label != "all" and
                    (prediction is None or prediction["label"] != predicted_label)):
                continue
            if region != "all" and sample.get("region") != region:
                continue
            has_issues = bool(sample.get("segmentation_issues"))
            if issue_filter == "clean" and has_issues:
                continue
            if issue_filter == "issues" and not has_issues:
                continue
            if query and query not in sample_id.lower():
                continue
            records.append(CropReviewRecord(
                sample_id=sample_id,
                path=self.corpus_root / sample["path"],
                metadata=sample,
                status=current_status,
                label=review["label"] if review is not None else None,
                reviewed_by=(review["reviewed_by"]
                             if review is not None else None),
                reviewed_at=(review["reviewed_at"]
                             if review is not None else None),
                rejection_reason=(review["rejection_reason"]
                                  if review is not None else None),
                dataset_path=(review["dataset_path"]
                              if review is not None else None),
                prediction_label=(prediction["label"]
                                  if prediction is not None else None),
                prediction_confidence=(prediction["confidence"]
                                       if prediction is not None else None),
                prediction_model=(prediction["model"]
                                  if prediction is not None else None),
                predicted_at=(prediction["predicted_at"]
                              if prediction is not None else None),
                prediction_status=(prediction["status"]
                                   if prediction is not None else None),
                prediction_human_label=(prediction["human_label"]
                                        if prediction is not None else None),
            ))
        if order == "balanced_positions":
            return self._balance_pending_by_position(records)
        if order == "frequency_balanced_positions":
            return self._order_pending_by_frequency(records)
        if order == "reviewed_at_desc":
            return sorted(
                records, key=lambda record: record.reviewed_at or "", reverse=True)
        return records

    def get(self, sample_id: str) -> CropReviewRecord:
        if sample_id not in self.samples:
            raise KeyError(f"Unknown crop sample: {sample_id}")
        reviews = self._reviews()
        predictions = self._predictions()
        sample = self.samples[sample_id]
        review = reviews.get(sample_id)
        prediction = predictions.get(sample_id)
        return CropReviewRecord(
            sample_id=sample_id,
            path=self.corpus_root / sample["path"],
            metadata=sample,
            status=review["status"] if review else ReviewStatus.PENDING.value,
            label=review["label"] if review else None,
            reviewed_by=review["reviewed_by"] if review else None,
            reviewed_at=review["reviewed_at"] if review else None,
            rejection_reason=review["rejection_reason"] if review else None,
            dataset_path=review["dataset_path"] if review else None,
            prediction_label=prediction["label"] if prediction else None,
            prediction_confidence=prediction["confidence"] if prediction else None,
            prediction_model=prediction["model"] if prediction else None,
            predicted_at=prediction["predicted_at"] if prediction else None,
            prediction_status=prediction["status"] if prediction else None,
            prediction_human_label=(prediction["human_label"]
                                    if prediction else None))

    def unclassified(self) -> list[CropReviewRecord]:
        """Return unreviewed samples that still need a model classification."""
        return self.scan(
            status=ReviewStatus.PENDING.value,
            classification="unclassified")

    def save_predictions(self, values: list[dict], *, model: str,
                         prompt_version: str) -> None:
        """Atomically persist one validated model batch as review suggestions."""
        if not model.strip() or not prompt_version.strip():
            raise ValueError("model and prompt_version are required")
        if not values:
            raise ValueError("Prediction batch must not be empty")
        seen = set()
        rows = []
        predicted_at = _utc_now()
        for value in values:
            sample_id = value.get("id")
            if sample_id not in self.samples or sample_id in seen:
                raise ValueError(f"Invalid or duplicate prediction id: {sample_id}")
            seen.add(sample_id)
            raw_label = value.get("label")
            label = ("reject" if raw_label == "reject" else
                     validate_label(ReviewKind.TILE, str(raw_label)))
            confidence = value.get("confidence")
            if (type(confidence) not in (float, int) or
                    not 0 <= float(confidence) <= 1):
                raise ValueError(f"Invalid prediction confidence: {sample_id}")
            rows.append((
                sample_id, label, float(confidence), model.strip(),
                prompt_version.strip(), predicted_at,
                "awaiting_human_confirmation"))
        with self._connect() as connection:
            reviewed = {row[0] for row in connection.execute(
                "SELECT sample_id FROM reviews WHERE sample_id IN (%s)" %
                ",".join("?" for _ in rows), tuple(row[0] for row in rows))}
            if reviewed:
                raise ValueError(
                    f"Cannot classify already reviewed crop: {sorted(reviewed)[0]}")
            connection.executemany("""
                INSERT INTO predictions(
                    sample_id, label, confidence, model, prompt_version,
                    predicted_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sample_id) DO NOTHING
            """, rows)

    def latest_ai_review(self, sample_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ai_review_audit WHERE sample_id=? ORDER BY audit_id DESC LIMIT 1",
                (sample_id,)).fetchone()
        return dict(row) if row else None

    def ai_review(self, sample_id: str, *, run_id: str, reviewer: str,
                  decision: str, reason: str, expected_label: str | None,
                  label: str | None = None, confidence: float | None = None) -> str:
        """Audit an AI inspection; corrections remain unverified predictions.

        Lock before checking human state and the expected prediction. A review
        completed by the human since the sheet was generated always wins.
        Replaying a run/sample is idempotent. Uncertain inspections change no label.
        """
        if sample_id not in self.samples:
            raise KeyError(sample_id)
        if not run_id.strip() or not reviewer.strip() or not reason.strip():
            raise ValueError("run_id, reviewer and reason are required")
        if decision not in {"keep", "correct", "uncertain"}:
            raise ValueError("Unsupported AI decision")
        if decision == "correct":
            label = ("reject" if label == "reject" else
                     validate_label(ReviewKind.TILE, str(label)))
            if label == expected_label:
                raise ValueError("Correction must change the prediction")
            if type(confidence) not in (float, int) or not 0 <= confidence <= 1:
                raise ValueError("Correction requires confidence in [0, 1]")
        elif label is not None or confidence is not None:
            raise ValueError("Only corrections may supply a new prediction")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                    "SELECT 1 FROM reviews WHERE sample_id = ?", (sample_id,)
            ).fetchone():
                return "skipped_human_reviewed"
            if connection.execute(
                    "SELECT 1 FROM ai_review_audit WHERE run_id = ? AND sample_id = ?",
                    (run_id, sample_id)).fetchone():
                return "already_recorded"
            row = connection.execute(
                "SELECT * FROM predictions WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            previous = dict(row) if row else None
            if (previous["label"] if previous else None) != expected_label:
                return "skipped_prediction_changed"
            if previous and previous["status"] != "awaiting_human_confirmation":
                raise ValueError("Prediction is not awaiting human confirmation")
            now = _utc_now()
            if decision == "correct":
                connection.execute("""
                    INSERT INTO predictions(sample_id, label, confidence, model,
                        prompt_version, predicted_at, status, human_label)
                    VALUES (?, ?, ?, ?, ?, ?, 'awaiting_human_confirmation', NULL)
                    ON CONFLICT(sample_id) DO UPDATE SET
                        label=excluded.label, confidence=excluded.confidence,
                        model=excluded.model, prompt_version=excluded.prompt_version,
                        predicted_at=excluded.predicted_at,
                        status='awaiting_human_confirmation', human_label=NULL
                """, (sample_id, label, confidence, reviewer, run_id, now))
            current = connection.execute(
                "SELECT * FROM predictions WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            connection.execute("""
                INSERT INTO ai_review_audit(run_id, sample_id, reviewer, reviewed_at,
                    decision, reason, previous_prediction, new_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, sample_id, reviewer, now, decision, reason,
                  json.dumps(previous, ensure_ascii=False),
                  json.dumps(dict(current) if current else None, ensure_ascii=False)))
        return decision

    def _pending_sample(self, sample_id: str) -> tuple[dict, Path, str]:
        if sample_id not in self.samples:
            raise KeyError(f"Unknown crop sample: {sample_id}")
        sample = self.samples[sample_id]
        source = self.corpus_root / sample["path"]
        if not source.is_file():
            raise FileNotFoundError(f"Crop image is missing: {source}")
        crop_hash = sha256(source.read_bytes()).hexdigest()
        if crop_hash != sample.get("crop_sha256"):
            raise ValueError(f"Crop content hash changed: {sample_id}")
        with self._connect() as connection:
            if connection.execute(
                    "SELECT 1 FROM reviews WHERE sample_id = ?", (sample_id,)).fetchone():
                raise ValueError("Only pending crop samples can be reviewed")
        return sample, source, crop_hash

    def label(self, sample_id: str, label: str, *, verified_by: str) -> CropReviewRecord:
        reviewer = verified_by.strip()
        if not reviewer:
            raise ValueError("verified_by is required")
        normalized = validate_label(ReviewKind.TILE, label)
        sample, source, crop_hash = self._pending_sample(sample_id)
        relative = Path("verified") / "tiles" / normalized / f"{sample_id}.png"
        destination = self.dataset_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Verified sample already exists: {destination}")
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            reviewed_at = _utc_now()
            try:
                with self._connect() as connection:
                    connection.execute("""
                        INSERT INTO reviews(
                            sample_id, status, label, reviewed_by, reviewed_at,
                            rejection_reason, dataset_path, crop_sha256, source,
                            player_count, mode, region, training_status)
                        VALUES (?, 'verified', ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, 'pending')
                    """, (
                        sample_id, normalized, reviewer, reviewed_at,
                        relative.as_posix(), crop_hash, sample["source"],
                        sample["player_count"], sample["mode"], sample["region"]))
                    prediction = connection.execute(
                        "SELECT label FROM predictions WHERE sample_id = ?",
                        (sample_id,)).fetchone()
                    if prediction is not None:
                        connection.execute("""
                            UPDATE predictions
                            SET status = ?, human_label = ?
                            WHERE sample_id = ?
                        """, (
                            "confirmed_by_human"
                            if prediction["label"] == normalized
                            else "corrected_by_human",
                            normalized, sample_id))
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        finally:
            temporary.unlink(missing_ok=True)
        return self.get(sample_id)

    def reject(self, sample_id: str, *, rejected_by: str,
               reason: str) -> CropReviewRecord:
        reviewer = rejected_by.strip()
        reason = reason.strip()
        if not reviewer:
            raise ValueError("rejected_by is required")
        if not reason:
            raise ValueError("rejection reason is required")
        if len(reason) > 500:
            raise ValueError("rejection reason is too long")
        sample, _source, crop_hash = self._pending_sample(sample_id)
        with self._connect() as connection:
            connection.execute("""
                INSERT INTO reviews(
                    sample_id, status, label, reviewed_by, reviewed_at,
                    rejection_reason, dataset_path, crop_sha256, source,
                    player_count, mode, region, training_status)
                VALUES (?, 'rejected', NULL, ?, ?, ?, NULL, ?, ?, ?, ?, ?, NULL)
            """, (
                sample_id, reviewer, _utc_now(), reason, crop_hash,
                sample["source"], sample["player_count"], sample["mode"],
                sample["region"]))
            connection.execute("""
                UPDATE predictions
                SET status = 'rejected_by_human', human_label = NULL
                WHERE sample_id = ?
            """, (sample_id,))
        return self.get(sample_id)

    def correction_history(self, sample_id: str) -> list[dict]:
        """Return the append-only correction audit for one reviewed crop."""
        if sample_id not in self.samples:
            raise KeyError(f"Unknown crop sample: {sample_id}")
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT * FROM review_corrections
                WHERE sample_id = ?
                ORDER BY correction_id
            """, (sample_id,)).fetchall()
        return [dict(row) for row in rows]

    def correct_verified(self, sample_id: str, corrected_label: str, *,
                         corrected_by: str) -> CropReviewRecord:
        """Relabel or invalidate a verified crop while preserving an audit trail."""
        reviewer = corrected_by.strip()
        if not reviewer:
            raise ValueError("corrected_by is required")
        if sample_id not in self.samples:
            raise KeyError(f"Unknown crop sample: {sample_id}")
        normalized = (
            "reject" if corrected_label == "reject"
            else validate_label(ReviewKind.TILE, corrected_label))

        with self._connect() as connection:
            review = connection.execute(
                "SELECT * FROM reviews WHERE sample_id = ?", (sample_id,)
            ).fetchone()
        if review is None or review["status"] != ReviewStatus.VERIFIED.value:
            raise ValueError("Only verified crop samples can be corrected")
        if normalized == review["label"]:
            raise ValueError("Corrected label must differ from the verified label")

        source_relative = Path(review["dataset_path"])
        if source_relative.is_absolute() or ".." in source_relative.parts:
            raise ValueError(f"Unsafe verified dataset path: {source_relative}")
        source = self.dataset_root / source_relative
        if not source.is_file():
            raise FileNotFoundError(f"Verified sample is missing: {source}")
        if sha256(source.read_bytes()).hexdigest() != review["crop_sha256"]:
            raise ValueError(f"Verified sample content hash changed: {sample_id}")

        if normalized == "reject":
            new_status = ReviewStatus.REJECTED.value
            new_label = None
            destination_relative = Path("rejected") / "tiles" / f"{sample_id}.png"
        else:
            new_status = ReviewStatus.VERIFIED.value
            new_label = normalized
            destination_relative = (
                Path("verified") / "tiles" / normalized / f"{sample_id}.png")
        destination = self.dataset_root / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Corrected sample already exists: {destination}")

        corrected_at = _utc_now()
        source.replace(destination)
        try:
            with self._connect() as connection:
                if new_status == ReviewStatus.VERIFIED.value:
                    updated = connection.execute("""
                        UPDATE reviews
                        SET status = 'verified', label = ?, reviewed_by = ?,
                            reviewed_at = ?, rejection_reason = NULL,
                            dataset_path = ?, training_status = 'pending'
                        WHERE sample_id = ? AND status = 'verified'
                    """, (
                        new_label, reviewer, corrected_at,
                        destination_relative.as_posix(), sample_id))
                    prediction = connection.execute(
                        "SELECT label FROM predictions WHERE sample_id = ?",
                        (sample_id,)).fetchone()
                    if prediction is not None:
                        connection.execute("""
                            UPDATE predictions
                            SET status = ?, human_label = ?
                            WHERE sample_id = ?
                        """, (
                            "confirmed_by_human"
                            if prediction["label"] == new_label
                            else "corrected_by_human",
                            new_label, sample_id))
                else:
                    updated = connection.execute("""
                        UPDATE reviews
                        SET status = 'rejected', label = NULL, reviewed_by = ?,
                            reviewed_at = ?,
                            rejection_reason = 'verified_label_invalidated_by_human',
                            dataset_path = NULL, training_status = NULL
                        WHERE sample_id = ? AND status = 'verified'
                    """, (reviewer, corrected_at, sample_id))
                    connection.execute("""
                        UPDATE predictions
                        SET status = 'rejected_by_human', human_label = NULL
                        WHERE sample_id = ?
                    """, (sample_id,))
                if updated.rowcount != 1:
                    raise RuntimeError(
                        "Verified review changed concurrently; correction aborted")
                connection.execute("""
                    INSERT INTO review_corrections(
                        sample_id, previous_status, previous_label,
                        previous_dataset_path, new_status, new_label,
                        new_dataset_path, corrected_by, corrected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sample_id, review["status"], review["label"],
                    review["dataset_path"], new_status, new_label,
                    destination_relative.as_posix(), reviewer, corrected_at))
        except Exception:
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(source)
            except OSError as rollback_error:
                raise RuntimeError(
                    "Correction database write failed and file rollback also failed"
                ) from rollback_error
            raise
        return self.get(sample_id)

    def correct_rejected(self, sample_id: str, corrected_label: str, *,
                         corrected_by: str) -> CropReviewRecord:
        """Recover a mistakenly rejected crop into a verified tile label."""
        reviewer = corrected_by.strip()
        if not reviewer:
            raise ValueError("corrected_by is required")
        if sample_id not in self.samples:
            raise KeyError(f"Unknown crop sample: {sample_id}")
        if corrected_label == "reject":
            raise ValueError("Corrected label must differ from the rejected status")
        normalized = validate_label(ReviewKind.TILE, corrected_label)

        with self._connect() as connection:
            review = connection.execute(
                "SELECT * FROM reviews WHERE sample_id = ?", (sample_id,)
            ).fetchone()
        if review is None or review["status"] != ReviewStatus.REJECTED.value:
            raise ValueError("Only rejected crop samples can be corrected")

        sample = self.samples[sample_id]
        source = self.corpus_root / sample["path"]
        if not source.is_file():
            raise FileNotFoundError(f"Crop image is missing: {source}")
        if sha256(source.read_bytes()).hexdigest() != sample.get("crop_sha256"):
            raise ValueError(f"Crop content hash changed: {sample_id}")

        relative = Path("verified") / "tiles" / normalized / f"{sample_id}.png"
        destination = self.dataset_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Verified sample already exists: {destination}")
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            corrected_at = _utc_now()
            try:
                with self._connect() as connection:
                    updated = connection.execute("""
                        UPDATE reviews
                        SET status = 'verified', label = ?, reviewed_by = ?,
                            reviewed_at = ?, rejection_reason = NULL,
                            dataset_path = ?, training_status = 'pending'
                        WHERE sample_id = ? AND status = 'rejected'
                    """, (
                        normalized, reviewer, corrected_at,
                        relative.as_posix(), sample_id))
                    prediction = connection.execute(
                        "SELECT label FROM predictions WHERE sample_id = ?",
                        (sample_id,)).fetchone()
                    if prediction is not None:
                        connection.execute("""
                            UPDATE predictions
                            SET status = ?, human_label = ?
                            WHERE sample_id = ?
                        """, (
                            "confirmed_by_human"
                            if prediction["label"] == normalized
                            else "corrected_by_human",
                            normalized, sample_id))
                    if updated.rowcount != 1:
                        raise RuntimeError(
                            "Rejected review changed concurrently; correction aborted")
                    connection.execute("""
                        INSERT INTO review_corrections(
                            sample_id, previous_status, previous_label,
                            previous_dataset_path, new_status, new_label,
                            new_dataset_path, corrected_by, corrected_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sample_id, review["status"], review["label"],
                        review["dataset_path"], ReviewStatus.VERIFIED.value,
                        normalized, relative.as_posix(), reviewer, corrected_at))
            except Exception:
                destination.unlink(missing_ok=True)
                raise
        finally:
            temporary.unlink(missing_ok=True)
        return self.get(sample_id)
