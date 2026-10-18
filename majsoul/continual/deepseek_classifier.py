"""DeepSeek Flash vision classification for captured Mahjong tile crops."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import json
import math
import mimetypes
import os
from pathlib import Path
import time
from typing import Callable

import httpx

from ..strategy import ModelSettings
from .corpus_review import CorpusReviewStore, CropReviewRecord


TILE_LABELS = tuple(
    [f"{rank}{suit}" for suit in "mps" for rank in "123456789"] +
    [f"0{suit}" for suit in "mps"] +
    [f"{rank}z" for rank in "1234567"])
CLASSIFICATION_LABELS = (*TILE_LABELS, "reject")
PROMPT_VERSION = "deepseek-tile-v1"


class DeepSeekClassificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClassificationProgress:
    total: int
    completed: int
    failed: int
    started_at: float


class DeepSeekTileClassifier:
    """Strict image-to-label adapter; model output never becomes ground truth."""

    prompt_version = PROMPT_VERSION

    def __init__(self, settings: ModelSettings,
                 client: httpx.AsyncClient | None = None, *,
                 timeout: float = 120.0, proxy: str | None = None,
                 max_retries: int = 3):
        if not settings.api_key:
            raise DeepSeekClassificationError(
                "Missing DEEPSEEK_API_KEY/API_KEY for image classification")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Classification timeout must be finite and positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.settings = settings
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = client or httpx.AsyncClient(
            timeout=timeout, proxy=proxy, follow_redirects=False,
            trust_env=False)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    @staticmethod
    def _image_part(record: CropReviewRecord) -> dict:
        data = record.path.read_bytes()
        if not data:
            raise DeepSeekClassificationError(
                f"Empty crop image: {record.sample_id}")
        mime = mimetypes.guess_type(record.path.name)[0] or "image/png"
        if mime not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
            raise DeepSeekClassificationError(
                f"Unsupported crop image type: {record.sample_id}")
        encoded = base64.b64encode(data).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{encoded}",
                "detail": "original",
            },
        }

    def payload(self, records: list[CropReviewRecord]) -> dict:
        if not records or len({record.sample_id for record in records}) != len(records):
            raise ValueError("Classification batch must contain unique samples")
        labels = ", ".join(CLASSIFICATION_LABELS)
        content: list[dict] = [{
            "type": "text",
            "text": (
                "Classify every following Mahjong Soul crop independently. "
                "Return exactly one item for every supplied id. Allowed labels: "
                f"{labels}. Numbered suited tiles use 1-9 plus 0m/0p/0s for red fives; "
                "honors are 1z east, 2z south, 3z west, 4z north, 5z white, "
                "6z green, 7z red. Use reject when the crop is not exactly one "
                "recognizable tile face (animation, avatar, UI, tile back, severe "
                "occlusion, or genuinely ambiguous). Do not infer a hidden tile from "
                "screen position. Return JSON only as "
                '{"items":[{"id":"exact id","label":"allowed label",'
                '"confidence":0.0}]}.'),
        }]
        for record in records:
            metadata = record.metadata
            content.append({
                "type": "text",
                "text": (
                    f"id={record.sample_id}; region={metadata.get('region')}; "
                    f"orientation={metadata.get('orientation')}; "
                    f"segmentation_issues={metadata.get('segmentation_issues') or []}"),
            })
            content.append(self._image_part(record))
        return {
            "model": self.settings.model,
            "stream": False,
            "thinking": {"type": "disabled"},
            "messages": [
                {"role": "system", "content": (
                    "You are a strict visual classifier. Follow the supplied closed "
                    "label schema and never fabricate sample ids.")},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max(512, len(records) * 32),
        }

    @staticmethod
    def parse(content: str, expected_ids: list[str]) -> list[dict]:
        def unique_object(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate JSON key")
                value[key] = item
            return value

        try:
            body = json.loads(content, object_pairs_hook=unique_object)
            if not isinstance(body, dict) or set(body) != {"items"}:
                raise ValueError("unexpected response schema")
            items = body["items"]
            if not isinstance(items, list) or len(items) != len(expected_ids):
                raise ValueError("wrong item count")
            by_id = {}
            for item in items:
                if (not isinstance(item, dict) or
                        set(item) != {"id", "label", "confidence"}):
                    raise ValueError("unexpected item schema")
                sample_id = item["id"]
                label = item["label"]
                confidence = item["confidence"]
                if (not isinstance(sample_id, str) or sample_id in by_id or
                        label not in CLASSIFICATION_LABELS or
                        type(confidence) not in (float, int) or
                        not math.isfinite(float(confidence)) or
                        not 0 <= float(confidence) <= 1):
                    raise ValueError("invalid classification item")
                by_id[sample_id] = {
                    "id": sample_id, "label": label,
                    "confidence": float(confidence)}
            if set(by_id) != set(expected_ids):
                raise ValueError("classification ids do not match request")
            return [by_id[sample_id] for sample_id in expected_ids]
        except (json.JSONDecodeError, TypeError, ValueError):
            raise DeepSeekClassificationError(
                "DeepSeek returned an invalid crop classification batch") from None

    async def classify(self, records: list[CropReviewRecord]) -> list[dict]:
        payload = self.payload(records)
        expected_ids = [record.sample_id for record in records]
        endpoint = self.settings.base_url.rstrip("/") + "/chat/completions"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {self.settings.api_key}"},
                    json=payload,
                    timeout=self.timeout)
                if response.status_code == 200:
                    body = response.json()
                    choice = body["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        raise DeepSeekClassificationError(
                            "DeepSeek classification did not finish normally")
                    return self.parse(choice["message"]["content"], expected_ids)
                if response.status_code not in {408, 429, 500, 502, 503, 504}:
                    raise DeepSeekClassificationError(
                        f"DeepSeek classification HTTP {response.status_code}")
                last_error = DeepSeekClassificationError(
                    f"DeepSeek classification HTTP {response.status_code}")
            except (httpx.HTTPError, KeyError, TypeError, ValueError,
                    json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                await asyncio.sleep(min(8.0, 2.0 ** attempt))
        raise DeepSeekClassificationError(
            f"DeepSeek classification failed after retries: "
            f"{type(last_error).__name__ if last_error else 'unknown'}")


async def classify_corpus(
        store: CorpusReviewStore, classifier: DeepSeekTileClassifier, *,
        batch_size: int = 32, concurrency: int = 8,
        progress: Callable[[ClassificationProgress], None] | None = None,
        maximum: int | None = None) -> dict:
    """Classify every missing crop, splitting invalid batches down to one image."""
    if batch_size < 1 or batch_size > 128:
        raise ValueError("batch_size must be between 1 and 128")
    if concurrency < 1 or concurrency > 64:
        raise ValueError("concurrency must be between 1 and 64")
    records = store.unclassified()
    if maximum is not None:
        if maximum < 1:
            raise ValueError("maximum must be positive")
        records = records[:maximum]
    total = len(records)
    started = time.monotonic()
    completed = 0
    failures: list[dict] = []
    lock = asyncio.Lock()
    if progress is not None:
        progress(ClassificationProgress(total, 0, 0, started))
    queue = asyncio.Queue()
    for start in range(0, total, batch_size):
        queue.put_nowait(records[start:start + batch_size])

    async def report(done: int = 0, failed: dict | None = None) -> None:
        nonlocal completed
        async with lock:
            completed += done
            if failed is not None:
                failures.append(failed)
            if progress is not None:
                progress(ClassificationProgress(
                    total, completed, len(failures), started))

    async def classify_resilient(batch: list[CropReviewRecord]) -> None:
        try:
            predictions = await classifier.classify(batch)
            store.save_predictions(
                predictions, model=classifier.settings.model,
                prompt_version=classifier.prompt_version)
            await report(done=len(batch))
        except DeepSeekClassificationError as exc:
            if len(batch) > 1:
                middle = len(batch) // 2
                await classify_resilient(batch[:middle])
                await classify_resilient(batch[middle:])
            else:
                await report(failed={
                    "sample_id": batch[0].sample_id,
                    "error": str(exc)})

    async def worker() -> None:
        while True:
            try:
                batch = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await classify_resilient(batch)
            finally:
                queue.task_done()

    try:
        await asyncio.gather(*(worker() for _ in range(min(concurrency, total or 1))))
    finally:
        await classifier.close()
    return {
        "corpus_id": store.corpus_id,
        "model": classifier.settings.model,
        "prompt_version": classifier.prompt_version,
        "requested": total,
        "classified": completed,
        "failed": len(failures),
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
    }


def run_classification(
        manifest: str | Path, dataset_root: str | Path, *,
        project_root: str | Path = ".", batch_size: int = 32,
        concurrency: int = 8, maximum: int | None = None,
        proxy: str | None = None,
        progress: Callable[[ClassificationProgress], None] | None = None) -> dict:
    root = Path(project_root)
    settings = ModelSettings.load(root)
    timeout = float(os.environ.get("DEEPSEEK_CLASSIFY_TIMEOUT_SECONDS", "120"))
    classifier = DeepSeekTileClassifier(
        settings, timeout=timeout, proxy=proxy)
    store = CorpusReviewStore(manifest, dataset_root)
    return asyncio.run(classify_corpus(
        store, classifier, batch_size=batch_size, concurrency=concurrency,
        progress=progress, maximum=maximum))
