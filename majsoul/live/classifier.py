"""CNN tile classifier for the live pipeline.

Loads training checkpoints, including the October 2 augmented gold experiment
from tools/tile_augment_study.py. Each checkpoint is a dict
with ``model_name`` (a torchvision factory name), ``labels`` (37 tile labels
in the training order: m/p/s ranks 1-9, then 0m/0p/0s, then 1z-7z — see
``TILE_LABELS`` in tools/tile_model_select.py) and ``state_dict``.
Preprocessing matches the training eval transform: BGR->RGB, resize to
224x224 using PIL bilinear, ImageNet mean/std normalization. The Steam runtime
also verifies the configured model identity and content hash before loading.
"""

from __future__ import annotations

from pathlib import Path
from hashlib import sha256

import cv2
import numpy as np
from PIL import Image

from ..perception.config import VisionConfig
from ..perception.schemas import SegmentedTile, TilePrediction
from ..perception.tile_classifier import TileTemplateClassifier, _rotate

INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TOP_CANDIDATES = 4
TILE_LABELS = tuple([f"{rank}{suit}" for suit in "mps" for rank in range(1, 10)]
                    + [f"0{suit}" for suit in "mps"] + [f"{rank}z" for rank in range(1, 8)])


def checkpoint_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CnnTileClassifier requires torch; install a torch build to use the CNN backend"
        ) from exc
    return torch


def _import_models():
    try:
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError(
            "CnnTileClassifier requires torchvision to rebuild the checkpoint architecture"
        ) from exc
    return models


class CnnTileClassifier:
    """Classify segmented tile faces with the benchmarked CNN checkpoint."""

    def __init__(self, checkpoint: str | Path, config: VisionConfig, *, device: str | None = None,
                 expected_metadata: dict | None = None):
        torch = _import_torch()
        models = _import_models()
        path = Path(checkpoint)
        if not path.is_file():
            raise ValueError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        try:
            model_name = str(payload["model_name"])
            labels = tuple(str(label) for label in payload["labels"])
            state_dict = payload["state_dict"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed checkpoint: {path}") from exc
        if labels != TILE_LABELS:
            raise ValueError("Checkpoint must use the canonical ordered 37 tile labels")
        digest = checkpoint_hash(path)
        metadata = {"model_name": model_name, "experiment": payload.get("experiment"),
                    "seed": payload.get("seed"), "sha256": digest}
        for key, expected in (expected_metadata or {}).items():
            if key not in metadata or metadata[key] != expected:
                raise ValueError(f"Checkpoint metadata does not match configured expectation: {key}")
        try:
            factory = getattr(models, model_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported checkpoint architecture: {model_name}") from exc
        model = factory(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, len(labels))
        model.load_state_dict(state_dict)
        self._finish_init(model, labels, config, device)
        self.manifest = {**metadata, "checkpoint": str(path.resolve()), "labels": list(labels),
                         "device": str(self.device), "preprocessing": {
                             "version": "pil-bilinear-imagenet-v1", "color": "BGR->RGB",
                             "size": [INPUT_SIZE, INPUT_SIZE], "resize": "PIL bilinear",
                             "mean": IMAGENET_MEAN, "std": IMAGENET_STD}}

    @classmethod
    def _from_model(cls, model, labels, config: VisionConfig, *, device: str | None = None):
        """Inject an already-built model; used by tests to avoid real checkpoints."""
        _import_torch()
        instance = cls.__new__(cls)
        instance._finish_init(model, tuple(str(label) for label in labels), config, device)
        return instance

    def _finish_init(self, model, labels, config, device) -> None:
        torch = _import_torch()
        self._torch = torch
        self.config = config
        self.labels = tuple(labels)
        self.device = (torch.device(device) if device else
                       torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = model.to(self.device).eval()

    def _preprocess(self, image: np.ndarray):
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("Tile image must be a non-empty BGR image")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Match tools/tile_augment_study.py EvalTransform exactly. OpenCV cubic
        # produces different pixels from training's PIL bilinear resize.
        resized = np.array(Image.fromarray(rgb).resize(
            (INPUT_SIZE, INPUT_SIZE), resample=Image.Resampling.BILINEAR))
        tensor = self._torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        mean = self._torch.tensor(IMAGENET_MEAN, dtype=self._torch.float32).view(3, 1, 1)
        std = self._torch.tensor(IMAGENET_STD, dtype=self._torch.float32).view(3, 1, 1)
        return (tensor - mean) / std

    def classify(self, candidate: SegmentedTile) -> TilePrediction:
        return self.classify_many((candidate,))[0]

    def classify_many(self, candidates) -> tuple[TilePrediction, ...]:
        candidates = tuple(candidates)
        jobs = []
        for index, candidate in enumerate(candidates):
            rotations = (90, -90) if candidate.orientation == "horizontal" else (0,)
            for rotation in rotations:
                jobs.append((index, candidate, rotation))
        if not jobs:
            return ()
        torch = self._torch
        batch = torch.stack([self._preprocess(_rotate(candidate.image, rotation))
                             for _, candidate, rotation in jobs]).to(self.device)
        with torch.no_grad():
            logits = self._model(batch)
        probabilities = torch.softmax(logits.float(), dim=1).cpu()
        per_candidate: list[list[tuple[int, object]]] = [[] for _ in candidates]
        for (index, _, rotation), probs in zip(jobs, probabilities):
            per_candidate[index].append((rotation, probs))
        predictions = []
        for candidate, entries in zip(candidates, per_candidate):
            rotation, probs = max(entries, key=lambda item: float(item[1].max()))
            predictions.append(self._prediction(candidate, rotation, probs))
        return tuple(predictions)

    def _prediction(self, candidate: SegmentedTile, rotation: int, probs) -> TilePrediction:
        order = probs.argsort(descending=True)[:TOP_CANDIDATES]
        top = tuple((self.labels[int(index)], float(probs[index])) for index in order)
        p1 = top[0][1]
        p2 = top[1][1] if len(top) > 1 else 0.0
        distance = 1.0 - p1
        margin = p1 - p2
        accepted = (distance <= self.config.maximum_distance and
                    margin >= self.config.minimum_margin)
        orientation = (candidate.orientation if rotation == 0 else
                       f"{candidate.orientation}:{rotation:+d}")
        return TilePrediction(top[0][0] if accepted else None,
                              p1 if accepted else 0.0,
                              distance, margin, orientation, top)


def build_tile_classifier(kind: str, config: VisionConfig, *,
                          cnn_checkpoint: str | Path | None = None,
                          template_root: str | Path | None = None,
                          expected_metadata: dict | None = None):
    if kind not in ("cnn", "template", "auto"):
        raise ValueError(f"Unknown tile classifier kind: {kind}")
    if kind == "auto":
        try:
            # CPU convnext_tiny inference fits the turn budget; only a missing
            # torch installation falls back to the template backend.
            _import_torch()
            kind = "cnn"
        except RuntimeError:
            kind = "template"
    if kind == "cnn":
        if cnn_checkpoint is None:
            raise ValueError("The cnn backend requires cnn_checkpoint")
        classifier = CnnTileClassifier(cnn_checkpoint, config, **(
            {"expected_metadata": expected_metadata} if expected_metadata else {}))
    else:
        if template_root is None:
            raise ValueError("The template backend requires template_root")
        classifier = TileTemplateClassifier(template_root, config)
    classifier.backend_name = kind
    return classifier
