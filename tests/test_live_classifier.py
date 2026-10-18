from pathlib import Path

import numpy as np
import pytest

from majsoul.domain import Box
from majsoul.live.classifier import CnnTileClassifier, build_tile_classifier
from majsoul.perception import TileTemplateClassifier, VisionConfig
from majsoul.perception.schemas import SegmentedTile, TilePrediction

torch = pytest.importorskip("torch")
models = pytest.importorskip("torchvision.models")

ROOT = Path(__file__).resolve().parents[1]

LABELS = tuple(
    [f"{rank}{suit}" for suit in "mps" for rank in "123456789"]
    + [f"0{suit}" for suit in "mps"]
    + [f"{rank}z" for rank in "1234567"]
)


@pytest.fixture(scope="module")
def vision_config():
    return VisionConfig.load(ROOT / "configs/vision.yaml")


def _crop(rng, height=64, width=48):
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _candidate(image, orientation="upright"):
    return SegmentedTile(Box(0, 0, image.shape[1], image.shape[0]), image,
                         orientation=orientation)


def _build_cnn(config, model):
    return CnnTileClassifier._from_model(model, LABELS, config, device="cpu")


@pytest.fixture(scope="module")
def cnn(vision_config):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, len(LABELS))
    return _build_cnn(vision_config, model)


class _FixedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self._logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, batch):
        return self._logits.unsqueeze(0).repeat(batch.shape[0], 1).to(batch.device)


def test_classify_returns_prediction(cnn):
    rng = np.random.default_rng(0)
    prediction = cnn.classify(_candidate(_crop(rng)))
    assert isinstance(prediction, TilePrediction)
    assert prediction.tile is None or prediction.tile in LABELS
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.orientation == "upright"
    assert len(prediction.candidates) == 4
    assert all(label in LABELS for label, _ in prediction.candidates)
    probs = [prob for _, prob in prediction.candidates]
    assert probs == sorted(probs, reverse=True)
    assert prediction.distance == pytest.approx(1.0 - probs[0])
    assert prediction.margin == pytest.approx(probs[0] - probs[1])


def test_classify_horizontal_tries_both_rotations(cnn):
    rng = np.random.default_rng(1)
    prediction = cnn.classify(_candidate(_crop(rng, height=48, width=64),
                                         orientation="horizontal"))
    assert prediction.orientation in ("horizontal:+90", "horizontal:-90")


def test_classify_many_matches_classify(cnn):
    rng = np.random.default_rng(2)
    candidates = (
        _candidate(_crop(rng)),
        _candidate(_crop(rng, height=48, width=64), orientation="horizontal"),
        _candidate(_crop(rng, height=80, width=56)),
    )
    batch = cnn.classify_many(candidates)
    singles = tuple(cnn.classify(candidate) for candidate in candidates)
    assert len(batch) == len(candidates)
    for from_batch, single in zip(batch, singles):
        assert from_batch.tile == single.tile
        assert from_batch.orientation == single.orientation
        assert from_batch.confidence == pytest.approx(single.confidence, abs=1e-5)
        assert from_batch.distance == pytest.approx(single.distance, abs=1e-5)
        assert from_batch.margin == pytest.approx(single.margin, abs=1e-5)
        assert [label for label, _ in from_batch.candidates] == \
               [label for label, _ in single.candidates]
        for (_, p1), (_, p2) in zip(from_batch.candidates, single.candidates):
            assert p1 == pytest.approx(p2, abs=1e-5)


def test_classify_many_empty(cnn):
    assert cnn.classify_many(()) == ()


def test_acceptance_gate_accepts_confident_prediction(vision_config):
    rng = np.random.default_rng(3)
    logits = [0.0] * len(LABELS)
    logits[0] = 10.0
    cnn = _build_cnn(vision_config, _FixedLogits(logits))
    prediction = cnn.classify(_candidate(_crop(rng)))
    assert prediction.tile == LABELS[0]
    assert prediction.confidence == pytest.approx(prediction.candidates[0][1])


def test_acceptance_gate_rejects_uncertain_prediction(vision_config):
    rng = np.random.default_rng(4)
    cnn = _build_cnn(vision_config, _FixedLogits([0.0] * len(LABELS)))
    prediction = cnn.classify(_candidate(_crop(rng)))
    assert prediction.tile is None
    assert prediction.confidence == 0.0
    assert prediction.distance == pytest.approx(1.0 - 1.0 / len(LABELS), abs=1e-4)


def test_build_template_backend(vision_config, tmp_path):
    classifier = build_tile_classifier("template", vision_config,
                                       template_root=tmp_path / "missing")
    assert isinstance(classifier, TileTemplateClassifier)
    assert classifier.backend_name == "template"


def test_build_auto_backend_prefers_cnn_when_torch_importable(
        vision_config, tmp_path, monkeypatch):
    class FakeCnn:
        def __init__(self, checkpoint, config):
            self.checkpoint = checkpoint

    monkeypatch.setattr("majsoul.live.classifier.CnnTileClassifier", FakeCnn)
    classifier = build_tile_classifier(
        "auto", vision_config, cnn_checkpoint="checkpoint.pt",
        template_root=tmp_path / "missing")
    assert isinstance(classifier, FakeCnn)
    assert classifier.backend_name == "cnn"


def test_build_auto_backend_falls_back_to_template_without_torch(
        vision_config, tmp_path, monkeypatch):
    def missing_torch():
        raise RuntimeError("no torch")

    monkeypatch.setattr("majsoul.live.classifier._import_torch", missing_torch)
    classifier = build_tile_classifier("auto", vision_config,
                                       template_root=tmp_path / "missing")
    assert isinstance(classifier, TileTemplateClassifier)
    assert classifier.backend_name == "template"


def test_build_cnn_requires_checkpoint(vision_config):
    with pytest.raises(ValueError):
        build_tile_classifier("cnn", vision_config)


def test_build_invalid_kind(vision_config):
    with pytest.raises(ValueError):
        build_tile_classifier("svm", vision_config)


@pytest.mark.parametrize('shape', [(37,25), (57,83), (224,224)])
def test_runtime_preprocessing_matches_gold_training_eval(cnn, monkeypatch, shape):
    from PIL import Image
    monkeypatch.syspath_prepend(str(ROOT/'tools'))
    from tile_augment_study import EvalTransform
    bgr = _crop(np.random.default_rng(23), *shape)
    expected = EvalTransform()(Image.fromarray(bgr[:, :, ::-1]))
    actual = cnn._preprocess(bgr)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_checkpoint_rejects_changed_label_order_before_building_model(tmp_path, vision_config, monkeypatch):
    path = tmp_path/'checkpoint.pt'
    path.write_bytes(b'fixture')
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {
        'model_name': 'convnext_tiny', 'labels': list(reversed(LABELS)), 'state_dict': {}})
    with pytest.raises(ValueError, match='ordered 37'):
        CnnTileClassifier(path, vision_config)


def test_checkpoint_rejects_wrong_experiment_before_building_model(tmp_path, vision_config, monkeypatch):
    path = tmp_path/'checkpoint.pt'
    path.write_bytes(b'fixture')
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {
        'model_name': 'convnext_tiny', 'labels': LABELS, 'state_dict': {}, 'experiment': 'A'})
    with pytest.raises(ValueError, match='experiment'):
        CnnTileClassifier(path, vision_config, expected_metadata={'experiment': 'D'})
