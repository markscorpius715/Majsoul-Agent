import json
from hashlib import sha256

from PIL import Image
import pytest

from majsoul.continual.corpus_review import CorpusReviewStore


@pytest.fixture
def store(tmp_path):
    crop = tmp_path / "crop.png"
    Image.new("RGB", (20, 30), "white").save(crop)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "corpus_id": "ai_review_test", "label_status": "unverified",
        "training_allowed": False, "samples": [{
            "id": "crop_test", "path": "crop.png", "player_count": 4,
            "mode": "yonma", "source": "test", "region": "self_hand",
            "crop_sha256": sha256(crop.read_bytes()).hexdigest()}],
    }), encoding="utf-8")
    result = CorpusReviewStore(manifest, tmp_path / "datasets")
    result.save_predictions([{"id": "crop_test", "label": "0p", "confidence": .9}],
                            model="original", prompt_version="v1")
    return result


def review(store, **kwargs):
    return store.ai_review("crop_test", run_id="test-run", reviewer="ai",
                           reason="Visual inspection", expected_label="0p", **kwargs)


def test_ai_correction_preserves_original_and_cannot_enter_training(store):
    raw_before = store.manifest_path.read_bytes()
    assert review(store, decision="correct", label="0s", confidence=.95) == "correct"
    record = store.get("crop_test")
    assert record.prediction_label == "0s"
    assert record.prediction_status == "awaiting_human_confirmation"
    assert record.status == "pending"
    assert record.label is None and record.dataset_path is None
    assert store.counts()["verified"] == 0
    assert not (store.dataset_root / "verified").exists()
    assert store.manifest_path.read_bytes() == raw_before
    with store._connect() as connection:
        rows = connection.execute("SELECT * FROM ai_review_audit").fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0]["previous_prediction"])["label"] == "0p"
    assert json.loads(rows[0]["new_prediction"])["label"] == "0s"
    assert review(store, decision="correct", label="0s", confidence=.95) == "already_recorded"


@pytest.mark.parametrize("decision", ["keep", "uncertain"])
def test_ai_observation_does_not_rewrite_prediction(store, decision):
    before = store.get("crop_test")
    assert review(store, decision=decision) == decision
    assert store.get("crop_test") == before


def test_human_review_since_snapshot_wins(store):
    store.label("crop_test", "5s", verified_by="human")
    before = store.get("crop_test")
    assert review(store, decision="correct", label="0s", confidence=.95) == "skipped_human_reviewed"
    assert store.get("crop_test") == before


def test_stale_ai_snapshot_cannot_overwrite_new_prediction(store):
    with store._connect() as connection:
        connection.execute("UPDATE predictions SET label='1s'")
    assert review(store, decision="correct", label="0s", confidence=.95) == "skipped_prediction_changed"
    assert store.get("crop_test").prediction_label == "1s"


@pytest.mark.parametrize("decision", ["keep", "uncertain", "correct"])
def test_ai_queue_requires_audit_then_human_confirmation(store, decision):
    assert store.scan(ai_reviewed_only=True) == []
    kwargs = dict(label="0s", confidence=.95) if decision == "correct" else {}
    review(store, decision=decision, **kwargs)
    assert len(store.scan(ai_reviewed_only=True)) == 1
    record = store.get("crop_test")
    store.label("crop_test", record.prediction_label, verified_by="human")
    assert store.scan(ai_reviewed_only=True) == []
    verified = store.get("crop_test")
    assert (store.dataset_root / verified.dataset_path).is_file()


def test_exports_follow_audit_and_are_idempotent(store, tmp_path):
    from dataclasses import asdict
    from majsoul.continual.ai_review_exports import sync_exports
    run = tmp_path / "test-run"
    run.mkdir()
    record = asdict(store.get("crop_test"))
    record["path"] = str(record["path"])
    (run / "snapshot.json").write_text(json.dumps([record]), encoding="utf-8")
    exports = tmp_path / "exports"
    assert sync_exports(store, run, exports)["not_ai_reviewed_count"] == 1
    review(store, decision="uncertain")
    first = sync_exports(store, run, exports)
    assert first["ai_reviewed_count"] == 1
    assert first["not_ai_reviewed_count"] == 0
    assert (exports / "ai-reviewed" / "uncertain" / "crop.png").exists()
    assert sync_exports(store, run, exports) == first
    assert store.counts()["verified"] == 0
