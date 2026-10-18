import asyncio
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import cv2
import httpx
import numpy as np
import pytest

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.continual import (Candidate, CapturedTileCorpus, ContinualConfig,
                               CorpusReviewStore, CropReviewRecord,
                               KnownVisualIndex, PersistentUnknownDetector,
                               PerceptionLoopState, ReviewCaseIndex,
                               ReviewCaseStore, ReviewKind,
                               TileCropExtractor,
                               UnknownEvidence, UnknownReason, assess_unknown,
                               create_crop_corpus_sheet, create_review_sheet,
                               format_review_message, notify_review_case,
                               parse_label_command, parse_review_command,
                               preset_labels, register_captured_video,
                               tile_label_text)
from majsoul.continual.deepseek_classifier import (
    DeepSeekClassificationError, DeepSeekTileClassifier)
from majsoul.perception import VisionConfig
from majsoul.strategy import ModelSettings


ROOT = Path(__file__).resolve().parents[1]


def image(seed=1):
    return np.random.default_rng(seed).integers(0, 256, (80, 120, 3), dtype=np.uint8)


def evidence(crop, *, frame=1, kind=ReviewKind.TILE, region="river_left"):
    return UnknownEvidence(
        kind=kind,
        region=region,
        timestamp=frame / 30,
        frame_index=frame,
        reasons=(UnknownReason.MODEL_UNCERTAINTY,),
        full_frame=image(20),
        crop=crop,
        previous_frame=image(19),
        next_frame=image(21),
        candidates=(Candidate("5p", 0.31), Candidate("0p", 0.27)),
        model_version="tile_v0",
        state={"round_id": "1z:2:0"},
        metadata={"slot": "left", "index": 0,
                  "player_count": 3, "mode": "sanma"},
    )


def test_unknown_assessment_combines_model_state_and_novelty():
    config = ContinualConfig()
    confident_but_inconsistent = assess_unknown(
        config, confidence=0.99, margin=0.50,
        state_issues=("fifth visible 7p",), novelty_distance=0.05)
    confident_but_novel = assess_unknown(
        config, confidence=0.99, margin=0.50,
        novelty_distance=0.80)
    known = assess_unknown(config, confidence=0.95, margin=0.20,
                           novelty_distance=0.05)

    assert confident_but_inconsistent.reasons == (
        UnknownReason.STATE_INCONSISTENCY,)
    assert confident_but_novel.reasons == (
        UnknownReason.NOVEL_VISUAL_PATTERN,)
    assert not known.unknown


def test_persistent_unknown_requires_four_frames_and_deduplicates():
    detector = PersistentUnknownDetector(ContinualConfig(persistent_frames=4))
    crop = image(7)
    results = [detector.observe(evidence(crop, frame=index)) for index in range(1, 6)]

    assert [result.persistent for result in results] == [False, False, False, True, True]
    assert [result.new_case for result in results] == [False, False, False, True, False]
    assert results[3].safe_pause and results[4].safe_pause
    assert results[0].control_state == PerceptionLoopState.RESYNC
    assert results[3].control_state == PerceptionLoopState.HUMAN_REVIEW


def test_persistent_unknown_streak_requires_consecutive_frames():
    detector = PersistentUnknownDetector(ContinualConfig(persistent_frames=3))
    crop = image(7)
    key = (ReviewKind.TILE.value, "river_left")

    detector.observe(evidence(crop, frame=1))
    detector.end_frame({key})
    detector.observe(evidence(crop, frame=2))
    detector.end_frame(set())
    restarted = detector.observe(evidence(crop, frame=3))

    assert restarted.streak == 1
    assert not restarted.persistent


def test_visual_novelty_index_uses_verified_reference_distance():
    known = image(2)
    index = KnownVisualIndex()
    assert index.nearest_distance(known) is None
    index.add(known)

    assert index.nearest_distance(known) == pytest.approx(0.0, abs=1e-5)
    assert index.nearest_distance(image(3)) > 0


def test_review_package_and_human_label_enter_queue_without_training(tmp_path):
    cases = tmp_path / "unknown_cases"
    datasets = tmp_path / "datasets"
    store = ReviewCaseStore(cases, datasets)
    created = store.create(evidence(image(8), frame=127))
    case = cases / created.case_id

    assert created.case_id == "case_000001"
    assert created.created
    assert "/label case_000001 <label>" in created.message
    assert {path.name for path in case.iterdir()} == {
        "full_frame.png", "crop.png", "previous_frame.png", "next_frame.png",
        "state.json", "metadata.json"}
    pending = json.loads((case / "metadata.json").read_text(encoding="utf-8"))
    assert pending["status"] == "pending"
    assert pending["crop_sha256"]

    verified = store.label(created.case_id, "0p", verified_by="human")
    assert verified["status"] == "verified"
    assert (datasets / verified["dataset_path"]).is_file()
    manifest = json.loads(
        (datasets / "manifests/verified_samples.json").read_text(encoding="utf-8"))
    queue = json.loads(
        (datasets / "manifests/training_queue.json").read_text(encoding="utf-8"))
    assert manifest[0]["label"] == "0p"
    assert manifest[0]["player_count"] == 3
    assert manifest[0]["mode"] == "sanma"
    assert manifest[0]["split"] == "unassigned"
    assert queue == [{
        "case_id": "case_000001",
        "status": "pending",
        "label": "0p",
        "kind": "tile",
        "player_count": 3,
        "mode": "sanma",
        "dataset_path": "verified/tiles/0p/case_000001.png",
    }]
    assert not (datasets / "models").exists()
    with pytest.raises(ValueError, match="pending"):
        store.label(created.case_id, "0p", verified_by="human")


def test_reviewer_suggestion_remains_pending_and_outside_dataset(tmp_path):
    cases = tmp_path / "cases"
    datasets = tmp_path / "datasets"
    store = ReviewCaseStore(cases, datasets)
    created = store.create(evidence(image(9), frame=128))

    suggested = store.suggest(
        created.case_id, ["4z", "3z", "4z"],
        suggested_by="codex_visual_review", note="glyph resembles north")

    assert suggested["status"] == "pending"
    latest = suggested["reviewer_suggestions"][-1]
    assert latest["labels"] == ["4z", "3z"]
    assert latest["status"] == "awaiting_human_confirmation"
    assert "AI visual review candidate" in format_review_message(suggested)
    assert "1. 4z" in format_review_message(suggested)
    assert not (datasets / "verified").exists()
    assert not (datasets / "manifests" / "training_queue.json").exists()

    verified = store.label(created.case_id, "4z", verified_by="human")
    assert verified["reviewer_suggestions"][-1]["status"] == "confirmed_by_human"
    manifest = json.loads(
        (datasets / "manifests/verified_samples.json").read_text(encoding="utf-8"))
    assert manifest[0]["reviewer_suggestions"][-1]["human_label"] == "4z"


def test_review_labeler_index_filters_cases_and_exposes_complete_tile_schema(tmp_path):
    cases = tmp_path / "cases"
    datasets = tmp_path / "datasets"
    store = ReviewCaseStore(cases, datasets)
    tile = store.create(evidence(image(61), frame=1))
    action = store.create(evidence(
        image(62), frame=2, kind=ReviewKind.ACTION, region="action_buttons"))
    index = ReviewCaseIndex(cases)

    assert [record.case_id for record in index.scan(kind="tile")] == [tile.case_id]
    assert [record.case_id for record in index.scan(kind="action")] == [action.case_id]
    store.label(tile.case_id, "4z", verified_by="human_gui")
    assert [record.case_id for record in index.scan(
        status="verified", kind="tile")] == [tile.case_id]
    assert [record.case_id for record in index.scan(
        status="pending", kind="action")] == [action.case_id]

    labels = [label for label, _display in preset_labels(ReviewKind.TILE)]
    assert len(labels) == len(set(labels)) == 37
    assert {"0m", "0p", "0s", "1z", "7z"}.issubset(labels)
    assert "0z" not in labels
    assert tile_label_text("4z") == "4z North"
    assert tile_label_text("0p") == "0p red five pin"


def test_corpus_review_store_labels_and_rejects_without_mutating_raw_manifest(tmp_path):
    corpus = tmp_path / "raw" / "tile_crops"
    crop_dir = corpus / "crops" / "river_left"
    crop_dir.mkdir(parents=True)
    first_path = crop_dir / "crop_aaa111.png"
    second_path = crop_dir / "crop_bbb222.png"
    cv2.imwrite(str(first_path), image(71))
    cv2.imwrite(str(second_path), image(72))
    samples = []
    for sample_id, path, issues in (
            ("crop_aaa111", first_path, []),
            ("crop_bbb222", second_path, ["ambiguous rows"])):
        samples.append({
            "id": sample_id,
            "source": "captured_test",
            "player_count": 4,
            "mode": "yonma",
            "region": "river_left",
            "position": "r0c0",
            "orientation": "upright",
            "first_frame": 15,
            "last_frame": 30,
            "first_timestamp": 0.5,
            "last_timestamp": 1.0,
            "occurrence_count": 2,
            "segmentation_issues": issues,
            "label": None,
            "training_allowed": False,
            "crop_sha256": sha256(path.read_bytes()).hexdigest(),
            "path": path.relative_to(corpus).as_posix(),
        })
    manifest_path = corpus / "manifest.json"
    manifest = {
        "schema_version": 1,
        "corpus_id": "captured_test_tile_crops",
        "label_status": "unverified",
        "training_allowed": False,
        "samples": samples,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    original = manifest_path.read_bytes()

    store = CorpusReviewStore(manifest_path, tmp_path / "datasets")
    assert store.counts()["pending"] == 2
    assert [item.sample_id for item in store.scan(issue_filter="issues")] == [
        "crop_bbb222"]

    store.save_predictions([
        {"id": "crop_aaa111", "label": "4z", "confidence": 0.98},
        {"id": "crop_bbb222", "label": "reject", "confidence": 0.88},
    ], model="deepseek-flash", prompt_version="test-v1")
    assert store.prediction_counts()["classified"] == 2
    assert [item.sample_id for item in store.scan(
        classification="classified", predicted_label="4z")] == ["crop_aaa111"]
    assert store.get("crop_aaa111").prediction_label == "4z"

    verified = store.label("crop_aaa111", "3z", verified_by="human")
    rejected = store.reject(
        "crop_bbb222", rejected_by="human", reason="not a tile face")

    assert verified.status == "verified"
    assert verified.label == "3z"
    assert verified.prediction_status == "corrected_by_human"
    assert verified.prediction_human_label == "3z"
    assert verified.dataset_path == "verified/tiles/3z/crop_aaa111.png"
    assert (tmp_path / "datasets" / verified.dataset_path).is_file()

    old_verified_path = tmp_path / "datasets" / verified.dataset_path
    verified = store.correct_verified(
        "crop_aaa111", "2z", corrected_by="human_correction")
    assert verified.status == "verified"
    assert verified.label == "2z"
    assert verified.reviewed_by == "human_correction"
    assert verified.dataset_path == "verified/tiles/2z/crop_aaa111.png"
    assert not old_verified_path.exists()
    assert (tmp_path / "datasets" / verified.dataset_path).is_file()
    assert store.verified_label_counts() == Counter({"2z": 1})
    history = store.correction_history("crop_aaa111")
    assert [(item["previous_label"], item["new_label"])
            for item in history] == [("3z", "2z")]
    with pytest.raises(ValueError, match="must differ"):
        store.correct_verified(
            "crop_aaa111", "2z", corrected_by="human_correction")

    assert rejected.status == "rejected"
    assert rejected.rejection_reason == "not a tile face"
    assert store.counts() == Counter({"verified": 1, "rejected": 1, "pending": 0})

    verified_path = tmp_path / "datasets" / verified.dataset_path
    invalidated = store.correct_verified(
        "crop_aaa111", "reject", corrected_by="human_correction")
    assert invalidated.status == "rejected"
    assert invalidated.label is None
    assert invalidated.rejection_reason == "verified_label_invalidated_by_human"
    assert not verified_path.exists()
    assert (tmp_path / "datasets/rejected/tiles/crop_aaa111.png").is_file()
    assert store.verified_label_counts() == Counter()
    assert store.correction_history("crop_aaa111")[-1]["new_status"] == "rejected"
    assert store.counts() == Counter({"rejected": 2, "pending": 0})
    assert manifest_path.read_bytes() == original
    with pytest.raises(ValueError, match="pending"):
        store.label("crop_aaa111", "3z", verified_by="human")


def test_corpus_review_recovers_mistakenly_rejected_crop(tmp_path):
    corpus = tmp_path / "raw" / "tile_crops"
    crop_dir = corpus / "crops" / "hand_bottom"
    crop_dir.mkdir(parents=True)
    crop_path = crop_dir / "crop_ccc333.png"
    cv2.imwrite(str(crop_path), image(73))
    manifest_path = corpus / "manifest.json"
    manifest = {
        "schema_version": 1,
        "corpus_id": "captured_test_tile_crops",
        "label_status": "unverified",
        "training_allowed": False,
        "samples": [{
            "id": "crop_ccc333",
            "source": "captured_test",
            "player_count": 4,
            "mode": "yonma",
            "region": "hand_bottom",
            "position": "r0c0",
            "orientation": "upright",
            "first_frame": 15,
            "last_frame": 30,
            "first_timestamp": 0.5,
            "last_timestamp": 1.0,
            "occurrence_count": 2,
            "segmentation_issues": [],
            "label": None,
            "training_allowed": False,
            "crop_sha256": sha256(crop_path.read_bytes()).hexdigest(),
            "path": crop_path.relative_to(corpus).as_posix(),
        }],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    store = CorpusReviewStore(manifest_path, tmp_path / "datasets")
    store.save_predictions([
        {"id": "crop_ccc333", "label": "reject", "confidence": 0.9},
    ], model="deepseek-flash", prompt_version="test-v1")
    rejected = store.reject(
        "crop_ccc333", rejected_by="human", reason="misread as not a tile")
    assert rejected.status == "rejected"

    with pytest.raises(ValueError, match="differ"):
        store.correct_rejected(
            "crop_ccc333", "reject", corrected_by="human_correction")

    recovered = store.correct_rejected(
        "crop_ccc333", "5p", corrected_by="human_correction")
    assert recovered.status == "verified"
    assert recovered.label == "5p"
    assert recovered.rejection_reason is None
    assert recovered.reviewed_by == "human_correction"
    assert recovered.dataset_path == "verified/tiles/5p/crop_ccc333.png"
    assert (tmp_path / "datasets" / recovered.dataset_path).is_file()
    assert recovered.prediction_status == "corrected_by_human"
    assert recovered.prediction_human_label == "5p"
    assert store.counts() == Counter({"verified": 1, "pending": 0})
    assert store.verified_label_counts() == Counter({"5p": 1})
    history = store.correction_history("crop_ccc333")
    assert [(item["previous_status"], item["previous_label"],
             item["new_status"], item["new_label"])
            for item in history] == [("rejected", None, "verified", "5p")]

    with pytest.raises(ValueError, match="rejected"):
        store.correct_rejected(
            "crop_ccc333", "7s", corrected_by="human_correction")


def test_corpus_review_balances_pending_samples_by_verified_position_coverage(
        tmp_path):
    corpus = tmp_path / "raw" / "tile_crops"
    crop_dir = corpus / "crops" / "river_left"
    crop_dir.mkdir(parents=True)
    positions = ("r0c0", "r0c0", "r0c1", "r0c1", "r0c2")
    occurrence_counts = (1, 100, 20, 20, 50)
    samples = []
    predictions = []
    for index, (position, occurrence_count) in enumerate(zip(
            positions, occurrence_counts, strict=True)):
        sample_id = f"crop_balance{index}"
        path = crop_dir / f"{sample_id}.png"
        cv2.imwrite(str(path), image(80 + index))
        samples.append({
            "id": sample_id,
            "source": "captured_balance_test",
            "player_count": 4,
            "mode": "yonma",
            "region": "river_left",
            "position": position,
            "orientation": "upright",
            "first_frame": index,
            "last_frame": index,
            "first_timestamp": float(index),
            "last_timestamp": float(index),
            "occurrence_count": occurrence_count,
            "segmentation_issues": [],
            "label": None,
            "training_allowed": False,
            "crop_sha256": sha256(path.read_bytes()).hexdigest(),
            "path": path.relative_to(corpus).as_posix(),
        })
        predictions.append({
            "id": sample_id, "label": "4z", "confidence": 0.9})
    manifest_path = corpus / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "corpus_id": "captured_balance_tile_crops",
        "label_status": "unverified",
        "training_allowed": False,
        "samples": samples,
    }), encoding="utf-8")

    store = CorpusReviewStore(manifest_path, tmp_path / "datasets")
    store.save_predictions(
        predictions, model="deepseek-flash", prompt_version="test-v1")
    store.label("crop_balance0", "4z", verified_by="human")

    ordered = store.scan(
        status="pending", classification="classified",
        order="balanced_positions")
    ordered_positions = [item.metadata["position"] for item in ordered]
    assert set(ordered_positions[:2]) == {"r0c1", "r0c2"}
    assert ordered_positions.index("r0c0") >= 2

    frequency_ordered = store.scan(
        status="pending", classification="classified",
        order="frequency_balanced_positions")
    assert [item.metadata["occurrence_count"] for item in frequency_ordered] == [
        100, 50, 20, 20]
    assert frequency_ordered[0].metadata["position"] == "r0c0"


def test_corpus_review_orders_verified_by_latest_review_or_correction(tmp_path):
    corpus = tmp_path / "raw" / "tile_crops"
    crop_dir = corpus / "crops" / "river_left"
    crop_dir.mkdir(parents=True)
    samples = []
    for index, sample_id in enumerate(("crop_old111", "crop_new222")):
        path = crop_dir / f"{sample_id}.png"
        cv2.imwrite(str(path), image(90 + index))
        samples.append({
            "id": sample_id,
            "source": "captured_review_order_test",
            "player_count": 4,
            "mode": "yonma",
            "region": "river_left",
            "position": f"r0c{index}",
            "orientation": "upright",
            "first_frame": index,
            "last_frame": index,
            "first_timestamp": float(index),
            "last_timestamp": float(index),
            "occurrence_count": 1,
            "segmentation_issues": [],
            "label": None,
            "training_allowed": False,
            "crop_sha256": sha256(path.read_bytes()).hexdigest(),
            "path": path.relative_to(corpus).as_posix(),
        })
    manifest_path = corpus / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "corpus_id": "captured_review_order_tile_crops",
        "label_status": "unverified",
        "training_allowed": False,
        "samples": samples,
    }), encoding="utf-8")

    store = CorpusReviewStore(manifest_path, tmp_path / "datasets")
    store.label("crop_old111", "1m", verified_by="human")
    store.label("crop_new222", "2m", verified_by="human")
    with store._connect() as connection:
        connection.execute(
            "UPDATE reviews SET reviewed_at = ? WHERE sample_id = ?",
            ("2026-09-25T01:00:00+00:00", "crop_old111"))
        connection.execute(
            "UPDATE reviews SET reviewed_at = ? WHERE sample_id = ?",
            ("2026-09-25T02:00:00+00:00", "crop_new222"))

    assert [record.sample_id for record in store.scan(
            status="verified", order="reviewed_at_desc")] == [
                "crop_new222", "crop_old111"]
    with pytest.raises(ValueError, match="reviewed samples"):
        store.scan(status="pending", order="reviewed_at_desc")

    store.correct_verified("crop_old111", "3m", corrected_by="human")
    assert [record.sample_id for record in store.scan(
            status="verified", order="reviewed_at_desc")] == [
                "crop_old111", "crop_new222"]


def test_deepseek_tile_classifier_sends_images_and_requires_exact_closed_schema(
        tmp_path):
    crop = tmp_path / "crop_abc123.png"
    cv2.imwrite(str(crop), image(73))
    record = CropReviewRecord(
        "crop_abc123", crop,
        {"region": "river_left", "orientation": "upright",
         "segmentation_issues": []})

    def respond(request):
        body = json.loads(request.content)
        assert body["model"] == "deepseek-flash"
        parts = body["messages"][1]["content"]
        image_part = next(item for item in parts if item["type"] == "image_url")
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "stop", "message": {"content":
                json.dumps({"items": [{
                    "id": "crop_abc123", "label": "4z", "confidence": 0.97}]})}}]})

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        classifier = DeepSeekTileClassifier(
            ModelSettings("test-only"), client, max_retries=0)
        result = await classifier.classify([record])
        await client.aclose()
        return result

    assert asyncio.run(run()) == [{
        "id": "crop_abc123", "label": "4z", "confidence": 0.97}]
    with pytest.raises(DeepSeekClassificationError):
        DeepSeekTileClassifier.parse(
            '{"items":[{"id":"crop_abc123","label":"unknown",'
            '"confidence":0.9}]}', ["crop_abc123"])


def test_review_store_deduplicates_across_repeated_process_runs(tmp_path):
    first_store = ReviewCaseStore(tmp_path / "cases", tmp_path / "datasets")
    first = first_store.create(evidence(image(15), frame=10))
    second_store = ReviewCaseStore(tmp_path / "cases", tmp_path / "datasets")
    duplicate = second_store.create(evidence(image(15), frame=20))

    assert first.created
    assert not duplicate.created
    assert duplicate.case_id == first.case_id
    assert len(list((tmp_path / "cases").glob("case_*"))) == 1


def test_review_store_deduplicates_same_visual_across_region_indexes(tmp_path):
    store = ReviewCaseStore(tmp_path / "cases", tmp_path / "datasets")
    first = store.create(evidence(
        image(51), frame=10, region="river_left[0]"))
    duplicate = store.create(evidence(
        image(51), frame=20, region="river_left[7]"))

    assert first.created
    assert not duplicate.created
    assert duplicate.case_id == first.case_id


def test_verified_duplicate_does_not_request_another_label(tmp_path):
    cases = tmp_path / "cases"
    datasets = tmp_path / "datasets"
    store = ReviewCaseStore(cases, datasets)
    first = store.create(evidence(image(18), frame=10))
    store.label(first.case_id, "2z", verified_by="human")

    duplicate = ReviewCaseStore(cases, datasets).create(
        evidence(image(18), frame=20))

    assert not duplicate.created
    assert duplicate.status.value == "verified"
    assert "Verified label: 2z" in duplicate.message
    assert "/label" not in duplicate.message


def test_review_store_migrates_legacy_case_hash_during_dedup(tmp_path):
    store = ReviewCaseStore(tmp_path / "cases", tmp_path / "datasets")
    first = store.create(evidence(image(16), frame=10))
    metadata_path = Path(first.path) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("visual_hash")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    duplicate = ReviewCaseStore(
        tmp_path / "cases", tmp_path / "datasets").create(
            evidence(image(16), frame=20))
    migrated = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert not duplicate.created
    assert migrated["visual_hash"]


def test_legacy_case_context_must_be_explicitly_backfilled(tmp_path):
    cases = tmp_path / "cases"
    datasets = tmp_path / "datasets"
    store = ReviewCaseStore(cases, datasets)
    created = store.create(evidence(image(17), frame=10))
    metadata_path = Path(created.path) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["player_count"] = None
    metadata["mode"] = None
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="lacks player_count/mode"):
        store.label(created.case_id, "2z", verified_by="human")
    result = store.backfill_source_context(
        "runtime_unknown", player_count=3, mode="sanma")
    assert result["updated_case_ids"] == [created.case_id]
    assert store.label(created.case_id, "2z", verified_by="human")["mode"] == "sanma"


def test_action_labels_are_separate_from_game_events(tmp_path):
    store = ReviewCaseStore(tmp_path / "cases", tmp_path / "datasets")
    created = store.create(evidence(
        image(11), kind=ReviewKind.ACTION, region="action_buttons"))
    verified = store.label(created.case_id, "pass", verified_by="human")

    assert verified["label"] == "PASS"
    assert "verified/actions/PASS" in verified["dataset_path"]


def test_reject_unusable_crop_never_enters_training_data(tmp_path):
    cases = tmp_path / "cases"
    datasets = tmp_path / "datasets"
    store = ReviewCaseStore(cases, datasets)
    created = store.create(evidence(image(12), region="river_right[0]"))

    rejected = store.reject(
        created.case_id, rejected_by="human", reason="tile back, not a face")

    assert rejected["status"] == "rejected"
    assert rejected["rejection_reason"] == "tile back, not a face"
    assert not (datasets / "verified").exists()
    duplicate = ReviewCaseStore(cases, datasets).create(
        evidence(image(12), frame=3, region="river_right[0]"))
    assert not duplicate.created
    assert duplicate.status.value == "rejected"
    assert "/label" not in duplicate.message


def test_review_sheet_labels_cases_without_changing_status(tmp_path):
    cases = tmp_path / "cases"
    store = ReviewCaseStore(cases, tmp_path / "datasets")
    first = store.create(evidence(image(31), frame=1, region="self_hand[0]"))
    second = store.create(evidence(image(32), frame=2, region="river_right[0]"))

    output = tmp_path / "sheet.png"
    report = create_review_sheet(
        cases, [first.case_id, second.case_id], output, columns=2)

    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    assert [entry["case_id"] for entry in report["entries"]] == [
        first.case_id, second.case_id]
    assert all(entry["status"] == "pending" for entry in report["entries"])


def test_continual_config_is_versioned_and_validated():
    config = ContinualConfig.load(ROOT / "configs/continual.json")
    assert config.persistent_frames == 4
    assert config.training_batch_size == 50
    assert config.crop_minimum_frame_tiles == 5
    with pytest.raises(ValueError, match="between 3 and 5"):
        ContinualConfig(persistent_frames=2)


def test_captured_tile_corpus_is_unverified_deduplicated_and_resumable(tmp_path):
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(json.dumps({
        "source_id": "captured_test",
        "source_type": "captured_video",
        "domain": "mahjong_soul",
        "video": {"player_count": 3, "mode": "sanma", "frame_count": 2},
    }), encoding="utf-8")
    roi = ROIConfig.load(ROOT / "configs/roi.yaml")
    vision = VisionConfig.load(ROOT / "configs/vision.yaml")
    frame = read_image(ROOT / "artifacts/roi/review/frame_1590.png")
    batch = TileCropExtractor(
        roi, vision, ("bottom", "top", "left")).extract(frame)
    assert len(batch.crops) > 20

    config = ContinualConfig(
        crop_minimum_frame_tiles=1, crop_checkpoint_frames=1)
    output = tmp_path / "corpus"
    corpus = CapturedTileCorpus(
        output, source_manifest, roi=roi,
        active_slots=("bottom", "top", "left"), config=config,
        start=0, end=2, stride=1)
    corpus.add_frame(batch, frame_index=0, timestamp=0.0)
    unique_after_first = corpus.manifest["progress"]["unique_crops"]
    corpus.add_frame(batch, frame_index=1, timestamp=0.1)
    corpus.checkpoint(complete=True)

    assert unique_after_first > 0
    assert corpus.manifest["progress"]["duplicate_crops"] >= unique_after_first
    assert corpus.manifest["training_allowed"] is False
    assert all(sample["label"] is None and not sample["training_allowed"]
               for sample in corpus.manifest["samples"])
    assert all((output / sample["path"]).is_file()
               for sample in corpus.manifest["samples"])

    resumed = CapturedTileCorpus(
        output, source_manifest, roi=roi,
        active_slots=("bottom", "top", "left"), config=config,
        start=0, end=2, stride=1, resume=True)
    assert resumed.complete
    assert resumed.next_frame == 2
    sheet = create_crop_corpus_sheet(
        output / "manifest.json", tmp_path / "crop-sheet.png",
        max_per_region=2)
    assert Path(sheet["output"]).is_file()
    assert sheet["entries"]


def test_feishu_review_notification_is_audited_and_idempotent(tmp_path):
    cases = tmp_path / "cases"
    store = ReviewCaseStore(cases, tmp_path / "datasets")
    created = store.create(evidence(image(44), frame=127))
    helper = tmp_path / "feishu_send.mjs"
    helper.write_text("// test helper", encoding="utf-8")
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        stdout = (json.dumps({
            "credentialsReady": True, "targetReady": True, "ready": True,
        }) if "check-config" in command else json.dumps({"ok": True}))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    delivery = notify_review_case(
        cases, created.case_id, helper=helper, chat_id="oc_test", runner=runner)
    assert delivery["status"] == "sent"
    assert len(calls) == 5
    assert [command[2] for command in calls] == [
        "check-config", "text", "image", "image", "file"]
    assert (Path(created.path) / "review-package.zip").is_file()
    metadata = json.loads(
        (Path(created.path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["notifications"][0]["status"] == "sent"
    assert "oc_test" not in json.dumps(metadata)

    repeated = notify_review_case(
        cases, created.case_id, helper=helper, chat_id="oc_test", runner=runner)
    assert repeated["already_sent"] is True
    assert len(calls) == 5

    store.suggest(
        created.case_id, ["4z"], suggested_by="codex_visual_review")
    updated = notify_review_case(
        cases, created.case_id, helper=helper, chat_id="oc_test", runner=runner)
    assert updated["already_sent"] is False
    assert updated["artifacts"] == []
    assert [command[2] for command in calls[-2:]] == ["check-config", "text"]


def test_register_captured_video_is_review_only_and_hashed(tmp_path):
    import cv2

    video = tmp_path / "captured.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for value in (20, 40, 60):
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()

    target = tmp_path / "source.json"
    manifest = register_captured_video(
        video, source_id="captured_yonma", manifest_path=target,
        player_count=4, mode="yonma")

    assert manifest["video"]["player_count"] == 4
    assert manifest["sha256"]
    assert manifest["admission"]["training_allowed"] is False
    assert "kita" in manifest["admission"]["excluded_scope"]
    assert json.loads(target.read_text(encoding="utf-8"))["source_id"] == "captured_yonma"


@pytest.mark.parametrize(("text", "expected"), [
    ("/label 127 0p", ("case_000127", "0p")),
    ("label case_000127 TSUMO", ("case_000127", "TSUMO")),
])
def test_explicit_label_command_parser(text, expected):
    assert parse_label_command(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("/label 127 0p", ("label", "case_000127", "0p")),
    ("/reject case_000005 possible tile back", (
        "reject", "case_000005", "possible tile back")),
    ("reject 5 white dragon context incomplete", (
        "reject", "case_000005", "white dragon context incomplete")),
])
def test_explicit_review_command_parser(text, expected):
    assert parse_review_command(text) == expected


@pytest.mark.parametrize("text", [
    "please label 127 0p",
    "/label 0 0p",
    "/label 127 ../../secret",
    "/label 127 0p and continue",
])
def test_label_command_rejects_ambiguous_or_unsafe_chat(text):
    with pytest.raises(ValueError):
        parse_label_command(text)


@pytest.mark.parametrize("text", [
    "please reject 5 because blank",
    "/reject 0 no",
    "/reject 5",
    "/label 5 0p and continue",
])
def test_review_command_rejects_ambiguous_chat(text):
    with pytest.raises(ValueError):
        parse_review_command(text)
