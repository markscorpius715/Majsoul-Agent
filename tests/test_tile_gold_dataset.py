"""Synthetic-data test for tools/tile_gold_dataset.py (does not touch the real dataset)."""
import io
import json
import sqlite3
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from tools.tile_gold_dataset import main

BLOCK_SECONDS = 255.0
LABELS = ("1m", "4m", "0m", "2z")
REGIONS = ("river_bottom", "self_hand", "river_top")


def make_png_bytes(seed_byte, size=(16, 24)):
    image = Image.new("RGB", size, (seed_byte % 256, (seed_byte * 7) % 256,
                                    (seed_byte * 13) % 256))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def far_hash(index):
    """64-bit hash where adjacent indices have Hamming distance >= 8 (to avoid accidental near-duplicates)."""
    return f"{int.from_bytes(bytes([index % 256]) * 8, 'big'):016x}"


def make_dataset(root, *, blocks=6, per_block=4, with_case=True,
                 hash_overrides=None):
    """Build a synthetic gold-label dataset, return a path dict. hash_overrides: {sample_id: hexhash}."""
    datasets = root / "datasets"
    db_dir = datasets / "manifests" / "corpus_reviews"
    db_dir.mkdir(parents=True)
    raw_dir = datasets / "raw" / "captured" / "video_a" / "tile_crops"
    raw_dir.mkdir(parents=True)

    db_path = db_dir / "reviews.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE reviews (sample_id TEXT PRIMARY KEY, status TEXT,"
        " label TEXT, reviewed_by TEXT, reviewed_at TEXT, rejection_reason TEXT,"
        " dataset_path TEXT, crop_sha256 TEXT, source TEXT, player_count INTEGER,"
        " mode TEXT, region TEXT, training_status TEXT)")
    manifest_samples = []
    hash_overrides = hash_overrides or {}
    index = 0
    for block in range(blocks):
        for slot in range(per_block):
            index += 1
            sample_id = f"crop_{index:04d}"
            label = LABELS[slot % len(LABELS)]
            region = REGIONS[slot % len(REGIONS)]
            payload = make_png_bytes(index)
            rel_path = f"verified/tiles/{label}/{sample_id}.png"
            target = datasets / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            timestamp = block * BLOCK_SECONDS + 10.0 + slot
            manifest_samples.append({
                "id": sample_id, "region": region,
                "orientation": "horizontal" if slot % 5 == 4 else "upright",
                "first_timestamp": timestamp, "last_timestamp": timestamp + 0.5,
                "visual_hash": hash_overrides.get(sample_id, far_hash(index)),
                "path": f"crops/{region}/{sample_id}.png",
            })
            conn.execute(
                "INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sample_id, "verified", label, "human",
                 "2026-09-23T11:52:01+00:00", None, rel_path,
                 sha256(payload).hexdigest(), "video_a", 4, "yonma", region,
                 "pending"))
    conn.commit()
    conn.close()
    raw_manifest = raw_dir / "manifest.json"
    raw_manifest.write_text(json.dumps({"samples": manifest_samples}),
                            encoding="utf-8")

    verified_samples = datasets / "manifests" / "verified_samples.json"
    cases = []
    if with_case:
        # Deterministic pattern, black on the left and white on the right: each dHash row has at least 1 rising edge, weight >= 8,
        # will not accidentally hit criterion 1/criterion 2 with the small-weight synthetic hashes used in tests
        import numpy as np
        case_array = np.zeros((24, 16, 3), dtype=np.uint8)
        case_array[:, 8:] = 255
        buffer = io.BytesIO()
        Image.fromarray(case_array, "RGB").save(buffer, format="PNG")
        payload = buffer.getvalue()
        rel_path = "verified/tiles/1m/case_000001.png"
        target = datasets / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        cases.append({"case_id": "case_000001", "source": "runtime_unknown",
                      "kind": "tile", "label": "1m", "verified_by": "user",
                      "verified_at": "2026-09-22T11:35:16+00:00",
                      "crop_sha256": sha256(payload).hexdigest(),
                      "dataset_path": rel_path, "player_count": 3, "mode": "sanma"})
    verified_samples.write_text(json.dumps(cases), encoding="utf-8")

    return {"datasets_root": str(datasets), "reviews_db": str(db_path),
            "raw_manifest": str(raw_manifest),
            "verified_samples": str(verified_samples),
            "n_video_samples": blocks * per_block}


def run_cli(paths, command, out_dir, *extra):
    argv = [command, "--out-dir", str(out_dir),
            "--datasets-root", paths["datasets_root"],
            "--reviews-db", paths["reviews_db"],
            "--raw-manifest", paths["raw_manifest"],
            "--verified-samples", paths["verified_samples"], *extra]
    assert main(argv) == 0


def load_manifest(out_dir):
    return json.loads((Path(out_dir) / "split_manifest.json").read_text("utf-8"))


# ---------------------------------------------------------------- audit

def test_audit_detects_missing_mismatch_orphan_and_duplicate(tmp_path):
    paths = make_dataset(tmp_path, blocks=3, per_block=4, with_case=False)
    root = Path(paths["datasets_root"])

    # Missing file: delete the PNG corresponding to one DB record
    (root / "verified/tiles/1m/crop_0001.png").unlink()
    # Hash mismatch: overwrite the contents of another PNG
    (root / "verified/tiles/4m/crop_0002.png").write_bytes(make_png_bytes(99))
    # Orphan file: a PNG not in the DB/case
    orphan = root / "verified/tiles/1m/crop_orphan.png"
    orphan.write_bytes(make_png_bytes(77))
    # Exact duplicate: two sample_ids point to identical bytes (same crop_sha256)
    dup_payload = make_png_bytes(55)
    dup_paths = []
    for name in ("crop_dup_a", "crop_dup_b"):
        rel = f"verified/tiles/2z/{name}.png"
        (root / rel).write_bytes(dup_payload)
        dup_paths.append(rel)
    conn = sqlite3.connect(paths["reviews_db"])
    for rel, name in zip(dup_paths, ("crop_dup_a", "crop_dup_b")):
        conn.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (name, "verified", "2z", "human", "2026-09-23T12:00:00+00:00",
                      None, rel, sha256(dup_payload).hexdigest(), "video_a", 4,
                      "yonma", "river_bottom", "pending"))
    conn.commit()
    conn.close()

    out_dir = tmp_path / "out"
    run_cli(paths, "audit", out_dir)
    audit = json.loads((out_dir / "audit/gold_audit.json").read_text("utf-8"))
    issues = audit["issues"]
    assert [m["sample_id"] for m in issues["missing_files"]] == ["crop_0001"]
    assert [m["sample_id"] for m in issues["hash_mismatches"]] == ["crop_0002"]
    assert issues["orphan_files"] == ["verified/tiles/1m/crop_orphan.png"]
    assert issues["duplicate_hashes"] == [
        {"crop_sha256": sha256(dup_payload).hexdigest(),
         "sample_ids": ["crop_dup_a", "crop_dup_b"]}]
    assert audit["total_samples"] == 3 * 4 + 2
    for name in ("coverage_by_label.csv", "coverage_by_label_region.csv",
                 "coverage_by_label_orientation.csv"):
        assert (out_dir / "audit" / name).is_file()
    # Synthetic data has < 300 per class, declared comparison must flag mismatches
    assert audit["user_claim_check"]["claim_matches"] is False
    assert audit["user_claim_check"]["labels_below_threshold"]


# ---------------------------------------------------------------- split

def test_same_block_samples_share_split(tmp_path):
    paths = make_dataset(tmp_path)
    out_dir = tmp_path / "out"
    run_cli(paths, "split", out_dir, "--created-at", "2026-10-02T00:00:00+00:00")
    manifest = load_manifest(out_dir)
    by_block = {}
    for sample in manifest["samples"]:
        if sample["block_id"] is None:
            continue
        by_block.setdefault(sample["block_id"], set()).add(sample["split"])
    assert all(len(splits) == 1 for splits in by_block.values())


def test_near_dup_cluster_forced_same_split(tmp_path):
    # crop_0001 is in block 0, crop_0005 is in block 1 (same label/region), Hamming distance 1 -> criterion 1 merges directly into a cluster
    paths = make_dataset(tmp_path, hash_overrides={
        "crop_0001": "0000000000000000", "crop_0005": "0000000000000001"})
    out_dir = tmp_path / "out"
    run_cli(paths, "split", out_dir, "--created-at", "2026-10-02T00:00:00+00:00")
    manifest = load_manifest(out_dir)
    splits = {s["sample_id"]: s["split"] for s in manifest["samples"]}
    assert splits["crop_0001"] == splits["crop_0005"]
    clusters = {s["sample_id"]: s["cluster_id"] for s in manifest["samples"]}
    assert clusters["crop_0001"] == clusters["crop_0005"]
    report = json.loads((out_dir / "split_report.json").read_text("utf-8"))
    assert report["near_dup"]["hamming_histogram"]["1"] == 1
    assert report["near_dup"]["pairs_matched_hamming"] == 1


def _write_png(path, array):
    Image.fromarray(array, "RGB").save(path, format="PNG")


def test_near_dup_pixel_criterion(tmp_path):
    import numpy as np
    # Criterion 2 hit: crop_0001 (block 0) and crop_0005 (block 1) have the same label/region,
    # manifest Hamming distance 2 (> criterion 1's 1), but the PNG contents are nearly identical (same noise image with 1 pixel changed)
    paths = make_dataset(tmp_path, hash_overrides={
        "crop_0001": "0000000000000000", "crop_0005": "0000000000000003",
        "crop_0002": "8000000000000000", "crop_0006": "8000000000000003"})
    root = Path(paths["datasets_root"])
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (24, 16, 3), dtype=np.uint8)
    variant = base.copy()
    variant[0, 0, 0] ^= 1
    _write_png(root / "verified/tiles/1m/crop_0001.png", base)
    _write_png(root / "verified/tiles/1m/crop_0005.png", variant)
    # Criterion 2 not hit: crop_0002 (block 0) and crop_0006 (block 1) have the same label/region,
    # Hamming distance is also 2, but the contents are pure black vs pure white, pixel difference far exceeds the threshold
    _write_png(root / "verified/tiles/4m/crop_0002.png",
               np.zeros((24, 16, 3), dtype=np.uint8))
    _write_png(root / "verified/tiles/4m/crop_0006.png",
               np.full((24, 16, 3), 255, dtype=np.uint8))

    out_dir = tmp_path / "out"
    run_cli(paths, "split", out_dir, "--created-at", "2026-10-02T00:00:00+00:00")
    manifest = load_manifest(out_dir)
    by_id = {s["sample_id"]: s for s in manifest["samples"]}
    assert by_id["crop_0001"]["cluster_id"] == by_id["crop_0005"]["cluster_id"]
    assert by_id["crop_0001"]["split"] == by_id["crop_0005"]["split"]
    assert by_id["crop_0002"]["cluster_id"] != by_id["crop_0006"]["cluster_id"]
    report = json.loads((out_dir / "split_report.json").read_text("utf-8"))
    near_dup = report["near_dup"]
    assert near_dup["pairs_pixel_tested"] == 2
    assert near_dup["pairs_matched_pixel"] == 1
    assert near_dup["pairs_matched_hamming"] == 0
    assert near_dup["criteria"] == {"near_dup_hamming": 1,
                                    "near_dup_hamming_px": 4,
                                    "near_dup_pixel": 0.025}


def test_split_ratios_fields_and_conservation(tmp_path):
    paths = make_dataset(tmp_path)
    out_dir = tmp_path / "out"
    run_cli(paths, "split", out_dir, "--created-at", "2026-10-02T00:00:00+00:00")
    manifest = load_manifest(out_dir)
    total = paths["n_video_samples"] + 1
    counts = manifest["counts"]
    assert counts["train"] + counts["val"] + counts["test"] == total
    assert len(manifest["samples"]) == total
    assert abs(counts["train"] / total - 0.70) < 0.12
    assert abs(counts["val"] / total - 0.15) < 0.12
    assert abs(counts["test"] / total - 0.15) < 0.12

    required = {"sample_id", "label", "path", "region", "orientation", "width",
                "height", "first_timestamp", "last_timestamp", "block_id",
                "cluster_id", "split"}
    for sample in manifest["samples"]:
        assert required <= set(sample)
        assert "datasets/verified/tiles/" in sample["path"].replace("\\", "/")
        assert sample["split"] in ("train", "val", "test")
    case = next(s for s in manifest["samples"] if s["sample_id"] == "case_000001")
    assert case["split"] == "train"
    assert case["block_id"] is None and case["first_timestamp"] is None
    png = next(s for s in manifest["samples"] if s["sample_id"] == "crop_0001")
    assert (png["width"], png["height"]) == (16, 24)
    assert manifest["params"] == {"block_seconds": BLOCK_SECONDS,
                                  "near_dup_hamming": 1, "near_dup_hamming_px": 4,
                                  "near_dup_pixel": 0.025,
                                  "val_frac": 0.15, "test_frac": 0.15}
    assert manifest["grouping"]["n_blocks"] == 6
    # split includes audit output by default
    assert (out_dir / "audit/gold_audit.json").is_file()
    report = json.loads((out_dir / "split_report.json").read_text("utf-8"))
    for key in ("label_counts_by_split", "region_by_split", "orientation_by_split",
                "time_coverage_by_split", "near_dup", "limitations"):
        assert key in report


def test_split_deterministic_same_seed(tmp_path):
    paths = make_dataset(tmp_path)
    out_a, out_b = tmp_path / "out_a", tmp_path / "out_b"
    for out_dir in (out_a, out_b):
        run_cli(paths, "split", out_dir, "--skip-audit",
                "--created-at", "2026-10-02T00:00:00+00:00")
    assert ((out_a / "split_manifest.json").read_bytes()
            == (out_b / "split_manifest.json").read_bytes())
    assert ((out_a / "split_report.json").read_bytes()
            == (out_b / "split_report.json").read_bytes())
