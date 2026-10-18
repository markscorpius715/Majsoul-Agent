#!/usr/bin/env python3
"""Offline tool for the gold-label tile dataset: audit, coverage report, and train/val/test split grouped by time.

Usage (from the project root):
    .venv/Scripts/python tools/tile_gold_dataset.py audit --out-dir artifacts/tile-augment-20261002
    .venv/Scripts/python tools/tile_gold_dataset.py split --out-dir artifacts/tile-augment-20261002

split runs audit first by default (can be disabled with --skip-audit). It is read-only on the dataset and does not modify any human labels or main-database data.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sqlite3
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASETS_ROOT = "datasets"
DEFAULT_REVIEWS_DB = ("datasets/manifests/corpus_reviews/"
                      "captured_yonma_20260923_005221_tile_crops.sqlite3")
DEFAULT_RAW_MANIFEST = ("datasets/raw/captured/captured_yonma_20260923_005221/"
                        "tile_crops/manifest.json")
DEFAULT_VERIFIED_SAMPLES = "datasets/manifests/verified_samples.json"

USER_CLAIM_TEXT = "User verbal statement: each class has >= 300 samples, red-five-man (0m) is slightly fewer"
USER_CLAIM_MIN_PER_LABEL = 300

SPLIT_NAMES = ("train", "val", "test")


# ---------------------------------------------------------------- Data loading

def _parse_visual_hash(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(str(value), 16)


def load_samples(reviews_db, raw_manifest, verified_samples, datasets_root):
    """Load DB verified rows + human-confirmed cases from verified_samples.json, and associate the original manifest fields.

    Returns a list of sample dicts; each sample contains sample_id/label/dataset_path/crop_sha256/
    source/player_count/mode/region/orientation/reviewed_by/reviewed_at/
    first_timestamp/last_timestamp/visual_hash(int or None)/is_case.
    """
    with open(raw_manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    raw_entries = manifest["samples"] if isinstance(manifest, dict) else manifest
    raw_index = {entry["id"]: entry for entry in raw_entries}

    samples = []
    seen_paths = set()
    conn = sqlite3.connect(str(reviews_db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT sample_id, label, reviewed_by, reviewed_at, dataset_path,"
            " crop_sha256, source, player_count, mode, region"
            " FROM reviews WHERE status = 'verified' ORDER BY sample_id"
        ).fetchall()
    finally:
        conn.close()
    for row in rows:
        raw = raw_index.get(row["sample_id"], {})
        samples.append({
            "sample_id": row["sample_id"],
            "label": row["label"],
            "dataset_path": row["dataset_path"],
            "crop_sha256": row["crop_sha256"],
            "source": row["source"],
            "player_count": row["player_count"],
            "mode": row["mode"],
            "region": row["region"],
            "orientation": raw.get("orientation"),
            "reviewed_by": row["reviewed_by"],
            "reviewed_at": row["reviewed_at"],
            "first_timestamp": raw.get("first_timestamp"),
            "last_timestamp": raw.get("last_timestamp"),
            "visual_hash": _parse_visual_hash(raw.get("visual_hash")),
            "is_case": False,
        })
        seen_paths.add(row["dataset_path"])

    with open(verified_samples, encoding="utf-8") as fh:
        cases = json.load(fh)
    for case in cases:
        path = case.get("dataset_path")
        if not path or not case.get("crop_sha256") or not case.get("label"):
            continue
        if path in seen_paths:
            continue
        seen_paths.add(path)
        samples.append({
            "sample_id": case.get("case_id") or Path(path).stem,
            "label": case["label"],
            "dataset_path": path,
            "crop_sha256": case["crop_sha256"],
            "source": case.get("source"),
            "player_count": case.get("player_count"),
            "mode": case.get("mode"),
            "region": case.get("region"),
            "orientation": case.get("orientation"),
            "reviewed_by": case.get("verified_by"),
            "reviewed_at": case.get("verified_at"),
            "first_timestamp": None,
            "last_timestamp": None,
            "visual_hash": None,
            "is_case": True,
        })
    return samples


# ---------------------------------------------------------------- File validation

def _png_size(data):
    """Read pixel dimensions from the PNG byte header; returns (None, None) for non-PNG."""
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    return None, None


def collect_file_info(samples, datasets_root):
    """Read each sample file's bytes, returns {sample_id: {exists, sha256, width, height}}."""
    info = {}
    for sample in samples:
        path = Path(datasets_root) / sample["dataset_path"]
        entry = {"exists": path.is_file(), "sha256": None, "width": None, "height": None}
        if entry["exists"]:
            data = path.read_bytes()
            entry["sha256"] = sha256(data).hexdigest()
            width, height = _png_size(data)
            if width is None:
                import cv2
                image = cv2.imread(str(path))
                if image is not None:
                    height, width = int(image.shape[1]), int(image.shape[0])
            entry["width"], entry["height"] = width, height
        info[sample["sample_id"]] = entry
    return info


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _nested_count(samples, key_a, key_b):
    table = defaultdict(Counter)
    for sample in samples:
        table[sample[key_a] or "<none>"][sample[key_b] or "<none>"] += 1
    return {a: dict(sorted(col.items())) for a, col in sorted(table.items())}


# ---------------------------------------------------------------- audit

def run_audit(samples, datasets_root, out_dir, file_info=None, verbose=True):
    """Gold-label set audit: file existence/hash consistency/orphan files/exact duplicates + coverage summary."""
    if file_info is None:
        file_info = collect_file_info(samples, datasets_root)

    missing_files, hash_mismatches = [], []
    for sample in samples:
        entry = file_info[sample["sample_id"]]
        if not entry["exists"]:
            missing_files.append({"sample_id": sample["sample_id"],
                                  "dataset_path": sample["dataset_path"]})
        elif entry["sha256"] != sample["crop_sha256"]:
            hash_mismatches.append({"sample_id": sample["sample_id"],
                                    "dataset_path": sample["dataset_path"],
                                    "expected": sample["crop_sha256"],
                                    "actual": entry["sha256"]})

    expected_paths = {sample["dataset_path"] for sample in samples}
    tiles_root = Path(datasets_root) / "verified" / "tiles"
    orphan_files = []
    if tiles_root.is_dir():
        for png in sorted(tiles_root.rglob("*.png")):
            rel = png.relative_to(datasets_root).as_posix()
            if rel not in expected_paths:
                orphan_files.append(rel)

    by_hash = defaultdict(list)
    for sample in samples:
        by_hash[sample["crop_sha256"]].append(sample["sample_id"])
    duplicate_hashes = [{"crop_sha256": digest, "sample_ids": sorted(ids)}
                        for digest, ids in sorted(by_hash.items()) if len(ids) > 1]

    counts = {
        "by_label": dict(sorted(Counter(s["label"] for s in samples).items())),
        "by_label_region": _nested_count(samples, "label", "region"),
        "by_label_orientation": _nested_count(samples, "label", "orientation"),
        "by_region_orientation": _nested_count(samples, "region", "orientation"),
        "by_source": dict(sorted(Counter(str(s["source"]) for s in samples).items())),
        "by_player_count": dict(sorted(Counter(str(s["player_count"]) for s in samples).items())),
        "by_mode": dict(sorted(Counter(str(s["mode"]) for s in samples).items())),
        "by_reviewed_by": dict(sorted(Counter(str(s["reviewed_by"]) for s in samples).items())),
        "by_reviewed_date": dict(sorted(
            Counter((s["reviewed_at"] or "<none>")[:10] for s in samples).items())),
    }

    size_extremes = {}
    for label in counts["by_label"]:
        sized = [(s["sample_id"], file_info[s["sample_id"]]["width"],
                  file_info[s["sample_id"]]["height"])
                 for s in samples if s["label"] == label
                 and file_info[s["sample_id"]]["width"] is not None]
        if not sized:
            continue
        size_extremes[label] = {
            "min_width": min(sized, key=lambda item: (item[1], item[0]))[:2],
            "max_width": max(sized, key=lambda item: (item[1], item[0]))[:2],
            "min_height": min(sized, key=lambda item: (item[2], item[0]))[::2],
            "max_height": max(sized, key=lambda item: (item[2], item[0]))[::2],
        }

    below_claim = {label: count for label, count in counts["by_label"].items()
                   if count < USER_CLAIM_MIN_PER_LABEL}
    audit = {
        "total_samples": len(samples),
        "case_samples": sum(1 for s in samples if s["is_case"]),
        "issues": {
            "missing_files": missing_files,
            "hash_mismatches": hash_mismatches,
            "orphan_files": orphan_files,
            "duplicate_hashes": duplicate_hashes,
        },
        "counts": counts,
        "label_size_extremes": size_extremes,
        "user_claim_check": {
            "claim": USER_CLAIM_TEXT,
            "min_per_label_threshold": USER_CLAIM_MIN_PER_LABEL,
            "labels_below_threshold": dict(sorted(below_claim.items())),
            "claim_matches": not below_claim,
        },
    }

    audit_dir = Path(out_dir) / "audit"
    _write_json(audit_dir / "gold_audit.json", audit)
    _write_csv(audit_dir / "coverage_by_label.csv", ["label", "count"],
               sorted(counts["by_label"].items()))
    _write_csv(audit_dir / "coverage_by_label_region.csv", ["label", "region", "count"],
               [(label, region, n) for label, regions in counts["by_label_region"].items()
                for region, n in regions.items()])
    _write_csv(audit_dir / "coverage_by_label_orientation.csv",
               ["label", "orientation", "count"],
               [(label, orientation, n)
                for label, orientations in counts["by_label_orientation"].items()
                for orientation, n in orientations.items()])

    if verbose:
        print(f"[audit] total samples: {audit['total_samples']} (including case {audit['case_samples']})")
        print(f"[audit] missing files {len(missing_files)}, hash mismatches {len(hash_mismatches)}, "
              f"orphan files {len(orphan_files)}, exact duplicate hashes {len(duplicate_hashes)}")
        print(f"[audit] {USER_CLAIM_TEXT}")
        if below_claim:
            print(f"[audit] measured below {USER_CLAIM_MIN_PER_LABEL} classes "
                  f"({len(below_claim)}): "
                  + ", ".join(f"{k}={v}" for k, v in sorted(below_claim.items())))
        else:
            print(f"[audit] all measured classes >= {USER_CLAIM_MIN_PER_LABEL},consistent with the claim")
        print(f"[audit] output: {audit_dir}")
    return {"audit": audit, "file_info": file_info}


# ---------------------------------------------------------------- split

class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _detector_hash_funcs():
    try:
        from majsoul.continual.detector import difference_hash, hamming_distance
        return difference_hash, hamming_distance
    except Exception:
        import cv2

        def difference_hash(image):
            gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if image.ndim == 3 else image)
            normalized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            bits = normalized[:, 1:] > normalized[:, :-1]
            value = 0
            for bit in bits.flat:
                value = (value << 1) | int(bit)
            return value

        def hamming_distance(left, right):
            return (left ^ right).bit_count()

        return difference_hash, hamming_distance


def _compute_case_hashes(samples, datasets_root, file_info):
    """Case samples have no manifest visual_hash, so compute dHash on the fly from the file pixels."""
    difference_hash, _ = _detector_hash_funcs()
    import cv2
    for sample in samples:
        if not sample["is_case"] or sample["visual_hash"] is not None:
            continue
        if not file_info[sample["sample_id"]]["exists"]:
            continue
        image = cv2.imread(str(Path(datasets_root) / sample["dataset_path"]))
        if image is not None:
            sample["visual_hash"] = difference_hash(image)


def _read_image_like_corpus(path):
    """PNG reading (BGR) exactly consistent with majsoul.calibration.frames.read_image."""
    try:
        from majsoul.calibration.frames import read_image
        return read_image(path)
    except ImportError:
        import cv2
        import numpy as np
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        return image


def _thumbnail(image):
    """Exactly consistent with majsoul.continual.crop_corpus._thumbnail."""
    import cv2
    return cv2.resize(image, (48, 64), interpolation=cv2.INTER_AREA)


def _build_clusters(samples, datasets_root, *, near_dup_hamming,
                    near_dup_hamming_px, near_dup_pixel):
    """Dual-criterion near-duplicate clustering; satisfying either one merges into the same union-find cluster:

    1. Hamming distance <= near_dup_hamming (merged directly, without computing pixel difference);
    2. Hamming distance <= near_dup_hamming_px and 48x64 thumbnail mean absolute difference
       <= near_dup_pixel (consistent with the dedup criterion used during corpus construction).
    Pairs with Hamming distance > near_dup_hamming_px are not checked. Case sample regions are unreliable,
    so they are compared with all same-label samples using the same dual criterion.
    Returns (uf, histogram, stats); histogram counts the number of sample pairs at distances 1..8.
    """
    _, hamming_distance = _detector_hash_funcs()
    import cv2
    import numpy as np
    uf = _UnionFind()
    histogram = {distance: 0 for distance in range(1, 9)}
    stats = {"pairs_hamming": 0, "pairs_pixel_tested": 0, "pairs_pixel": 0}
    thumbnails = {}

    def thumbnail(sample):
        cached = thumbnails.get(sample["sample_id"])
        if cached is None:
            image = _read_image_like_corpus(
                Path(datasets_root) / sample["dataset_path"])
            cached = _thumbnail(image)
            thumbnails[sample["sample_id"]] = cached
        return cached

    def consider(left, right, distance):
        if distance <= near_dup_hamming:
            uf.union(left["sample_id"], right["sample_id"])
            stats["pairs_hamming"] += 1
        elif distance <= near_dup_hamming_px:
            stats["pairs_pixel_tested"] += 1
            difference = float(np.mean(cv2.absdiff(
                thumbnail(left), thumbnail(right)))) / 255.0
            if difference <= near_dup_pixel:
                uf.union(left["sample_id"], right["sample_id"])
                stats["pairs_pixel"] += 1

    hashed = [s for s in samples if s["visual_hash"] is not None]
    groups = defaultdict(list)
    for sample in hashed:
        if not sample["is_case"]:
            groups[(sample["label"], sample["region"])].append(sample)
    for group in groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                distance = hamming_distance(group[i]["visual_hash"],
                                            group[j]["visual_hash"])
                if 1 <= distance <= 8:
                    histogram[distance] += 1
                if distance <= near_dup_hamming_px:
                    consider(group[i], group[j], distance)

    by_label = defaultdict(list)
    for sample in hashed:
        by_label[sample["label"]].append(sample)
    for sample in hashed:
        if not sample["is_case"]:
            continue
        for other in by_label[sample["label"]]:
            if other["sample_id"] == sample["sample_id"] or other["is_case"]:
                continue
            distance = hamming_distance(sample["visual_hash"],
                                        other["visual_hash"])
            if distance <= near_dup_hamming_px:
                consider(sample, other, distance)
    return uf, histogram, stats


def _evaluate_assignment(assignment, block_stats, all_labels, all_regions, targets):
    counts = [0, 0, 0]
    labels_in = [set(), set(), set()]
    regions_in = [set(), set(), set()]
    for block_id, split_idx in enumerate(assignment):
        stats = block_stats[block_id]
        counts[split_idx] += stats["count"]
        labels_in[split_idx] |= stats["labels"]
        regions_in[split_idx] |= stats["regions"]
    if len(assignment) >= 3 and min(counts) == 0:
        return float("inf")
    total = sum(counts)
    if total == 0:
        return float("inf")
    deviation = sum(abs(counts[i] / total - targets[i]) for i in range(3))
    missing_labels = sum(len(all_labels - labels_in[i]) for i in (1, 2))
    missing_regions = sum(len(all_regions - regions_in[i]) for i in (1, 2))
    adjacency = sum(1 for i in range(len(assignment) - 1)
                    if assignment[i] == assignment[i + 1] and assignment[i] != 0)
    return (1000.0 * missing_labels + 200.0 * missing_regions
            + 10.0 * adjacency + deviation)


def _search_block_assignment(n_blocks, block_stats, all_labels, all_regions,
                             targets, seed, iterations):
    rng = random.Random(seed)
    best_assignment, best_score = None, float("inf")
    for _ in range(iterations):
        assignment = tuple(rng.randrange(3) for _ in range(n_blocks))
        score = _evaluate_assignment(assignment, block_stats, all_labels,
                                     all_regions, targets)
        if score < best_score:
            best_assignment, best_score = assignment, score
    if best_assignment is None:
        raise RuntimeError("Block-assignment search failed: no feasible assignment found")
    return best_assignment, best_score


def run_split(samples, datasets_root, out_dir, *, block_seconds=255,
              near_dup_hamming=1, near_dup_hamming_px=4, near_dup_pixel=0.025,
              val_frac=0.15, test_frac=0.15, seed=42,
              search_iterations=2000, created_at=None, file_info=None,
              reviews_db=DEFAULT_REVIEWS_DB, raw_manifest=DEFAULT_RAW_MANIFEST,
              verified_samples=DEFAULT_VERIFIED_SAMPLES, verbose=True):
    if file_info is None:
        file_info = collect_file_info(samples, datasets_root)
    _compute_case_hashes(samples, datasets_root, file_info)

    video_samples = [s for s in samples if not s["is_case"]]
    case_samples = [s for s in samples if s["is_case"]]
    for sample in video_samples:
        if sample["first_timestamp"] is None:
            raise ValueError(f"samples {sample['sample_id']} missing first_timestamp,cannot assign time blocks")
        sample["block_id"] = int(sample["first_timestamp"] // block_seconds)
    n_blocks = max(s["block_id"] for s in video_samples) + 1 if video_samples else 0

    uf, histogram, near_dup_stats = _build_clusters(
        samples, datasets_root, near_dup_hamming=near_dup_hamming,
        near_dup_hamming_px=near_dup_hamming_px, near_dup_pixel=near_dup_pixel)

    block_stats = []
    for block_id in range(n_blocks):
        members = [s for s in video_samples if s["block_id"] == block_id]
        block_stats.append({
            "count": len(members),
            "labels": {s["label"] for s in members},
            "regions": {s["region"] for s in members},
        })
    all_labels = {s["label"] for s in video_samples}
    all_regions = {s["region"] for s in video_samples}
    targets = (1.0 - val_frac - test_frac, val_frac, test_frac)
    assignment, best_score = _search_block_assignment(
        n_blocks, block_stats, all_labels, all_regions, targets, seed,
        search_iterations)

    split_of = {}
    for sample in video_samples:
        split_of[sample["sample_id"]] = SPLIT_NAMES[assignment[sample["block_id"]]]
    for sample in case_samples:
        split_of[sample["sample_id"]] = "train"

    clusters = defaultdict(list)
    for sample in samples:
        clusters[uf.find(sample["sample_id"])].append(sample)

    moved_samples = 0
    case_forced = []
    for root, members in sorted(clusters.items()):
        present = {split_of[m["sample_id"]] for m in members}
        if len(present) <= 1:
            continue
        cases_in = [m for m in members if m["is_case"]]
        if cases_in:
            target = "train"
        else:
            tallies = Counter(split_of[m["sample_id"]] for m in members)
            target = min(SPLIT_NAMES, key=lambda name: (-tallies[name],
                                                        SPLIT_NAMES.index(name)))
        moved = [m for m in members if split_of[m["sample_id"]] != target]
        for member in moved:
            split_of[member["sample_id"]] = target
        moved_samples += len(moved)
        if cases_in:
            case_forced.append({
                "case_ids": sorted(m["sample_id"] for m in cases_in),
                "moved_sample_ids": sorted(m["sample_id"] for m in moved),
            })

    cluster_ids = {}
    for cluster_id, (root, members) in enumerate(sorted(
            clusters.items(), key=lambda item: min(m["sample_id"] for m in item[1]))):
        for member in members:
            cluster_ids[member["sample_id"]] = cluster_id
    n_cross_block = sum(1 for members in clusters.values()
                        if len({m.get("block_id") for m in members
                                if m.get("block_id") is not None}) > 1)

    datasets_prefix = Path(datasets_root).as_posix().rstrip("/")
    manifest_samples = []
    for sample in sorted(samples, key=lambda s: s["sample_id"]):
        entry = file_info[sample["sample_id"]]
        manifest_samples.append({
            "sample_id": sample["sample_id"],
            "label": sample["label"],
            "path": f"{datasets_prefix}/{sample['dataset_path']}",
            "region": sample["region"],
            "orientation": sample["orientation"],
            "width": entry["width"],
            "height": entry["height"],
            "first_timestamp": sample["first_timestamp"],
            "last_timestamp": sample["last_timestamp"],
            "block_id": sample.get("block_id"),
            "cluster_id": (None if sample["is_case"] and
                           len(clusters[uf.find(sample["sample_id"])]) == 1
                           else cluster_ids[sample["sample_id"]]),
            "split": split_of[sample["sample_id"]],
        })

    counts = Counter(split_of.values())
    total = len(manifest_samples)
    manifest = {
        "version": 1,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "params": {"block_seconds": block_seconds, "near_dup_hamming": near_dup_hamming,
                   "near_dup_hamming_px": near_dup_hamming_px,
                   "near_dup_pixel": near_dup_pixel,
                   "val_frac": val_frac, "test_frac": test_frac},
        "sources": {"reviews_db": str(reviews_db), "raw_manifest": str(raw_manifest),
                    "verified_samples": str(verified_samples),
                    "verified_root": f"{datasets_prefix}/verified"},
        "grouping": {
            "method": "time_blocks+near_dup_clusters",
            "n_blocks": n_blocks,
            "moved_samples": moved_samples,
            "notes": ("Samples are grouped by first_timestamp into time blocks of about block_seconds seconds, and whole blocks are assigned to partitions;"
                      "Samples within the same (label, region) that satisfy one of the two criteria form an atomic cluster: "
                      "(1) dHash Hamming distance <= near_dup_hamming; "
                      "(2) Hamming distance <= near_dup_hamming_px and 48x64 thumbnail mean absolute difference"
                      " <= near_dup_pixel (consistent with the deduplication criteria during corpus construction);"
                      "Minority members of cross-partition clusters are moved into the majority partition (ties go to train);"
                      "case samples without reliable timestamps are fixed into train, and their near-duplicate clusters are merged entirely into train."),
        },
        "counts": {name: counts.get(name, 0) for name in SPLIT_NAMES},
        "samples": manifest_samples,
    }

    label_by_split = {name: dict(sorted(Counter(
        s["label"] for s in manifest_samples if s["split"] == name).items()))
        for name in SPLIT_NAMES}
    region_by_split = defaultdict(lambda: {name: 0 for name in SPLIT_NAMES})
    orientation_by_split = defaultdict(lambda: {name: 0 for name in SPLIT_NAMES})
    for sample in manifest_samples:
        region_by_split[sample["region"] or "<none>"][sample["split"]] += 1
        orientation_by_split[sample["orientation"] or "<none>"][sample["split"]] += 1
    time_coverage = {}
    for name in SPLIT_NAMES:
        times = [(s["first_timestamp"], s["last_timestamp"])
                 for s in manifest_samples
                 if s["split"] == name and s["first_timestamp"] is not None]
        time_coverage[name] = ({"min_first_timestamp": min(t[0] for t in times),
                                "max_last_timestamp": max(t[1] or t[0] for t in times)}
                               if times else None)

    report = {
        "seed": seed,
        "params": manifest["params"],
        "search": {"iterations": search_iterations, "best_score": best_score},
        "counts": manifest["counts"],
        "final_fractions": ({name: counts.get(name, 0) / total for name in SPLIT_NAMES}
                            if total else {}),
        "target_fractions": {"train": targets[0], "val": targets[1], "test": targets[2]},
        "fraction_deviation": ({name: abs(counts.get(name, 0) / total - target)
                                for name, target in zip(SPLIT_NAMES, targets)}
                               if total else {}),
        "block_assignment": {str(b): SPLIT_NAMES[s] for b, s in enumerate(assignment)},
        "label_counts_by_split": label_by_split,
        "region_by_split": {k: v for k, v in sorted(region_by_split.items())},
        "orientation_by_split": {k: v for k, v in sorted(orientation_by_split.items())},
        "time_coverage_by_split": time_coverage,
        "near_dup": {
            "hamming_histogram": {str(k): v for k, v in histogram.items()},
            "criteria": {"near_dup_hamming": near_dup_hamming,
                         "near_dup_hamming_px": near_dup_hamming_px,
                         "near_dup_pixel": near_dup_pixel},
            "pairs_matched_hamming": near_dup_stats["pairs_hamming"],
            "pairs_pixel_tested": near_dup_stats["pairs_pixel_tested"],
            "pairs_matched_pixel": near_dup_stats["pairs_pixel"],
            "n_clusters": len(clusters),
            "n_multi_member_clusters": sum(1 for m in clusters.values() if len(m) > 1),
            "max_cluster_size": max((len(m) for m in clusters.values()), default=0),
            "n_clusters_over_10": sum(1 for m in clusters.values() if len(m) > 10),
            "n_cross_block_clusters": n_cross_block,
            "moved_samples": moved_samples,
            "case_near_dup_forced_train": case_forced,
        },
        "limitations": [
            "Almost all samples come from the same four-player mahjong recording captured_yonma_20260923_005221, with no independent-source test set.",
            "Partitioning is done by time blocks, which can reduce temporally adjacent leakage, but residual similarity may still exist for near-duplicates in the same cluster.",
            "The case samples (4) have no reliable timestamps and are fixed into train.",
            "Near-duplicate protection is based on a 64-bit dHash Hamming distance threshold; samples that are semantically similar but have a large hash distance cannot be identified.",
            "The recognized screen is a four-player mahjong layout; using it directly for a sanma model requires attention to domain differences.",
        ],
    }

    out_dir = Path(out_dir)
    _write_json(out_dir / "split_manifest.json", manifest)
    _write_json(out_dir / "split_report.json", report)
    _write_json(out_dir / "hamming_histogram.json",
                {"near_dup_hamming": near_dup_hamming,
                 "near_dup_hamming_px": near_dup_hamming_px,
                 "near_dup_pixel": near_dup_pixel,
                 "histogram": report["near_dup"]["hamming_histogram"],
                 "pairs_matched_hamming": near_dup_stats["pairs_hamming"],
                 "pairs_pixel_tested": near_dup_stats["pairs_pixel_tested"],
                 "pairs_matched_pixel": near_dup_stats["pairs_pixel"]})

    if verbose:
        print(f"[split] time blocks {n_blocks} items (block_seconds={block_seconds}), "
              f"clusters {len(clusters)} items(across blocks {n_cross_block}), moved samples {moved_samples}")
        print(f"[split] counts: train={counts.get('train', 0)} val={counts.get('val', 0)} "
              f"test={counts.get('test', 0)}")
        if total:
            print("[split] ratio: " + " ".join(
                f"{name}={counts.get(name, 0) / total:.3f}(target {target:.3f})"
                for name, target in zip(SPLIT_NAMES, targets)))
        print(f"[split] output: {out_dir}/split_manifest.json, split_report.json")
    return {"manifest": manifest, "report": report}


# ---------------------------------------------------------------- CLI

def build_parser():
    parser = argparse.ArgumentParser(
        prog="tile_gold_dataset",
        description="Gold-label tile dataset audit and time-grouped train/val/test split (read-only dataset)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        sub.add_argument("--out-dir", required=True, help="Output directory")
        sub.add_argument("--datasets-root", default=DEFAULT_DATASETS_ROOT)
        sub.add_argument("--reviews-db", default=DEFAULT_REVIEWS_DB)
        sub.add_argument("--raw-manifest", default=DEFAULT_RAW_MANIFEST)
        sub.add_argument("--verified-samples", default=DEFAULT_VERIFIED_SAMPLES)

    add_common(subparsers.add_parser("audit", help="Gold-label set audit + coverage report"))

    split_parser = subparsers.add_parser("split", help="Split train/val/test by time blocks + near-duplicate clusters")
    add_common(split_parser)
    split_parser.add_argument("--block-seconds", type=float, default=255.0)
    split_parser.add_argument("--near-dup-hamming", type=int, default=1,
                              help="Criterion 1: Hamming distance <= this value is directly clustered")
    split_parser.add_argument("--near-dup-hamming-px", type=int, default=4,
                              help="Criterion 2: only pairs with Hamming distance <= this value are checked for pixel difference")
    split_parser.add_argument("--near-dup-pixel", type=float, default=0.025,
                              help="Criterion 2: 48x64 thumbnail mean absolute difference threshold")
    split_parser.add_argument("--val-frac", type=float, default=0.15)
    split_parser.add_argument("--test-frac", type=float, default=0.15)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--search-iterations", type=int, default=2000)
    split_parser.add_argument("--created-at", default=None,
                              help="Fixed manifest created_at (for reproducible byte-level consistency)")
    split_parser.add_argument("--skip-audit", action="store_true",
                              help="Skip the audit that runs by default with split")
    return parser


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    args = build_parser().parse_args(argv)
    samples = load_samples(args.reviews_db, args.raw_manifest,
                           args.verified_samples, args.datasets_root)
    if args.command == "audit":
        run_audit(samples, args.datasets_root, args.out_dir)
        return 0
    file_info = None
    if not args.skip_audit:
        file_info = run_audit(samples, args.datasets_root, args.out_dir)["file_info"]
    run_split(samples, args.datasets_root, args.out_dir,
              block_seconds=args.block_seconds,
              near_dup_hamming=args.near_dup_hamming,
              near_dup_hamming_px=args.near_dup_hamming_px,
              near_dup_pixel=args.near_dup_pixel,
              val_frac=args.val_frac, test_frac=args.test_frac,
              seed=args.seed, search_iterations=args.search_iterations,
              created_at=args.created_at, file_info=file_info,
              reviews_db=args.reviews_db, raw_manifest=args.raw_manifest,
              verified_samples=args.verified_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
