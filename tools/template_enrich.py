"""Enrich the tile template library from verified sources.

Two sources, both human-verified only:

1. ``--gold``: pull verified samples from the corpus review database
   (``reviews.status == 'verified'``), preferring upright regions
   (self_hand / river_bottom / river_top / dora_indicators).  Meld and
   left/right river crops are skipped because they are stored as-seen and may
   be sideways, which would pollute the upright template set.
2. ``--verified-json``: copy an explicit operator-labeled mapping produced by
   visual inspection (see artifacts/roi/browser-20261002-eval/east2_labels.json).

Every copied file keeps provenance in its name (``gold_<region>_<id>`` or
``op_<source>_<i>``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import cv2

UPRIGHT_REGION_QUOTA = {
    "self_hand": 2,
    "river_bottom": 1,
    "river_top": 1,
    "dora_indicators": 1,
}


def enrich_gold(db_path: str, dataset_root: str, templates: str, quotas) -> int:
    db = sqlite3.connect(db_path)
    copied = 0
    labels = [row[0] for row in db.execute(
        "select distinct label from reviews where status='verified' order by label")]
    for label in labels:
        for region, quota in quotas.items():
            rows = db.execute(
                "select dataset_path, crop_sha256 from reviews "
                "where status='verified' and label=? and region=? "
                "order by sample_id", (label, region)).fetchall()
            kept = 0
            for dataset_path, sha in rows:
                if kept >= quota:
                    break
                source = Path(dataset_root) / dataset_path
                if not source.exists():
                    continue
                image = cv2.imread(str(source))
                if image is None:
                    continue
                height, width = image.shape[:2]
                if height <= width:
                    continue  # skip sideways tiles; templates stay upright
                target_dir = Path(templates) / label
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"gold_{region}_{sha[:10]}.png"
                if target.exists():
                    continue
                shutil.copyfile(source, target)
                kept += 1
                copied += 1
    return copied


def enrich_verified(mapping_path: str, templates: str) -> int:
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    base = Path(mapping_path).parent / "east2_tiles"
    copied = 0
    for i, entry in enumerate(mapping["tiles"]):
        source = base / entry["file"]
        if not source.exists():
            raise SystemExit(f"missing verified crop: {source}")
        target_dir = Path(templates) / entry["label"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"op_east2_{i:02d}_{source.stem}.png"
        shutil.copyfile(source, target)
        copied += 1
    return copied


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", action="store_true",
                        help="pull verified samples from the corpus review DB")
    parser.add_argument("--db", default="datasets/manifests/corpus_reviews/"
                        "captured_yonma_20260923_005221_tile_crops.sqlite3")
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--verified-json", default=None)
    parser.add_argument("--templates", default="artifacts/templates/tiles")
    args = parser.parse_args(argv)
    total = 0
    if args.gold:
        total += enrich_gold(args.db, args.dataset_root, args.templates,
                             UPRIGHT_REGION_QUOTA)
    if args.verified_json:
        total += enrich_verified(args.verified_json, args.templates)
    print(f"copied {total} templates into {args.templates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
