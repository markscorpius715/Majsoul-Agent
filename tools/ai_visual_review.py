"""Prepare indexed contact sheets and apply explicitly inspected AI decisions."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont
from majsoul.continual.corpus_review import CorpusReviewStore


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "sheet", "context", "apply", "report", "sync"])
    parser.add_argument("--exports", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, default=ROOT / "datasets")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    snapshot = args.run / "snapshot.json"
    if args.command == "prepare":
        if snapshot.exists():
            raise ValueError("Snapshot already exists; resume this run instead")
        store = CorpusReviewStore(args.manifest, args.datasets)
        # SQLite backup is transactionally consistent even with the GUI open.
        with store._connect() as source, sqlite3.connect(
                args.run / "before.sqlite3") as destination:
            source.backup(destination)
        records = store.scan(status="pending", order="frequency_balanced_positions")
        data = [dict(asdict(record), path=str(record.path.resolve()))
                for record in records]
        snapshot.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"pending": len(data), "human_counts": store.counts()}))
        return
    records = json.loads(snapshot.read_text(encoding="utf-8"))
    if args.command == "context":
        from majsoul.continual.corpus_label_gui import SourceFrameReader
        reader = SourceFrameReader(args.manifest)
        record = records[args.start]
        try:
            canvas = reader.read(record["metadata"]["first_frame"])
            ImageDraw.Draw(canvas).rectangle(record["metadata"]["screen_box"], outline="magenta", width=8)
            canvas.thumbnail((1800, 1200))
            target = args.run / f"context_{args.start:05d}.png"
            canvas.save(target)
            print(target.resolve())
        finally:
            reader.close()
        return
    if args.command == "sheet":
        if args.count < 1 or args.start < 0:
            raise ValueError("Invalid sheet range")
        selected = records[args.start:args.start + args.count]
        cols, cell_w, cell_h = 8, 190, 160
        canvas = Image.new("RGB", (cols * cell_w, ((len(selected)+cols-1)//cols)*cell_h), "#242830")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 17)
        for offset, record in enumerate(selected):
            idx = args.start + offset
            x, y = (offset % cols)*cell_w, (offset//cols)*cell_h
            with Image.open(record["path"]) as source:
                crop = source.convert("RGB")
            crop.thumbnail((cell_w-8, cell_h-42), Image.Resampling.LANCZOS)
            # Enlarge small crops without changing their aspect ratio.
            scale = min((cell_w-8)/crop.width, (cell_h-42)/crop.height)
            crop = crop.resize((round(crop.width*scale), round(crop.height*scale)), Image.Resampling.NEAREST)
            canvas.paste(crop, (x+(cell_w-crop.width)//2, y+27))
            draw.text((x+4, y+3), f'{idx}: {record["prediction_label"]}', fill="white", font=font)
            draw.text((x+4, y+cell_h-20), f'n={record["metadata"].get("occurrence_count", 1)}', fill="#cccccc", font=font)
        target = args.run / f"sheet_{args.start:05d}.png"
        canvas.save(target)
        print(target.resolve())
        return
    store = CorpusReviewStore(args.manifest, args.datasets)
    if args.command in {"apply", "sync"}:
        with store._connect() as source, sqlite3.connect(
                args.run / ("before_stage_" + time.strftime("%Y%m%d_%H%M%S") + ".sqlite3")) as destination:
            source.backup(destination)
    if args.command == "apply":
        if args.decisions is None:
            raise ValueError("--decisions required")
        decisions = json.loads(args.decisions.read_text(encoding="utf-8"))
        counts = Counter()
        seen = set()
        for decision in decisions:
            index = decision["index"]
            if type(index) is not int or not 0 <= index < len(records) or index in seen:
                raise ValueError("Invalid/duplicate snapshot index")
            if (decision.get("decision") == "correct" and
                    decision.get("label") == records[index]["prediction_label"]):
                raise ValueError(f"Index {index}: unchanged label must be recorded as keep")
            seen.add(index)
        for decision in decisions:
            record = records[decision["index"]]
            result = store.ai_review(
                record["sample_id"], run_id=args.run.name,
                reviewer="codex_direct_visual_review",
                expected_label=record["prediction_label"],
                **{key: value for key, value in decision.items() if key != "index"})
            counts[result] += 1
        print(json.dumps(counts))
    if args.command in {"apply", "sync"}:
        from majsoul.continual.ai_review_exports import sync_exports
        print(json.dumps(sync_exports(store, args.run, args.exports)))
    with store._connect() as connection:
        counts = dict(connection.execute(
            "SELECT decision, COUNT(*) FROM ai_review_audit WHERE run_id=? GROUP BY decision",
            (args.run.name,)))
        reviewed_ids = {row[0] for row in connection.execute(
            "SELECT sample_id FROM ai_review_audit WHERE run_id=?", (args.run.name,))}
    report = {"run_id": args.run.name, "snapshot_pending": len(records),
              "inspected": sum(counts.values()), "decisions": counts,
              "remaining_in_snapshot": len(records)-len(reviewed_ids),
              "next_uninspected_index": next((i for i, r in enumerate(records)
                  if r["sample_id"] not in reviewed_ids), None),
              "human_counts_now": store.counts(), "training_allowed": False}
    (args.run / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
