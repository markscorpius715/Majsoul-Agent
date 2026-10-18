"""Refresh review partitions from audited decisions, never from predictions."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil


def sync_exports(store, run: Path, export_root: Path) -> dict:
    run, export_root = Path(run), Path(export_root).resolve()
    records = json.loads((run / "snapshot.json").read_text(encoding="utf-8"))
    with store._connect() as connection:
        connection.execute("BEGIN")
        audits = {row["sample_id"]: dict(row) for row in connection.execute(
            "SELECT * FROM ai_review_audit WHERE run_id=?", (run.name,))}
    reviewed = export_root / "ai-reviewed"
    pending = export_root / "ai-unreviewed"
    groups = [reviewed / "corrected-and-retained", reviewed / "uncertain", pending]
    for folder in groups:
        folder.mkdir(parents=True, exist_ok=True)
    samples = {reviewed: [], pending: []}
    expected = {folder: set() for folder in groups}
    # Validate every source before changing any export.
    for index, record in enumerate(records):
        source = Path(record["path"])
        if sha256(source.read_bytes()).hexdigest() != record["metadata"]["crop_sha256"]:
            raise ValueError(f"Source hash mismatch: {source}")
        if (index + 1) % 5000 == 0:
            print(f"Validated source hashes: {index + 1}/{len(records)}", flush=True)
    for index, record in enumerate(records):
        sid = record["sample_id"]
        source = Path(record["path"])
        audit = audits.get(sid)
        folder = (groups[1] if audit and audit["decision"] == "uncertain"
                  else groups[0] if audit else pending)
        target = folder / source.name
        digest = record["metadata"]["crop_sha256"]
        if not target.exists():
            shutil.copy2(source, target)
        if sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Export hash mismatch: {target}")
        # Remove only validated redundant export copies inside these exact folders.
        for other in groups:
            stale = other / source.name
            if stale != target and stale.exists():
                if not stale.resolve().is_relative_to(export_root):
                    raise ValueError(f"Export escapes root: {stale}")
                if sha256(stale.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Conflicting export: {stale}")
                stale.unlink()
        expected[folder].add(source.name)
        base = reviewed if audit else pending
        samples[base].append(dict(
            sample_id=sid, snapshot_index=index, path=target.relative_to(base).as_posix(),
            source_path=str(source), sha256=digest, ai_review=audit,
            label=None, training_allowed=False, metadata=record["metadata"]))
        if (index + 1) % 5000 == 0:
            print(f"Synchronized exports: {index + 1}/{len(records)}", flush=True)
    for folder, names in expected.items():
        if {p.name for p in folder.glob("*.png")} != names:
            raise ValueError(f"Unexpected files in export directory: {folder}")
    report = dict(snapshot_count=len(records), ai_reviewed_count=len(samples[reviewed]),
                  not_ai_reviewed_count=len(samples[pending]),
                  decisions=dict(Counter(a["decision"] for a in audits.values())),
                  all_source_and_export_hashes_verified=True, disjoint_and_complete=True,
                  training_allowed=False, reviewed_subfolders={p.name: len(expected[p])
                      for p in groups[:2]})
    for base, entries in samples.items():
        _write_json(base / "manifest.json", dict(export=report, samples=entries))
    _write_json(run / "separation_report.json", report)
    (run / "separation-notes.md").write_text(
        "# AI review directory\n\n"
        "This directory is now synced by stage, based on the main repository ai_review_audit.\n\n"
        f"- ai-reviewed：{report['ai_reviewed_count']} tiles（corrected-and-retained and uncertain two subdirectories）。\n"
        f"- ai-unreviewed：{report['not_ai_reviewed_count']} tiles。\n\n"
        "These directories are traceable copies of the original snapshots, not human ground truth. Manually approved images are saved in "
        "datasets/verified/tiles/<label>/; manual status is based on the main repository reviews.\n"
        "Each apply batch automatically syncs the directory; sync can be safely rerun. The original raw is not moved or rewritten.\n",
        encoding="utf-8")
    return report


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
