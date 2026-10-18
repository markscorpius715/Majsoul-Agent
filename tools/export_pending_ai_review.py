"""Copy unaudited snapshot images without changing review data or originals."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.snapshot.read_text(encoding='utf-8'))
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        audits = [dict(r) for r in db.execute(
            'SELECT * FROM ai_review_audit WHERE run_id=?', (args.run_id,))]
        predictions = {r['sample_id']: dict(r) for r in db.execute('SELECT * FROM predictions')}
        human_ids = {r['sample_id'] for r in db.execute('SELECT sample_id FROM reviews')}
    audited_ids = {r['sample_id'] for r in audits}
    snapshot_ids = {r['sample_id'] for r in records}
    if len(snapshot_ids) != len(records) or not audited_ids <= snapshot_ids:
        raise ValueError('Snapshot IDs are duplicated or do not cover this audit run')
    corrections = [r for r in audits if r['decision'] == 'correct']
    for audit in corrections:
        expected = json.loads(audit['new_prediction'])
        actual = predictions.get(audit['sample_id'])
        if actual != expected or actual['status'] != 'awaiting_human_confirmation':
            raise ValueError(f"Persisted correction differs: {audit['sample_id']}")
    selected = [(i, r) for i, r in enumerate(records) if r['sample_id'] not in audited_ids]
    if any(r['sample_id'] in human_ids for _, r in selected):
        raise ValueError('Some remaining snapshot samples have since been human reviewed')
    names = [Path(r['path']).name for _, r in selected]
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError('Duplicate destination filenames')
    total_bytes = sum(Path(r['path']).stat().st_size for _, r in selected)
    if shutil.disk_usage(args.output.resolve().parent).free < total_bytes:
        raise ValueError('Insufficient disk space')
    args.output.mkdir(exist_ok=False)
    exported = []
    for index, record in selected:
        source = Path(record['path'])
        target = args.output / source.name
        shutil.copy2(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != source_digest or digest != record['metadata']['crop_sha256']:
            raise ValueError(f'Image hash mismatch: {source}')
        exported.append({'sample_id': record['sample_id'], 'snapshot_index': index,
                         'path': target.name, 'source_path': str(source), 'sha256': digest,
                         'label': None, 'training_allowed': False,
                         'prediction': predictions.get(record['sample_id']),
                         'metadata': record['metadata']})
        if len(exported) % 1000 == 0:
            print(f'Copied and hash-verified {len(exported)}/{len(selected)}', flush=True)
    actual_images = {p.name for p in args.output.iterdir() if p.is_file()}
    if actual_images != set(names):
        raise ValueError('Exported file set mismatch')
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'run_id': args.run_id, 'snapshot': str(args.snapshot.resolve()),
              'database': str(args.database.resolve()), 'snapshot_count': len(records),
              'audited_count': len(audits), 'audit_decisions': dict(Counter(r['decision'] for r in audits)),
              'persisted_corrections_verified': len(corrections), 'exported_count': len(exported),
              'all_hashes_verified': True, 'originals_preserved': True,
              'human_review_modified': False, 'training_allowed': False,
              'note': 'Static export of images awaiting both direct AI review and human review.'}
    (args.output / 'manifest.json').write_text(json.dumps(
        {'export': report, 'samples': exported}, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'export_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
