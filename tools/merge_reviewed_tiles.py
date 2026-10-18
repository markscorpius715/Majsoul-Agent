"""Export reviewed crops with provenance, byte deduplication and coverage gaps.

AI reviews remain AI reviews. No labels or review states are written to sources.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sqlite3


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def export(root: Path, output: Path):
    labels = [f'{n}{s}' for s in 'mps' for n in range(10)] + [f'{n}z' for n in range(1, 8)]
    records, pending, excluded = [], [], Counter()
    regions = set()
    for db in sorted((root / 'datasets/manifests/corpus_reviews').glob('*.sqlite3')):
        with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as con:
            con.row_factory = sqlite3.Row
            manifest = Path(dict(con.execute('SELECT key,value FROM corpus_metadata'))['source_manifest'])
            samples = json.loads(manifest.read_text(encoding='utf-8-sig'))['samples']
            reviews = {r['sample_id']: dict(r) for r in con.execute('SELECT * FROM reviews')}
            audits = {r['sample_id']: dict(r) for r in con.execute('SELECT * FROM ai_review_audit ORDER BY audit_id')}
            predictions = {r['sample_id']: dict(r) for r in con.execute('SELECT * FROM predictions')}
        for s in samples:
            sid = s['id']
            regions.add(s['region'])
            r, a, p = reviews.get(sid), audits.get(sid), predictions.get(sid)
            record = dict(s, source_path=str((manifest.parent / s['path']).resolve()),
                          human_review=r, ai_review=a, training_allowed=False)
            if r:
                if r['status'] != 'verified':
                    excluded['human_rejected'] += 1
                    continue
                record.update(label=r['label'], review_status='human_verified', training_allowed=True)
            elif a and a['decision'] in ('keep', 'correct'):
                evidence = json.loads(a['new_prediction'] or a['previous_prediction'] or '{}') or {}
                if not p or p['label'] != evidence.get('label'):
                    excluded['stale_ai_evidence'] += 1
                    continue
                if p['label'] not in labels:
                    excluded['ai_non_tile'] += 1
                    continue
                record.update(label=p['label'], review_status='ai_reviewed_awaiting_human')
            else:
                record.update(label=None, predicted_label=p['label'] if p else None)
                pending.append(record)
                excluded['ai_uncertain' if a else 'unreviewed'] += 1
                continue
            records.append(record)
    for r in json.loads((root / 'datasets/manifests/verified_samples.json').read_text(encoding='utf-8-sig')):
        if r['kind'] != 'tile':
            continue
        matches = list((root / 'artifacts/unknown_cases').glob(f"**/{r['case_id']}/metadata.json"))
        metadata = [json.loads(p.read_text(encoding='utf-8-sig')) for p in matches]
        metadata = [m for m in metadata if m.get('crop_sha256') == r['crop_sha256']]
        region = metadata[0].get('region', 'unknown') if metadata else 'unknown'
        # Region metadata may include an individual crop index.
        region = region.split('[')[0]
        records.append(dict(r, id=r['case_id'], region=region,
                            source_path=str(root / 'datasets' / r['dataset_path']),
                            human_review=r, ai_review=None, review_status='human_verified', training_allowed=True))
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for record in records:
        path = Path(record['source_path'])
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != record['crop_sha256']:
            raise ValueError(f'Hash mismatch: {path}')
        grouped[digest].append(record)
    exported, conflicts = [], []
    expected_files = set()
    for digest, sources in sorted(grouped.items()):
        human = [r for r in sources if r['review_status'] == 'human_verified']
        authoritative = human or sources
        if len({r['label'] for r in authoritative}) != 1:
            conflicts.append({'sha256': digest, 'sources': sources})
            continue
        chosen = authoritative[0]
        dest = output / 'tiles' / chosen['label'] / f'{digest}.png'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(chosen['source_path'], dest)
        if sha256(dest.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Export hash mismatch: {dest}')
        expected_files.add(dest.resolve())
        exported.append(dict(label=chosen['label'], path=dest.relative_to(output).as_posix(),
                             crop_sha256=digest, review_status=chosen['review_status'],
                             training_allowed=chosen['training_allowed'],
                             regions=sorted({r['region'] for r in sources}), sources=sources))
    # Reruns cannot silently leave stale tiles.
    unexpected = {p.resolve() for p in (output / 'tiles').glob('*/*.png')} - expected_files
    if unexpected:
        raise ValueError('Output has stale files; use a new output directory')
    counts = Counter(r['label'] for r in exported)
    coverage = Counter((r['label'], region) for r in exported for region in r['regions'])
    candidates = Counter((r['predicted_label'], r['region']) for r in pending)
    target = max(counts.values(), default=0)
    gaps = []
    with (output / 'tile_counts.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'unique_images', 'target_no_downsampling', 'remaining'])
        for label in labels:
            writer.writerow([label, counts[label], target, target-counts[label]])
    with (output / 'region_coverage.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'region', 'unique_images', 'unverified_prediction_candidates'])
        for label in labels:
            for region in sorted(regions):
                n = coverage[label, region]
                writer.writerow([label, region, n, candidates[label, region]])
                if n == 0:
                    gaps.append({'label': label, 'region': region, 'unverified_candidates': candidates[label, region]})
    pending.sort(key=lambda r: (coverage[r['predicted_label'], r['region']] > 0,
                               counts[r['predicted_label']], -r.get('occurrence_count', 0), r['id']))
    summary = dict(unique_images=len(exported), source_records=len(records), duplicates=len(records)-sum(len(c['sources']) for c in conflicts)-len(exported),
                   review_status_counts=dict(Counter(r['review_status'] for r in exported)),
                   excluded=dict(excluded), conflicts=len(conflicts), target_per_class=target,
                   remaining_to_equal=sum(target-counts[l] for l in labels), missing_regions=gaps,
                   balanced=len(set(counts[l] for l in labels)) == 1, region_complete=not gaps,
                   source_hashes_verified=True, exported_hashes_verified=True)
    write_json(output / 'manifest.json', {'schema_version': 1, 'training_allowed': False, 'samples': exported})
    write_json(output / 'summary.json', summary)
    write_json(output / 'conflicts.json', conflicts)
    write_json(output / 'supplement_queue.json', pending)
    (output / 'README.md').write_text(
        '# AI-reviewed + human-reviewed\n\nThis is the union of the two parts and does not mean every image has been double-reviewed. Images are under tiles/<tile class>/, deduplicated by SHA-256; human labels take priority, and conflicts are listed separately.\n\n'
        'manifest.json retains source, mode, region, AI/human audit and per-image admission status. AI-only images still await human confirmation, and the whole directory cannot be used directly as a human gold-label training set.\n\n'
        'tile_counts.csv is the sole image count; region_coverage.csv is region coverage; supplement_queue.json is only unconfirmed candidates and does not count as supplementation. The goal is to keep all reviewed samples and then supplement up to the maximum tile-class count, without copying or rotating images to pad the numbers.\n\n'
        'Whether supplementation is complete is determined by balanced / region_complete in summary.json.\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.root.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))
