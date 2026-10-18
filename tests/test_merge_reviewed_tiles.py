import json
from hashlib import sha256

from PIL import Image
import pytest

from majsoul.continual.corpus_review import CorpusReviewStore
from tools.merge_reviewed_tiles import export


def make_store(root):
    crop = root / 'crop.png'
    Image.new('RGB', (20, 30), 'white').save(crop)
    samples = [dict(id=f'crop_{i}', path='crop.png', source='test',
                    player_count=4, mode='yonma', region=region,
                    crop_sha256=sha256(crop.read_bytes()).hexdigest())
               for i, region in enumerate(('self_hand', 'river_left'))]
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps(dict(corpus_id='merge_test', label_status='unverified',
                                       training_allowed=False, samples=samples)), encoding='utf-8')
    store = CorpusReviewStore(manifest, root / 'datasets')
    (root / 'datasets/manifests/verified_samples.json').write_text('[]', encoding='utf-8')
    store.save_predictions([dict(id=s['id'], label='5z', confidence=.9) for s in samples],
                           model='test', prompt_version='test')
    return store


def test_duplicate_human_priority_and_all_region_provenance(tmp_path):
    store = make_store(tmp_path)
    store.ai_review('crop_0', run_id='test', reviewer='ai', decision='keep',
                    expected_label='5z', reason='inspected')
    store.label('crop_1', '1z', verified_by='human')
    out = tmp_path / 'export'
    before = store.database_path.read_bytes()
    result = export(tmp_path, out)
    assert result['unique_images'] == 1 and result['duplicates'] == 1
    sample = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))['samples'][0]
    assert sample['label'] == '1z' and sample['review_status'] == 'human_verified'
    assert sample['regions'] == ['river_left', 'self_hand']
    assert store.database_path.read_bytes() == before
    assert export(tmp_path, out) == result


def test_conflicting_human_duplicates_are_quarantined(tmp_path):
    store = make_store(tmp_path)
    store.label('crop_0', '5z', verified_by='human')
    store.label('crop_1', '1z', verified_by='human')
    result = export(tmp_path, tmp_path / 'export')
    assert result['unique_images'] == 0 and result['conflicts'] == 1


def test_ai_only_stays_unverified_and_unknown_is_excluded(tmp_path):
    store = make_store(tmp_path)
    store.ai_review('crop_0', run_id='test', reviewer='ai', decision='keep',
                    expected_label='5z', reason='inspected')
    store.ai_review('crop_1', run_id='test', reviewer='ai', decision='uncertain',
                    expected_label='5z', reason='unclear')
    out = tmp_path / 'export'
    result = export(tmp_path, out)
    sample = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))['samples'][0]
    assert sample['training_allowed'] is False
    assert result['excluded']['ai_uncertain'] == 1
    (tmp_path / 'crop.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Hash mismatch'):
        export(tmp_path, out)
