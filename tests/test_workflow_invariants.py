from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import json

import cv2
import numpy as np
import pytest

from majsoul.calibration.roi_config import ROIConfig, default_layout
from majsoul.live.capture import LiveFrame, roi_for_frame
from majsoul.live.collector import CollectConfig, CollectionUI
from majsoul.live.config import LiveConfig
from majsoul.live.buttons import ActionButtonReader
from majsoul.live.lifecycle import SessionLifecycle
from majsoul.live.notifications import NotificationQueue
from majsoul.live.session import AgentSession
from majsoul.perception.meld_reader import partition_meld_tiles
from majsoul.perception.meld_reader import MeldReader
from majsoul.perception.config import VisionConfig
from majsoul.state.schemas import ObservedMeld, ObservedTile
from majsoul.state.tracker import reconcile_own_ankan
from majsoul.perception.roi_extractor import extract_roi
from majsoul.calibration.frames import read_image

ROOT = Path(__file__).resolve().parents[1]


def test_polygon_masks_scale_and_preserve_screen_coordinates():
    slots, variants, own = default_layout(3)
    roi = ROIConfig([200, 100], 3, slots, variants, own, False,
                    {'melds_left': [.1, .1, .9, .9]},
                    region_polygons={'melds_left': [[.1,.9],[.9,.9],[.5,.1]]})
    roi.validate()
    for width, height in [(200, 100), (400, 200)]:
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        crop = extract_roi(image, roi, 'melds_left')
        assert not crop.image[0, 0].any()
        assert crop.image[-2, crop.image.shape[1]//2].all()
        assert crop.box.x1 == round(.1*width)
    baked = roi_for_frame(roi, (400, 250), 'fit_width_center')
    assert baked.region_polygons['melds_left'][2] == [.5, .18]
    baked.validate()
    roi.region_polygons['melds_left'][0] = [1, 1]
    with pytest.raises(ValueError, match='polygon'):
        roi.validate()


@pytest.mark.parametrize(('labels', 'lengths'), [
    (['4s']*3+['8s']*3, [3,3]),
    (['2p']*4+['5z']*3+['6z']*3, [4,3,3]),
    (['0p','5p','5p'], [3]),
    (['2s','3s','4s'], [3]),
    (['1p','3p','5p'], None),
    (['4s']*12, None),  # Both four triplets and three quads fit: do not guess.
    (['4s','4s',None], None),
])
def test_meld_partition_requires_unique_legal_explanation(labels, lengths):
    tiles = [SimpleNamespace(prediction=SimpleNamespace(tile=label)) for label in labels]
    groups = partition_meld_tiles(tiles)
    assert (None if groups is None else list(map(len, groups))) == lengths


def test_lobby_and_silence_do_not_invent_match_completion():
    lifecycle = SessionLifecycle(1)
    for now in (0, 1, 100):
        assert 'match_complete' not in lifecycle.update(
            confirm=False, table=False, lobby=True, now=now)
    lifecycle.update(confirm=False, table=True, lobby=False, now=101)
    for now in (102, 999):
        assert 'match_complete' not in lifecycle.update(
            confirm=False, table=False, lobby=False, now=now)
    lifecycle.update(confirm=True, table=False, lobby=False, now=1000)
    lifecycle.update(confirm=False, table=False, lobby=True, now=1001)
    assert lifecycle.update(confirm=False, table=False, lobby=True, now=1002) == ('match_complete',)


def test_scorecard_is_not_confused_with_later_rank_progress():
    ui = CollectionUI(CollectConfig.load(ROOT/'configs/collect.steam.yaml'))
    audit = ROOT/'artifacts/live/workflow-audit'
    assert ui.match_result(read_image(audit/'result-scorecard.jpg'))
    assert not ui.match_result(read_image(audit/'result-progress.jpg'))
    assert not ui.match_result(read_image(ROOT/'artifacts/live/observe-collect-debug/frame.png'))
    source = read_image(audit/'scorecard-character-overlay.jpg')
    for scale in (1, .75):
        assert ui.match_result(cv2.resize(source, None, fx=scale, fy=scale))


@pytest.mark.parametrize(('filename', 'expected'), [
    ('立直.jpg', {'riichi','pass'}),
    ('拔北.jpg', {'nuki','abort','pass'}),
    ('流局.jpg', {'nuki','abort','pass'}),
    ('自摸.png', {'tsumo','pass'}),
])
def test_reviewed_user_buttons_are_recognized_at_multiple_sizes(filename, expected):
    config = LiveConfig.load(ROOT/'configs/live.steam.yaml')
    reader = ActionButtonReader(config.buttons, config.button_templates)
    source = read_image(ROOT/'datasets'/filename)
    roi = reviewed_source_button_roi(config, ROOT/'datasets'/filename)
    # Import frames have known different chrome; explicitly reviewed stretch.
    for scale in (1, .75):
        image = cv2.resize(source, None, fx=scale, fy=scale)
        projected = roi_for_frame(roi, (image.shape[1], image.shape[0]), 'stretch')
        assert expected <= set(reader.read(image, projected))
        assert not any(candidate.clipped for candidate in reader.last_candidates)


def reviewed_source_button_roi(config, source):
    # Imported desktop sources use their already reviewed complete-icon ROI,
    # recorded when templates were exported. Production Steam ROI is unchanged.
    records=json.loads((ROOT/'artifacts/templates/buttons/full-icons-20261009-manifest.json').read_text(encoding='utf-8'))
    records=[r for r in records if ROOT/r['source']==source]
    assert records and all(r['source_sha256']==sha256(source.read_bytes()).hexdigest() for r in records)
    assert all(r['projection']=='reviewed_stretch' and not r['clipped'] for r in records)
    assert all(r['source_roi']==records[0]['source_roi'] for r in records)
    roi=ROIConfig.load(config.roi)
    regions=dict(roi.regions)
    regions['action_buttons']=records[0]['source_roi']
    return replace(roi,regions=regions)


def test_ron_and_current_theme_pon_from_user_frame():
    config = LiveConfig.load(ROOT/'configs/live.steam.yaml')
    reader = ActionButtonReader(config.buttons, config.button_templates)
    source = read_image(ROOT/'和.png')
    roi = reviewed_source_button_roi(config, ROOT/'和.png')
    for scale in (1, .75):
        image = cv2.resize(source, None, fx=scale, fy=scale)
        projected = roi_for_frame(roi, (image.shape[1], image.shape[0]), 'stretch')
        assert {'ron','pon','pass'} <= set(reader.read(image, projected))
        assert not any(candidate.clipped for candidate in reader.last_candidates)


def test_closed_quad_requires_back_pattern_and_matching_visible_faces():
    def tile(label, back=False):
        return SimpleNamespace(prediction=SimpleNamespace(tile=label),
                               detection=SimpleNamespace(face_down=back))
    good = [tile(None, True), tile('9m'), tile('9m'), tile(None, True)]
    assert partition_meld_tiles(good) == (tuple(good),)
    assert partition_meld_tiles([tile(None, True), tile('9m'), tile('1m'), tile(None, True)]) is None
    assert partition_meld_tiles([tile(None, True), tile(None, True), tile('9m'), tile('9m')]) is None
    assert partition_meld_tiles([tile(None, True)]*4) is None


def test_own_ankan_keeps_red_identity_only_with_hand_difference_proof():
    observation = ObservedMeld((ObservedTile(None, face_down=True),
                               ObservedTile('5p'), ObservedTile('5p'),
                               ObservedTile(None, face_down=True)), 'ankan')
    resolved = reconcile_own_ankan(observation, None,
                                  ['0p','5p','5p','5p','1s'], ['1s','2s'],
                                  before_frame=10, after_frame=11)
    assert sorted(resolved.tiles) == ['0p','5p','5p','5p']
    assert resolved.opened is False
    assert resolved.evidence_frames == (10,11)
    assert resolved.face_down == (True,False,False,True)
    # An unrelated removal or a missing fourth tile is not historical proof.
    uncertain = reconcile_own_ankan(observation, None,
                                   ['5p','5p','5p','1s','2s'], ['2s'],
                                   before_frame=10, after_frame=11)
    assert uncertain.tiles == (None,'5p','5p',None)
    carried = reconcile_own_ankan(observation, resolved, ['1s'], ['1s','3s'],
                                 before_frame=11, after_frame=12)
    assert carried.tiles == resolved.tiles and carried.evidence_frames == (10,11)


def test_real_ankan_back_faces_bypass_gold_cnn_and_scale():
    torch = pytest.importorskip('torch')
    from majsoul.live.classifier import build_tile_classifier
    config = LiveConfig.load(ROOT/'configs/live.steam.yaml')
    vision = VisionConfig.load(config.vision)
    roi = ROIConfig.load(config.roi)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        classifier = build_tile_classifier('cnn', vision, cnn_checkpoint=config.cnn_checkpoint)
        calls = []
        def classify(faces):
            calls.extend(faces)
            assert all(not face.face_down for face in faces)
            return classifier.classify_many(faces)
        reader = MeldReader(vision, SimpleNamespace(classify_many=classify), 'bottom')
        source = read_image(ROOT/'datasets/暗杠.jpg')
        for scale in (1, .75):
            image = cv2.resize(source, None, fx=scale, fy=scale)
            result = reader.read(extract_roi(image, roi, 'melds_bottom'))
            assert result.kinds == ('ankan','pon')
            assert result.opened == (False,True)
            assert result.labels[0] == (None,'9m','9m',None)
            assert result.complete
        assert len(calls) == 10
    finally:
        torch.set_num_threads(original_threads)


def test_notification_queue_does_not_block_play_or_claim_delivery(tmp_path):
    entered, release = Event(), Event()
    def send(*args):
        entered.set()
        release.wait(3)
        return True
    queue = NotificationQueue(SimpleNamespace(notify_text=send), tmp_path, capacity=1)
    first = queue.submit('notify_text', 'one')
    assert entered.wait(1)
    second = queue.submit('notify_text', 'two')
    third = queue.submit('notify_text', 'three')
    assert not first.done() and not second.done()
    assert third.result() is False
    statuses = [json.loads(p.read_text())['status'] for p in tmp_path.glob('*.json')]
    assert 'delivered' not in statuses and 'deferred' in statuses
    assert queue.close(.01) == 2
    release.set()
    assert first.result(timeout=2) and second.result(timeout=2)
    assert queue.close(1) == 0


def test_notification_retries_are_bounded_and_ledger_tracks_actual_delivery(tmp_path):
    calls = []
    def send(*args):
        calls.append(args)
        return len(calls) == 3
    queue = NotificationQueue(SimpleNamespace(notify_text=send), tmp_path,
                              max_attempts=3, retry_seconds=.01)
    assert queue.submit('notify_text', 'retry').result(timeout=2)
    assert len(calls) == 3
    receipt = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert receipt['status'] == 'delivered' and receipt['attempts'] == 3
    assert queue.close(1) == 0


def test_notification_partial_delivery_retries_image_without_duplicate_text(tmp_path):
    import subprocess
    from majsoul.live.notify import FeishuNotifier
    from majsoul.live.config import NotifyConfig
    modes = []
    def runner(command, **kwargs):
        mode = command[2]
        modes.append(mode)
        fail = mode == 'image' and modes.count('image') == 1
        return subprocess.CompletedProcess(command, int(fail),
                                           stdout=json.dumps({'ready': True}), stderr='TLS' if fail else '')
    image = tmp_path/'result.png'
    cv2.imwrite(str(image), np.zeros((20,20,3),dtype=np.uint8))
    notifier = FeishuNotifier(NotifyConfig(True, Path('helper.mjs'), 'test'), runner=runner)
    assert not notifier.notify_text('final result', image)
    assert notifier.notify_text('final result', image)
    assert modes == ['check-config','text','image','check-config','image']
    assert not notifier.notify_text('missing result', tmp_path/'missing.png')


@pytest.mark.parametrize('scale', [1, .75])
def test_reviewed_steam_south_seat_glyph_in_ankan_source(scale):
    from majsoul.perception import GlyphTemplateClassifier, VisionConfig
    from majsoul.perception.metadata_reader import GameMetadataReader
    from majsoul.perception.roi_extractor import rotate_roi
    vision = VisionConfig.load(ROOT/'configs/vision.steam.yaml')
    roi = ROIConfig.load(ROOT/'configs/roi.steam.yaml')
    reader = GameMetadataReader(vision, GlyphTemplateClassifier(
        ROOT/'artifacts/templates/glyphs', vision), read_scores=False)
    source = read_image(ROOT/'datasets/暗杠.jpg')
    image = cv2.resize(source, None, fx=scale, fy=scale)
    crop = rotate_roi(extract_roi(image, roi, 'seat_left'), vision.rotation_for_slot('left'))
    reading = reader._read_seat(crop, 'left')
    assert reading.wind == '2z'
    assert reading.dealer is False
    assert reading.issues == ()


@pytest.mark.parametrize('scale', [1, .75])
def test_side_meld_bounds_keep_called_face_and_exclude_hidden_hand_edges(scale):
    from majsoul.perception.tile_segmenter import MeldSegmenter
    from majsoul.perception.roi_extractor import rotate_roi
    vision = VisionConfig.load(ROOT/'configs/vision.steam.yaml')
    roi = ROIConfig.load(ROOT/'configs/roi.steam.yaml')
    positive = read_image(ROOT/'artifacts/live/workflow-audit/steam-right-called-pon.jpg')
    for source, expected in [(positive, 3), (read_image(ROOT/'datasets/暗杠.jpg'), 0)]:
        image = cv2.resize(source, None, fx=scale, fy=scale)
        crop = rotate_roi(extract_roi(image, roi, 'melds_right'), vision.rotation_for_slot('right'))
        result = MeldSegmenter(vision, vision.meld_for_slot('right')).segment(crop)
        assert len(result.tiles) == expected
        assert result.issues == ()


@pytest.mark.parametrize('execute', [False, True])
def test_agent_saves_scorecard_before_confirm_and_dry_run_sends_no_input(tmp_path, execute):
    config = CollectConfig.load(ROOT/'configs/collect.steam.yaml')
    frame = LiveFrame(np.zeros((200,400,3),dtype=np.uint8), 0, (0,0,400,200),(0,0))
    clicks = []
    def click(*args, **kwargs):
        assert (tmp_path/'match_result.png').is_file()
        clicks.append(args)
    window = SimpleNamespace(click=click)
    session = AgentSession(window, config, execute=execute, capture=lambda: frame,
                           sleep=lambda _: None, log=lambda *args, **kwargs: None,
                           directory=tmp_path, notify=lambda *args, **kwargs: True)
    session.classify = lambda f: (False, 'gold', False)
    session.ui = SimpleNamespace(match_result=lambda im: True, confirm=lambda im: 'gold')
    assert session.observe(frame, 0)[0]
    session.observe(frame, config.table_ready_seconds)
    assert len(clicks) == int(execute)
    assert (tmp_path/'match_result.png').is_file()
