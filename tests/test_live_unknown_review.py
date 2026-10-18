"""Unknown UI persistence is frame evidence, not a count of callbacks."""
from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from majsoul.continual.detector import difference_hash
from majsoul.domain import Box
from majsoul.live.buttons import button_descriptor, _cosine_distance
from majsoul.live.capture import LiveFrame
from majsoul.live.config import UnknownReviewConfig
from majsoul.live.loop import LiveError
from majsoul.live.unknown_review import UnknownButtonReview
from test_live_loop import make_unknown_loop, drive_committed_frame, HAND13, PASS_BOX, UNKNOWN_CROP
from test_live_verification import snapshot, FILLER


BOX = Box(210, 120, 280, 150)


def review(tmp_path, **options):
    config = replace(UnknownReviewConfig(), **options)
    return UnknownButtonReview(config, tmp_path/'cases', tmp_path/'datasets', session='unit',
                               versions={'source': 'synthetic_unit_fixture'})


def feed(reader, index, *, crop=UNKNOWN_CROP, eligible=True, fields=None, predictions=None,
         unknowns=None, epoch=0, window=(0,0,400,200), timestamp=None):
    timestamp = index/10 if timestamp is None else timestamp
    image = np.zeros((200,400,3), dtype=np.uint8)
    image[120:150,210:280] = crop
    observation = replace(snapshot(FILLER+('1s','2s','3s'), buttons=('pass',)),
                          timestamp=timestamp, frame_index=index, **(fields or {}))
    frame = LiveFrame(image, timestamp, window, (window[0],window[1]))
    return reader.observe(frame, index, observation, 'response',
        [('kan', crop, BOX)] if unknowns is None else unknowns,
        eligible=eligible, predictions=predictions, epoch=epoch)


def cases(tmp_path):
    return sorted((tmp_path/'cases').glob('case_*/metadata.json'))


def test_four_stable_unknown_frames_create_one_pending_case_with_provenance_and_no_training(tmp_path):
    reader = review(tmp_path)
    for index in range(1,4):
        events = feed(reader,index)
        assert all(e['state']=='RESYNC' for e in events)
        assert not cases(tmp_path)
    events = feed(reader,4)
    assert events[0]['state']=='HUMAN_REVIEW' and events[0]['streak']==4
    metadata = json.loads(cases(tmp_path)[0].read_text())
    assert metadata['status']=='pending' and metadata['kind']=='action'
    assert metadata['player_count']==3 and metadata['mode']=='sanma'
    assert metadata['context']['hint_is_label'] is False
    assert [s['frame_index'] for s in metadata['context']['samples']]==[1,2,3,4]
    assert metadata['context']['previous_frame_index']==3
    assert not (tmp_path/'datasets').exists()
    # Tiny pixel changes produce different exact hashes but one visual case.
    for index in range(5,25):
        jitter = np.clip(UNKNOWN_CROP.astype(int)+index%3,0,255).astype(np.uint8)
        assert not feed(reader,index,crop=jitter)
    assert len(cases(tmp_path))==1
    assert not reader.persistence._reported  # Completed ephemeral tracks released.


@pytest.mark.parametrize('interruption',['unstable','absence','gap','clock_gap','state','epoch','window'])
def test_a_gap_or_changed_window_requires_four_new_stable_unknown_samples(tmp_path,interruption):
    reader = review(tmp_path)
    feed(reader,1)
    feed(reader,2)
    index, options = 3, {}
    if interruption=='unstable':
        feed(reader,3,eligible=False)
        index=4
    elif interruption=='absence':
        feed(reader,3,unknowns=[])
        index=4
    elif interruption=='gap':
        index=4
    elif interruption=='clock_gap':
        options={'timestamp':3.0}
    elif interruption=='state':
        options={'fields':{'remaining_tiles':49}}
    elif interruption=='epoch':
        options={'epoch':1}
    else:
        options={'window':(100,0,400,200)}
    for offset in range(4):
        current = dict(options)
        if 'timestamp' in current:
            current['timestamp']+=offset/10
        events=feed(reader,index+offset,**current)
        assert bool(cases(tmp_path)) == (offset==3)
    assert events[0]['state']=='HUMAN_REVIEW'


def test_multiple_callbacks_and_repeated_frame_do_not_inflate_persistence(tmp_path):
    reader = review(tmp_path)
    repeated=[('kan',UNKNOWN_CROP,BOX)]*5
    feed(reader,1,unknowns=repeated)
    for _ in range(10):
        assert feed(reader,1,unknowns=repeated)==[]
    for index in (2,3):
        feed(reader,index,unknowns=repeated)
        assert not cases(tmp_path)
    feed(reader,4,unknowns=repeated)
    assert len(cases(tmp_path))==1


def test_equal_perceptual_hash_does_not_merge_different_button_shapes(tmp_path):
    x=np.arange(70,dtype=np.uint8)
    first=np.repeat(np.tile(x*3,(30,1))[:,:,None],3,axis=2)
    second=np.repeat(np.tile(np.where(x<50,x,50+8*(x.astype(int)-50)).astype(np.uint8),(30,1))[:,:,None],3,axis=2)
    assert difference_hash(first)==difference_hash(second)
    assert _cosine_distance(button_descriptor(first,(48,48)),button_descriptor(second,(48,48)))>.05
    reader=review(tmp_path)
    for i in range(1,5):
        feed(reader,i,crop=first)
    for i in range(5,9):
        feed(reader,i,crop=second)
    assert len(cases(tmp_path))==2
    # Cross-process same-shape repeats use the canonical case instead.
    other=review(tmp_path)
    for i in range(1,5):
        events=feed(other,i,crop=second)
    assert not events[0]['case'].created and len(cases(tmp_path))==2


def test_uniform_animation_never_becomes_glyph_training_case(tmp_path):
    reader=review(tmp_path)
    for i in range(1,9):
        feed(reader,i,crop=np.full_like(UNKNOWN_CROP,200))
    assert not cases(tmp_path)


def test_stable_clipped_ui_routes_to_geometry_review_without_an_action_label(tmp_path):
    reader=review(tmp_path)
    for i in range(1,5):
        events=feed(reader,i,predictions={BOX:{'clipped':True}})
    assert events[0]['state']=='HUMAN_REVIEW' and events[0]['kind']=='ui_state'
    metadata=json.loads(cases(tmp_path)[0].read_text())
    assert metadata['kind']=='ui_state' and metadata['status']=='pending'
    assert metadata['context']['classification_vs_geometry']=='ROI_boundary_clipping'
    assert 'state_inconsistency' in metadata['reasons']
    assert 'label' not in metadata and not (tmp_path/'datasets').exists()


def test_case_and_candidate_limits_are_bounded(tmp_path):
    reader=review(tmp_path,maximum_cases_per_session=1)
    for i in range(1,9):
        crop=UNKNOWN_CROP if i<5 else np.random.default_rng(4).integers(0,255,(30,70,3),dtype=np.uint8)
        events=feed(reader,i,crop=crop)
    assert len(cases(tmp_path))==1 and reader.created==1
    assert events[0]['reason']=='review_case_limit'
    bounded=review(tmp_path/'bounded',maximum_candidates_per_frame=1)
    for i in range(1,9):
        events=feed(bounded,i,unknowns=[('kan',UNKNOWN_CROP,BOX),('ron',UNKNOWN_CROP,Box(10,120,80,150))])
    assert events[0]['reason']=='too_many_unknown_candidates'
    assert not cases(tmp_path/'bounded')


def test_case_storage_failure_cannot_be_reported_as_created_and_service_can_retry(tmp_path,monkeypatch):
    reader=review(tmp_path)
    create=reader.store.create
    def fail(*args,**kwargs):
        raise OSError('unit storage failure')
    monkeypatch.setattr(reader.store,'create',fail)
    for i in range(1,5):
        events=feed(reader,i)
    assert events[0]['reason']=='review_storage_failed' and not cases(tmp_path)
    monkeypatch.setattr(reader.store,'create',create)
    events=feed(reader,5)
    assert events[0]['state']=='HUMAN_REVIEW' and len(cases(tmp_path))==1


def test_live_storage_failure_stops_before_any_model_or_mouse_input(tmp_path,monkeypatch):
    loop,window,policy,notifier=make_unknown_loop(tmp_path,hand=HAND13,buttons={'pass':PASS_BOX},
        unknowns=[('kan',UNKNOWN_CROP,PASS_BOX)],execute=False)
    drive_committed_frame(loop)
    def fail(*args,**kwargs):
        raise OSError('unit storage failure')
    monkeypatch.setattr(loop.unknown_review.store,'create',fail)
    for _ in range(2):
        frame,index=loop._capture()
        loop._process_frame(frame,index)
    frame,index=loop._capture()
    with pytest.raises(LiveError,match='Cannot archive'):
        loop._process_frame(frame,index)
    assert not window.clicks and not policy.calls and not notifier.sent


@pytest.mark.parametrize('data',[
    'bad', {'maximum_visual_distance':float('nan')}, {'maximum_sample_gap_seconds':0},
    {'geometry_tolerance':.5}, {'maximum_cases_per_session':False},
    {'maximum_candidates_per_frame':0}, {'unsupported':1},
])
def test_review_config_rejects_invalid_thresholds(data):
    with pytest.raises(ValueError):
        UnknownReviewConfig.from_dict(data,Path('.'))
