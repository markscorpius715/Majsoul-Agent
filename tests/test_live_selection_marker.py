"""Shared real cancel glyph plus input-bound all-bright selection negatives."""
from dataclasses import replace
import json

import cv2
import pytest

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.live.capture import LiveFrame
from majsoul.live.config import LiveConfig
from majsoul.live.selection import RiichiSelection
from majsoul.live.selection_marker import SelectionCancelMarker
from test_live_selection import baseline, capture, SOURCE, ROOT
from test_live_loop import small_roi


def config():
    return LiveConfig.load(ROOT/'configs/live.steam.yaml').riichi_selection


@pytest.mark.parametrize('index,expected', [(106440,False),(106448,True),(106450,True),(106458,True)])
@pytest.mark.parametrize('size', [(3070,1822),(2582,1550),(1535,911)])
def test_native_cancel_marker_is_distinct_from_the_preclick_window(index, expected, size):
    image = read_image(SOURCE/f'{index:07d}.png')
    image = cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    frame = LiveFrame(image,index/30,(0,0,*size),(0,0))
    # Historical/native source projection, not a production geometry override.
    roi = replace(ROIConfig.load(ROOT/'configs/roi.steam.yaml'),reference_size=list(size))
    result = SelectionCancelMarker(config()).read(frame,roi)
    assert result['accepted'] is expected, result
    assert result['region'] == 'countdown'


@pytest.mark.parametrize('path', ['artifacts/live/resume-preflight-20261010/frame.png',
    'artifacts/live/workflow-audit/selection-source-20261009/0049230.jpg',
    'artifacts/live/workflow-audit/kan-source-20261009/0118560.jpg'])
def test_real_lobby_and_kan_buttons_are_not_selection_cancel_markers(path):
    image=read_image(ROOT/path)
    h,w=image.shape[:2]
    roi=replace(ROIConfig.load(ROOT/'configs/roi.steam.yaml'),reference_size=[w,h])
    result=SelectionCancelMarker(config()).read(LiveFrame(image,1.,(0,0,w,h),(0,0)),roi)
    assert not result['accepted'],result


def fixture():
    _,before,visible,after=baseline()
    roi=replace(small_roi(),reference_size=[1000,800],regions={'countdown':[.73,.67,.90,.815]})
    service=RiichiSelection(config())
    frame=capture(before,visible)
    service.begin(frame,before,visible,16.,roi=roi)
    target=capture(after,visible)
    glyph=cv2.resize(service.cancel_marker.template,(30,41),interpolation=cv2.INTER_AREA)
    return service,before,visible,after,roi,target,glyph


def test_all_bright_hand_needs_a_new_real_cancel_marker_and_sent_intent():
    service,_,visible,after,roi,frame,glyph=fixture()
    assert service.read(frame,after,visible,roi=roi)[0] is None
    frame.image[600:641,760:790]=glyph
    indices,evidence=service.read(frame,after,visible,roi=roi)
    assert indices==tuple(range(14))
    assert evidence['source']=='client_cancel_marker_and_undimmed_hand'
    assert evidence['disabled_indices']==[] and evidence['cancel_marker']['accepted']
    assert RiichiSelection(config()).read(frame,after,visible,roi=roi)[0] is None
    assert service.read(frame,after,visible)[0] is None
    assert service.read(frame,replace(after,issues=('hand:unknown',)),visible,roi=roi)[0] is None
    assert service.read(frame,replace(after,timestamp=16.),visible,roi=roi)[0] is None


def test_duplicate_clipped_or_preexisting_cancel_markers_do_not_authorize_a_hand():
    service,before,visible,after,roi,frame,glyph=fixture()
    frame.image[600:641,750:780]=glyph
    frame.image[550:591,830:860]=glyph
    assert service.read(frame,after,visible,roi=roi)[0] is None
    with pytest.raises(ValueError,match='baseline already'):
        service.begin(replace(frame,captured_at=before.timestamp),before,visible,16.,roi=roi)
    clipped=capture(after,visible)
    clipped.image[600:641,730:760]=glyph
    assert service.read(clipped,after,visible,roi=roi)[0] is None


def test_changed_marker_assets_fail_closed(tmp_path):
    marker=SelectionCancelMarker(config())
    data=dict(marker.manifest)
    data['template']=str(ROOT/data['template'])
    data['template_sha256']='0'*64
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='hash changed'):
        SelectionCancelMarker(replace(config(),cancel_marker_manifest=str(path)))
    data['template_sha256']=marker.manifest['template_sha256']
    data['crop_box']=[0,0,1,1]
    path.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='dimensions changed'):
        SelectionCancelMarker(replace(config(),cancel_marker_manifest=str(path)))
