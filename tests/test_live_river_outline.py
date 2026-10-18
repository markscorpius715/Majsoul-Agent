"""Steam highlight must not absorb a known face into an oversized component."""
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
import pytest

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.domain import Box
from majsoul.live.capture import roi_for_frame
from majsoul.perception import RiverReader,RiverSegmenter,VisionConfig,extract_roi
from majsoul.perception.roi_extractor import ROICrop,rotate_roi

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/live/acceptance-20261010/agent-recheck-122139/last-frame.png'


@pytest.mark.parametrize('scale',[1.,.75])
def test_real_highlight_separates_from_original_face_pixels(gold_tile_classifier,scale):
    image=read_image(SOURCE)
    image=cv2.resize(image,None,fx=scale,fy=scale)
    roi=roi_for_frame(ROIConfig.load(ROOT/'configs/roi.steam.yaml'),image.shape[1::-1],'auto')
    vision=VisionConfig.load(ROOT/'configs/vision.steam.yaml')
    crop=extract_roi(image,roi,'river_left')
    rotated=rotate_roi(crop,-90)
    assert 'Oversized white component may be animation or occlusion' in RiverSegmenter(
        replace(vision,river_outline_opening_fraction=0)).segment(rotated).issues
    segmented=RiverSegmenter(vision).segment(rotated)
    assert not segmented.issues and len(segmented.tiles)==1
    # The classifier receives exact source pixels, never an opened/eroded glyph.
    detected=segmented.tiles[0]
    assert detected.image.shape[0]>detected.image.shape[1]
    reading=RiverReader(vision,gold_tile_classifier).read(crop,-90)
    assert reading.complete and reading.labels==('1m',)


def test_solid_oversized_occlusion_is_not_silenced_by_outline_recovery():
    vision=VisionConfig.load(ROOT/'configs/vision.steam.yaml')
    image=np.zeros((420,353,3),np.uint8)
    image[30:150,30:200]=255
    crop=ROICrop('river_left',Box(0,0,353,420),image)
    reading=RiverSegmenter(vision).segment(crop)
    assert not reading.tiles
    assert 'Oversized white component may be animation or occlusion' in reading.issues
