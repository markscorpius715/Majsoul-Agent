"""Real Steam failure: button glyphs must not influence the river grid."""
from pathlib import Path

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.domain import Box
from majsoul.live.buttons import ActionButtonReader
from majsoul.live.config import LiveConfig
from majsoul.perception import RiverReader, RiverSegmenter, VisionConfig, extract_roi

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/live/acceptance-20261010/114539/agent/diagnostics/000076.jpg'


def test_real_button_components_are_excluded_before_river_anomaly_checks(gold_tile_classifier):
    image=read_image(SOURCE)
    config=LiveConfig.load(ROOT/'configs/live.steam.yaml')
    roi=ROIConfig.load(config.roi)
    vision=VisionConfig.load(config.vision)
    buttons=ActionButtonReader(config.buttons,config.button_templates)
    assert set(buttons.read(image,roi))=={'nuki','pass'}
    crop=extract_roi(image,roi,'river_bottom')
    segmenter=RiverSegmenter(vision)
    assert 'Off-grid river components ignored' in segmenter.segment(crop).issues
    # Merely having an unrelated UI box must not silence this failure.
    assert 'Off-grid river components ignored' in segmenter.segment(
        crop,excluded_boxes=(Box(0,0,10,10),)).issues
    filtered=segmenter.segment(crop,excluded_boxes=buttons.last_occlusion_boxes)
    assert not filtered.issues
    reading=RiverReader(vision,gold_tile_classifier).read(
        crop,excluded_boxes=buttons.last_occlusion_boxes)
    # Enlarged source crop shows two black + four red circles (six circles).
    assert reading.labels==('7z','3s','6p')
    assert reading.issues==('UI occlusion; region unavailable',)
    assert not reading.complete  # Hidden river suffix still isn't known.
