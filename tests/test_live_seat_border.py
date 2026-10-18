from pathlib import Path
import numpy as np

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.live.config import LiveConfig
from majsoul.perception import GameMetadataReader,GlyphTemplateClassifier,SeatLayoutResolver,VisionConfig
from majsoul.perception.roi_extractor import ROICrop
from majsoul.domain import Box

ROOT=Path(__file__).resolve().parents[1]


def test_real_remaining_six_is_not_read_as_eight():
    c=LiveConfig.load(ROOT/'configs/live.steam.yaml')
    vision=VisionConfig.load(c.vision)
    reader=GameMetadataReader(vision,GlyphTemplateClassifier(c.glyph_templates,vision),read_scores=False)
    roi=ROIConfig.load(c.roi)
    image=read_image(ROOT/'artifacts/live/acceptance-20261010/agent-recheck-131424/diagnostics/000096.jpg')
    reading=reader.read(image,roi,('bottom','top','left'))
    assert reading.remaining_tiles.value==46 and reading.remaining_tiles.complete


def test_real_right_west_retains_glyph_and_excludes_touching_frame_border():
    c=LiveConfig.load(ROOT/'configs/live.steam.yaml')
    vision=VisionConfig.load(c.vision)
    reader=GameMetadataReader(vision,GlyphTemplateClassifier(c.glyph_templates,vision),read_scores=False)
    roi=ROIConfig.load(c.roi)
    image=read_image(ROOT/'artifacts/live/acceptance-20261010/agent-recheck-115437/check-01.png')
    metadata=reader.read(image,roi,('bottom','top','right'))
    assert metadata.complete
    assert {s.wind for s in metadata.seats.values()}=={'1z','2z','3z'}
    assert metadata.seats['right'].wind=='3z'
    assert all(s.value is None for s in metadata.scores.values())
    resolver=SeatLayoutResolver(roi,reader,starting_points=35000,confirmations_required=2)
    assert resolver.observe(image).candidates==(('bottom','top','right'),)
    assert resolver.observe(image).locked_layout==('bottom','top','right')


def test_seat_frame_border_alone_cannot_supply_a_wind_glyph():
    c=LiveConfig.load(ROOT/'configs/live.steam.yaml')
    vision=VisionConfig.load(c.vision)
    reader=GameMetadataReader(vision,GlyphTemplateClassifier(c.glyph_templates,vision),read_scores=False)
    pixels=np.zeros((80,100,3),np.uint8)
    pixels[:,0:5]=255
    pixels[0:5,:]=255
    crop=ROICrop('seat_right',Box(0,0,100,80),pixels)
    reading=reader._read_seat(crop,'right')
    assert reading.wind is None
    assert 'Seat wind glyph is unknown' in reading.issues


def test_real_steam_five_honba_locks_layout_without_scores():
    c=LiveConfig.load(ROOT/'configs/live.steam.yaml')
    vision=VisionConfig.load(c.vision)
    reader=GameMetadataReader(vision,GlyphTemplateClassifier(c.glyph_templates,vision),read_scores=False)
    roi=ROIConfig.load(c.roi)
    image=read_image(ROOT/'artifacts/live/acceptance-20261010/agent-recheck-123649/afk-after.png')
    reading=reader.read(image,roi,('bottom','top','left'))
    assert reading.complete and reading.honba.value==5
    assert reading.round_id=='1z:2:5'
    assert all(score.value is None for score in reading.scores.values())
    resolver=SeatLayoutResolver(roi,reader,starting_points=35000,confirmations_required=2)
    assert resolver.observe(image).candidates==(('bottom','top','left'),)
    assert resolver.observe(image).locked_layout==('bottom','top','left')
