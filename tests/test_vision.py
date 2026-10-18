from dataclasses import replace

from majsoul.domain import Box, Detection
from majsoul.tiles import parse_hand
from majsoul.vision import Layout, ObservationBuilder, REQUIRED_REGIONS, ordered


def synthetic_layout():
    # Synthetic grid is test data, never a production window layout.
    regions = {}
    for index, name in enumerate(REQUIRED_REGIONS):
        row, col = divmod(index, 3)
        regions[name] = {"rect": [col/3, row/5, (col+1)/3, (row+1)/5]}
    return Layout(regions, {})


def synthetic_detections():
    hand = parse_hand("19m123456p123s111z")
    ds = [Detection("tile:"+t, .99, Box(2+i*24, 10, 20+i*24, 50)) for i, t in enumerate(hand[:-1])]
    ds.append(Detection("tile:"+hand[-1], .99, Box(350,10,370,50)))
    ds.append(Detection("tile:9s", .99, Box(700,10,720,50)))
    for index, label in enumerate(("phase:discard", "meta:seat:1z", "meta:round:1z", "meta:number:1", "meta:honba:0")):
        ds.append(Detection(label, .99, Box(20+index*100, 850, 80+index*100, 890)))
    return ds


def test_observation_complete_hand(engine):
    builder = ObservationBuilder(synthetic_layout(), engine.rules)
    obs = builder.build(synthetic_detections(), 1.0, (0,0,1000,1000))
    assert not obs.issues
    assert len(obs.hand) == 14
    assert obs.draw_index == 13
    assert obs.round_id == "1z:1:0"


def test_missing_low_confidence_duplicate_and_metadata_fail_closed(engine):
    builder = ObservationBuilder(synthetic_layout(), engine.rules)
    detections = synthetic_detections()
    low = [replace(d, confidence=.2) if i == 0 else d for i,d in enumerate(detections)]
    # Two missing tiles force an invalid hand count (13 can be a valid waiting hand).
    for bad in (detections[2:], low, detections + [detections[0]], detections[:-1]):
        assert builder.build(bad, 1.0, (0,0,1000,1000)).issues


def test_order_rows_tolerates_box_jitter():
    ds = [Detection("a", 1, Box(20,11,30,31)), Detection("b", 1, Box(1,10,11,30)), Detection("c", 1, Box(1,40,11,60))]
    assert [d.label for d in ordered(ds, {})] == ["b", "a", "c"]
