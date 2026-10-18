from pathlib import Path

import pytest

from majsoul.domain import Box, Observation, State, TileView
from majsoul.rules import RuleEngine, Rules
from majsoul.tiles import parse_hand


@pytest.fixture(scope="session")
def gold_tile_classifier():
    """Production gold weights with the historical video's segmentation config."""
    import torch
    from majsoul.live.classifier import build_tile_classifier
    from majsoul.live.config import LiveConfig
    from majsoul.perception import VisionConfig
    root = Path(__file__).resolve().parents[1]
    live = LiveConfig.load(root / "configs/live.steam.yaml")
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        yield build_tile_classifier("cnn", VisionConfig.load(root / "configs/vision.yaml"),
                                    cnn_checkpoint=live.cnn_checkpoint,
                                    expected_metadata=live.cnn_expected)
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def engine():
    return RuleEngine(Rules.load(Path(__file__).resolve().parents[1] / "configs/sanma.json"))


def observation(hand="19m123456p123s111z", *, time=1.0, **kwargs):
    defaults = dict(draw_index=13, phase="discard", round_id="1z:1:0", seat_wind="1z", round_wind="1z", dora_indicators=("9s",))
    defaults.update(kwargs)
    return Observation(time, (100, 100, 1000, 800),
                       hand=tuple(TileView(t, Box(20+i*40, 600, 50+i*40, 660)) for i, t in enumerate(parse_hand(hand))),
                       **defaults)


@pytest.fixture
def state():
    return State(observation(), True, ((), (), ()))
