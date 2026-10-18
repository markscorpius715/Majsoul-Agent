from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.live.capture import FocusUnavailable, LiveFrame
from majsoul.live.collector import CollectConfig
from majsoul.live.keepalive import LobbyGuard
from test_live_loop import ROOT


@pytest.fixture
def guard():
    return LobbyGuard(read_image(ROOT / "artifacts/live/restored-observe-20261008/frame.png"),
                       (.51, .21), CollectConfig.load(ROOT / "configs/collect.steam.yaml"),
                       ROIConfig.load(ROOT / "configs/roi.steam.yaml"), "auto")


def test_keepalive_rejects_real_table_popup_and_wrong_aspect(guard):
    assert guard.accepts(guard.reference)
    assert guard.accepts(cv2.resize(guard.reference, None, fx=.75, fy=.75))
    for path in ("datasets/暗杠.jpg", "artifacts/live/countdown-readonly-observe-20261008/frame.png"):
        assert not guard.accepts(read_image(ROOT / path))
    assert not guard.accepts(cv2.resize(guard.reference, (100, 100)))
    obstructed = guard.reference.copy()
    height, width = obstructed.shape[:2]
    obstructed[round(.19*height):round(.23*height),round(.49*width):round(.53*width)] = 0
    assert not guard.accepts(obstructed)


@pytest.mark.parametrize("change", ["none", "geometry", "overlay", "focus"])
def test_keepalive_requires_two_fresh_matching_frames_and_verified_focus(guard, change):
    image = guard.reference
    frame = LiveFrame(image, 1, (0, 0, image.shape[1], image.shape[0]), (0, 0))
    fresh = replace(frame, captured_at=1.1)
    if change == "geometry":
        fresh = replace(fresh, window=(1, 0, image.shape[1], image.shape[0]))
    elif change == "overlay":
        fresh = replace(fresh, image=np.zeros_like(image))
    frames, clicks = iter((frame, fresh)), []
    def click(point, *, expected):
        assert expected is fresh
        if change == "focus":
            raise FocusUnavailable("no input was sent")
        clicks.append(point)
    window = SimpleNamespace(capture=lambda: next(frames), click=click)
    outcome, _ = guard.pulse(window, sleep=lambda _: None)
    if change == "none":
        assert outcome == "input_sent" and len(clicks) == 1
    else:
        assert outcome.startswith("skipped_") and clicks == []
