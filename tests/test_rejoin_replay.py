"""Offline replay of a real takeover: the rejoin Agent must become ready again."""
from pathlib import Path

import pytest

from tools.rejoin_replay import ReplayWindow, replay

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'artifacts/live/protected-20261018-151224'


def test_rejoin_agent_is_ready_on_opponent_turns_beside_two_north_tiles(tmp_path):
    """Needs the live readers (rapidocr); run with the live environment.

Live, readiness stopped once our second North tile entered the button
    ROI: its clipped edge was an unknown button candidate on every frame."""
    pytest.importorskip('rapidocr')
    report = replay(RUN/'client-002', tmp_path, start=9487.6, end=9499.5)
    assert report['ready_signals'] > 0
    assert not any('unknown_buttons' in change['blockers'] for change in report['readiness_changes'])


def test_replay_window_refuses_input():
    window = ReplayWindow([], (0, 0, 10, 10))
    with pytest.raises(AssertionError):
        window.click((1, 1))
    with pytest.raises(KeyboardInterrupt):
        window.capture()
