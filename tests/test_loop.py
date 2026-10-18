from dataclasses import replace
import time
import sys

from PIL import Image
import pytest

from majsoul.controller import AgentLoop, ControlError
from majsoul.strategy import DeepSeekPolicy
from majsoul.domain import Box, Detection
from majsoul.vision import ObservationBuilder
from majsoul.window import Frame
from test_vision import synthetic_detections, synthetic_layout


@pytest.fixture(autouse=True)
def deterministic_capture_clock(monkeypatch):
    # Windows Python 3.11 can return the same monotonic value for several fast
    # fake captures. Advance a virtual clock instead of depending on OS ticks.
    class Clock:
        value = 100.

        def monotonic(self):
            self.value += .001
            return self.value

        def sleep(self, seconds):
            self.value += seconds
    clock = Clock()
    monkeypatch.setattr(sys.modules[__name__], 'time', clock)
    monkeypatch.setattr('majsoul.controller.time', clock)


class FakeWindow:
    def __init__(self):
        self.clicks = []

    def capture(self):
        return Frame(Image.new("RGB", (1000,1000)), time.monotonic(), (0,0,1000,1000))

    def click(self, point, expected):
        self.clicks.append((point, expected))


class Detector:
    def __init__(self, window):
        self.window = window

    def detect(self, image):
        detections = synthetic_detections()
        if not self.window.clicks:
            return detections
        after = []
        for i,d in enumerate(detections):
            if i == 0:
                continue
            if i == 13:  # Drawn tile joins concealed row after discard.
                d = replace(d, box=Box(314,10,332,50))
            if d.label == "phase:discard":
                d = replace(d, label="phase:waiting")
            after.append(d)
        return after + [Detection("tile:1m", .99, Box(10,220,30,260))]


class FirstDiscardPolicy:
    def choose(self, state, actions, *, deadline=None):
        return next(a for a in actions if a.kind == "discard")


def test_full_detection_state_choice_click_verification(engine, tmp_path):
    window = FakeWindow()
    layout = replace(synthetic_layout(), poll_seconds=.001)
    agent = AgentLoop(window, Detector(window), ObservationBuilder(layout, engine.rules),
                      engine, FirstDiscardPolicy(), layout, tmp_path)
    assert agent.step(execute=True) is None  # First unstable frame cannot click.
    action = agent.step(execute=True)
    assert action.tile == "1m"
    assert len(window.clicks) == 1
    assert (tmp_path / "00001-after.png").exists()
    assert agent.step(execute=True) is None
    assert len(window.clicks) == 1


def test_dry_run_never_clicks_and_suppresses_repeat(engine, tmp_path):
    window = FakeWindow()
    layout = synthetic_layout()
    agent = AgentLoop(window, Detector(window), ObservationBuilder(layout, engine.rules),
                      engine, FirstDiscardPolicy(), layout, tmp_path)
    agent.step()
    assert agent.step()
    assert agent.step() is None
    assert not window.clicks


def test_expired_base_turn_does_not_spend_unobserved_reserve(engine, tmp_path):
    window = FakeWindow()
    layout = synthetic_layout()
    agent = AgentLoop(window, Detector(window), ObservationBuilder(layout, engine.rules),
                      engine, FirstDiscardPolicy(), layout, tmp_path)
    agent.step()
    agent.actionable_since = time.monotonic() - engine.rules.data["turn_seconds"]
    with pytest.raises(ControlError, match="No model budget"):
        agent.step(execute=True)
    assert not window.clicks


def test_live_model_gets_remaining_base_budget(engine, tmp_path):
    class Model(DeepSeekPolicy):
        def __init__(self):
            self.deadline = None
            self.last_timing = {"status": "ok"}

        def choose(self, state, actions, *, deadline=None, stream=True):
            self.deadline = deadline
            return actions[0]

    window, model = FakeWindow(), Model()
    layout = synthetic_layout()
    agent = AgentLoop(window, Detector(window), ObservationBuilder(layout, engine.rules),
                      engine, model, layout, tmp_path)
    agent.step()
    origin = agent.actionable_since
    agent.step()
    assert model.deadline == origin + engine.rules.data["turn_seconds"] - layout.action_margin_seconds
    assert not window.clicks
