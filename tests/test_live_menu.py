from collections import namedtuple
import time

import numpy as np
import pytest

from majsoul.live.menu import (MenuConfig, MenuNavigator, MenuStep,
                               NavigationError, calibrate_menu)
from majsoul.window import WindowError
from majsoul.live.capture import FocusUnavailable

FakeFrame = namedtuple("FakeFrame", ["image", "captured_at", "window"])


class FakeWindow:
    def __init__(self, size=(2560, 1600), fail_step=None, fail_on="click"):
        self.size = size
        self.clicks = []
        self.click_options = []
        self.captures = 0
        self.fail_step = fail_step
        self.fail_on = fail_on

    def capture(self):
        if self.fail_on == "capture" and self.captures == self.fail_step:
            raise WindowError("window gone")
        self.captures += 1
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        return FakeFrame(image, time.monotonic(), (0, 0, *self.size))

    def click(self, point, **kwargs):
        if self.fail_on == "click" and len(self.clicks) == self.fail_step:
            raise WindowError("click rejected")
        self.clicks.append(point)
        self.click_options.append(kwargs)


def make_config(*clicks, settles=None):
    settles = settles or [0.0] * len(clicks)
    steps = tuple(
        MenuStep(name=f"step{i}", click=click, settle_seconds=settle)
        for i, (click, settle) in enumerate(zip(clicks, settles))
    )
    return MenuConfig(steps=steps)


def write_config(tmp_path, body):
    path = tmp_path / "menu.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_run_clicks_steps_in_order_with_scaled_pixels():
    config = make_config((0.5, 0.25), (0.1, 0.9), settles=[1.5, 2.0])
    window = FakeWindow()
    sleeps = []
    MenuNavigator(window, config, sleep=sleeps.append).run()
    assert window.captures == 2
    assert window.clicks == [(1280, 400), (256, 1440)]
    assert sleeps == [1.5, 2.0]


def test_run_rejects_uncalibrated_config():
    config = make_config(None, (0.5, 0.5))
    assert not config.ready
    with pytest.raises(NavigationError, match="calibrate-menu"):
        MenuNavigator(FakeWindow(), config, sleep=lambda _: None).run()


def test_long_menu_settle_keeps_capturing_without_shortening_wait_or_reclicking():
    clock = [0.0]
    captures = []
    class Window(FakeWindow):
        def capture(self):
            captures.append(clock[0])
            return super().capture()
    def sleep(seconds):
        clock[0] += seconds
    window = Window()
    MenuNavigator(window, make_config((.5,.5), (.2,.2), settles=[3, 2]),
                  sleep=sleep, now=lambda: clock[0], poll_seconds=.25).run()
    assert clock[0] == 5
    assert window.clicks == [(1280,800), (512,320)]
    assert len(captures) > 2
    assert max(b-a for a,b in zip(captures,captures[1:])) <= .25


def test_capture_failure_during_menu_settle_stops_before_next_click():
    window = FakeWindow(fail_on='capture', fail_step=1)
    with pytest.raises(NavigationError, match='step0'):
        MenuNavigator(window, make_config((.5,.5), (.2,.2), settles=[3, 2]),
                      poll_seconds=.25).run()
    assert len(window.clicks) == 1


def test_menu_reobserves_and_retries_only_a_focus_failure_with_no_input_sent():
    class Window(FakeWindow):
        attempts=0
        observed=[]
        def click(self,point,**kwargs):
            self.attempts+=1
            self.observed.append(kwargs['expected'])
            if self.attempts==1:raise FocusUnavailable('No input sent')
            super().click(point,**kwargs)
    window=Window()
    MenuNavigator(window,make_config((.5,.5)),sleep=lambda _:None).run()
    assert window.attempts==2 and len(window.clicks)==1
    assert window.observed[0] is not window.observed[1]


def test_repeated_focus_failure_is_bounded_and_never_advances_to_next_step():
    class Window(FakeWindow):
        attempts=0
        def click(self,point,**kwargs):
            self.attempts+=1
            raise FocusUnavailable('No input sent')
    window=Window()
    config=make_config((.5,.5),(.2,.2))
    with pytest.raises(NavigationError,match='step0'):
        MenuNavigator(window,config,sleep=lambda _:None).run()
    assert window.attempts==config.focus_attempts and not window.clicks


def test_menu_configured_press_duration_reaches_window(tmp_path):
    config = MenuConfig.load(write_config(tmp_path, """
steps:
- name: ranked
  click: [0.5, 0.5]
  mouse_press_seconds: 0.15
"""))
    window = FakeWindow()
    MenuNavigator(window, config, sleep=lambda _: None).run()
    assert window.click_options[0]['hold_seconds'] == 0.15
    assert window.click_options[0]['expected'].window == (0, 0, 2560, 1600)


@pytest.mark.parametrize('duration', ['0', '-0.1', '0.21', '.nan', '.inf', 'true', 'invalid'])
def test_menu_rejects_unsafe_press_duration(tmp_path, duration):
    body = f'steps:\n- name: a\n  click: [0.5, 0.5]\n  mouse_press_seconds: {duration}\n'
    with pytest.raises(ValueError):
        MenuConfig.load(write_config(tmp_path, body))


def test_run_wraps_window_error_with_step_name():
    config = make_config((0.5, 0.5), (0.5, 0.5))
    window = FakeWindow(fail_step=1)
    with pytest.raises(NavigationError, match="step1"):
        MenuNavigator(window, config, sleep=lambda _: None).run()


def test_run_wraps_capture_error():
    config = make_config((0.5, 0.5))
    window = FakeWindow(fail_step=0, fail_on="capture")
    with pytest.raises(NavigationError, match="step0"):
        MenuNavigator(window, config, sleep=lambda _: None).run()


def test_load_shipped_config_is_calibrated_and_ready():
    config = MenuConfig.load("configs/menu.steam.yaml")
    assert [step.name for step in config.steps] == [
        "main_ranked", "bronze", "sanma_east"]
    assert all(step.click is not None for step in config.steps)
    assert all(0 < x < 1 and 0 < y < 1 for step in config.steps for x, y in [step.click])
    assert config.ready
    assert config.table_timeout_seconds == 120.0


def test_load_ready_when_all_clicks_present(tmp_path):
    path = write_config(tmp_path, """
steps:
- name: a
  click: [0.5, 0.5]
  settle_seconds: 1.0
table_timeout_seconds: 60
""")
    config = MenuConfig.load(path)
    assert config.ready
    assert config.steps[0].click == (0.5, 0.5)
    assert config.table_timeout_seconds == 60.0


@pytest.mark.parametrize("body", [
    "steps: []\n",
    "steps:\n- name: a\n  click: [0.5, 0.5]\n- name: a\n  click: [0.4, 0.4]\n",
    "steps:\n- name: a\n  click: [1.5, 0.5]\n",
    "steps:\n- name: a\n  click: [0.5, 0.0]\n",
    "steps:\n- name: a\n  click: [0.5, 0.5]\n  settle_seconds: -1\n",
    "steps:\n- name: ''\n  click: [0.5, 0.5]\n",
])
def test_load_rejects_invalid_configs(tmp_path, body):
    with pytest.raises(ValueError):
        MenuConfig.load(write_config(tmp_path, body))


def test_calibrate_menu_saves_numbered_frames_until_quit(tmp_path):
    window = FakeWindow()
    answers = iter(["", "", "q"])
    saved = calibrate_menu(window, tmp_path / "out", prompt=lambda _: next(answers))
    assert [path.name for path in saved] == ["step-01.png", "step-02.png"]
    for path in saved:
        assert path.is_file()
        assert path.parent == tmp_path / "out"


def test_calibrate_menu_quits_immediately(tmp_path):
    window = FakeWindow()
    saved = calibrate_menu(FakeWindow(), tmp_path / "out", prompt=lambda _: "q")
    assert saved == []
    assert window.captures == 0
