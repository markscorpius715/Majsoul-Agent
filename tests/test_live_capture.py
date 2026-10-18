import numpy as np
import pytest

from majsoul.calibration.roi_config import ROIConfig, default_layout
from majsoul.live.capture import LiveWindow, roi_for_frame
from majsoul.perception.roi_extractor import extract_roi
from majsoul.window import WindowError


def config(regions=None):
    slots, variants, self_slot = default_layout(3)
    return ROIConfig(reference_size=[200, 100], player_count=3,
                     screen_slots=slots, layout_variants=variants,
                     self_slot=self_slot, kita=True,
                     regions=regions if regions is not None
                     else {"self_hand": [0.13, 0.25, 0.87, 0.75]})


def test_stretch_bakes_normalized_geometry():
    baked = roi_for_frame(config(), (400, 300), "stretch")
    assert baked.reference_size == [400, 300]
    # 2x/3y anisotropic scaling of [0.13, 0.25, 0.87, 0.75] lands on whole pixels.
    assert baked.regions["self_hand"] == pytest.approx([0.13, 0.25, 0.87, 0.75])
    assert baked.player_count == 3 and baked.kita is True


def test_fit_width_top_shifts_only_after_scaling():
    baked = roi_for_frame(config(), (200, 150), "fit_width_top")
    assert baked.reference_size == [200, 150]
    # Uniform scale 1.0 with top alignment: (26, 25, 174, 75) px over 200x150.
    assert baked.regions["self_hand"] == pytest.approx([0.13, 25 / 150, 0.87, 0.5])


def test_auto_rejects_review_required_and_keeps_safe_config():
    item = config()
    assert roi_for_frame(item, (400, 200)) is item
    assert roi_for_frame(item, (202, 100)) is item
    with pytest.raises(ValueError, match="projection"):
        roi_for_frame(item, (400, 300))


def test_explicit_projection_refuses_clipped_regions():
    item = config({"self_hand": [0.1, 0.6, 0.9, 1.0]})
    # fit_width_top at 100x40 scales content to 100x50, clipping self_hand below.
    with pytest.raises(ValueError, match="self_hand"):
        roi_for_frame(item, (100, 40), "fit_width_top")


def test_baked_config_round_trips_through_extract_roi():
    item = config({"self_hand": [0.1, 0.6, 0.9, 1.0], "river_top": None})
    baked = roi_for_frame(item, (400, 200), "stretch")
    assert baked.regions["river_top"] is None
    image = np.arange(400 * 200 * 3, dtype=np.uint8).reshape(200, 400, 3)
    crop = extract_roi(image, baked, "self_hand")
    np.testing.assert_array_equal(crop.image, image[120:200, 40:360])


def test_find_requires_exactly_one_title_match(monkeypatch):
    windows = [{"hwnd": 11, "title": "雀魂麻将"}, {"hwnd": 22, "title": "other windows"}]
    monkeypatch.setattr("majsoul.live.capture.list_windows", lambda: windows)
    assert LiveWindow.find() == 11
    assert LiveWindow.find(r"other") == 22

    both = windows + [{"hwnd": 33, "title": "雀魂 - Steam"}]
    monkeypatch.setattr("majsoul.live.capture.list_windows", lambda: both)
    with pytest.raises(WindowError, match="雀魂麻将"):
        LiveWindow.find()

    monkeypatch.setattr("majsoul.live.capture.list_windows", lambda: [])
    with pytest.raises(WindowError, match="found 0"):
        LiveWindow.find()


def test_find_filters_same_title_in_browser_or_conversation(monkeypatch):
    windows = [
        {"hwnd": 11, "title": "雀魂麻將", "class_name": "UnityWndClass"},
        {"hwnd": 22, "title": "雀魂麻将后台自动采集", "class_name": "Chrome_WidgetWin_1"},
    ]
    monkeypatch.setattr("majsoul.live.capture.list_windows", lambda: windows)
    assert LiveWindow.find("雀魂", "^UnityWndClass$") == 11


def test_click_rejects_stale_observation_before_stealing_focus(monkeypatch):
    from majsoul.live.capture import LiveFrame
    window = LiveWindow.__new__(LiveWindow)
    window._check_capture = lambda: None
    window.window_rect = lambda: (0, 0, 400, 300)
    window._force_foreground = lambda: pytest.fail("must not steal focus")
    frame = LiveFrame(np.zeros((300,400,3), dtype=np.uint8), 0,
                      (0,0,400,300), (0,0))
    monkeypatch.setattr("majsoul.live.capture.time.monotonic", lambda: 10)
    with pytest.raises(WindowError, match="stale"):
        window.click((100,100), expected=frame)


def test_click_rejects_moved_window_before_stealing_focus():
    from majsoul.live.capture import LiveFrame
    window = LiveWindow.__new__(LiveWindow)
    window._check_capture = lambda: None
    window.window_rect = lambda: (20, 0, 400, 300)
    window._force_foreground = lambda: pytest.fail("must not steal focus")
    frame = LiveFrame(np.zeros((300,400,3), dtype=np.uint8), 0,
                      (0,0,400,300), (0,0))
    with pytest.raises(WindowError, match="moved"):
        window.click((100,100), expected=frame)


@pytest.mark.parametrize('fails',[False,True])
def test_click_measures_foreground_occupancy_and_restores_on_input_error(monkeypatch,fails):
    from types import SimpleNamespace
    from majsoul.live.capture import LiveFrame
    clock=[100.]
    focus=[17]
    monkeypatch.setattr('majsoul.live.capture.time.monotonic',lambda:clock[0])
    monkeypatch.setattr('majsoul.live.capture.time.sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    window=LiveWindow.__new__(LiveWindow)
    window._check_capture=lambda:None
    window.window_rect=lambda:(0,0,400,300)
    window.api=SimpleNamespace(GetForegroundWindow=lambda:focus[0],GetCursorPos=lambda _:False)
    window._force_foreground=lambda:focus.__setitem__(0,42)
    window._restore_foreground=lambda previous:focus.__setitem__(0,previous)
    def click(point,bounds,*,hold_seconds):
        assert focus[0]==42 and point==(100,100)
        clock[0]+=hold_seconds
        if fails:
            raise WindowError('input adapter failure')
    window.client=SimpleNamespace(bounds=lambda:(0,0,400,300),click=click)
    frame=LiveFrame(np.zeros((300,400,3),np.uint8),100.,(0,0,400,300),(0,0))
    if fails:
        with pytest.raises(WindowError,match='adapter'):
            window.click((100,100),expected=frame)
    else:
        window.click((100,100),expected=frame)
    assert focus[0]==17
    timing=window.last_click_timing
    assert timing['input_completed'] is (not fails)
    assert timing['previous_foreground_restored']
    assert timing['foreground_seconds']==pytest.approx(.13 if fails else .21)
