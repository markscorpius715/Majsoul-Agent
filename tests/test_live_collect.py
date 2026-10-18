from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from majsoul.live.capture import LiveFrame
from majsoul.live.collector import (CollectConfig, CollectError, CollectionUI,
                                    MatchCollector, RoundLifecycle, TogglePanel,
                                    detect_confirm, find_on_dots, menu_score)
from majsoul.live.config import LiveConfig

ROOT = Path(__file__).resolve().parents[1]
COLLECT_CONFIG = ROOT / "configs" / "collect.steam.yaml"
LIVE_CONFIG = ROOT / "configs" / "live.steam.yaml"
SIZE = (2582, 1550)  # captured live window convention (w, h)


def _image(fill=(20, 25, 40)):
    image = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    image[:, :] = fill
    return image


def _draw_dot(image, y_norm, x_norm=0.0356, color=(80, 220, 120), radius=12):
    cv2.circle(image, (int(x_norm * SIZE[0]), int(y_norm * SIZE[1])),
               radius, color, -1)


def _paste_closed_tab(image):
    template = cv2.imread(str(ROOT / "artifacts/templates/collect/closed_tab.png"))
    x, y = int(.005 * SIZE[0]), int(.74 * SIZE[1])
    image[y:y+template.shape[0], x:x+template.shape[1]] = template


def _table_image():
    image = _image()
    _paste_closed_tab(image)
    return image


# ------------------------------------------------------------------- config

def test_collect_config_loads_shipped_file():
    config = CollectConfig.load(COLLECT_CONFIG)
    assert config.live == ROOT / "configs" / "live.steam.yaml"
    assert config.dataset_root == Path(r"E:\majsoul_dataset")
    assert config.toggle_rows["auto_sort"] == 0.402
    assert config.toggle_targets == ("auto_win", "no_call", "auto_discard",
                                     "auto_kita")
    assert config.menu_template.is_file()


def test_collect_config_rejects_bad_toggle_rows(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("toggle_rows: {auto_sort: 0.2}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="toggle_rows"):
        CollectConfig.load(path)


def test_collect_config_rejects_unknown_target(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "toggle_rows: {auto_sort: 0.2, auto_win: 0.3, no_call: 0.4,\n"
        "  auto_discard: 0.5, auto_kita: 0.6}\n"
        "toggle_targets: [nope]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="toggle_targets"):
        CollectConfig.load(path)


# ---------------------------------------------------------------- detection

def test_find_on_dots_detects_filled_discs_only():
    image = _image()
    _draw_dot(image, 0.402)
    _draw_dot(image, 0.732)
    # green glyph-like strokes must be rejected by the fill-ratio filter
    cv2.rectangle(image, (int(0.02 * SIZE[0]), int(0.48 * SIZE[1])),
                  (int(0.05 * SIZE[0]), int(0.5 * SIZE[1])), (80, 220, 120), 2)
    dots = find_on_dots(image, (0.005, 0.30, 0.08, 0.92))
    assert len(dots) == 2
    assert dots[0] == pytest.approx(0.402, abs=0.005)
    assert dots[1] == pytest.approx(0.732, abs=0.005)


def test_detect_confirm_yellow_region():
    image = _image()
    region = (0.84, 0.86, 0.97, 0.935)
    assert not detect_confirm(image, region)
    x1, y1 = int(0.85 * SIZE[0]), int(0.87 * SIZE[1])
    x2, y2 = int(0.96 * SIZE[0]), int(0.925 * SIZE[1])
    cv2.rectangle(image, (x1, y1), (x2, y2), (60, 205, 240), -1)
    assert detect_confirm(image, region)


def test_detect_confirm_blue_variant_needs_gold_text():
    image = _image()
    region = (0.84, 0.86, 0.97, 0.935)
    # plain blue table felt must not be mistaken for the in-round confirm
    felt = _image((110, 80, 60))
    assert not detect_confirm(felt, region)
    # blue button body + gold text strokes
    blue_bgr = cv2.cvtColor(np.uint8([[[100, 128, 150]]]),
                            cv2.COLOR_HSV2BGR)[0, 0].tolist()
    x1, y1 = int(0.86 * SIZE[0]), int(0.875 * SIZE[1])
    x2, y2 = int(0.95 * SIZE[0]), int(0.92 * SIZE[1])
    cv2.rectangle(image, (x1, y1), (x2, y2), [int(c) for c in blue_bgr], -1)
    assert not detect_confirm(image, region)  # no gold text yet
    cv2.putText(image, "OK", (x1 + 30, y1 + 55), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (60, 190, 235), 4)
    assert detect_confirm(image, region)


def test_menu_score_matches_pasted_template():
    template = cv2.imread(str(ROOT / "artifacts/templates/menu/ranked_banner.png"))
    assert template is not None
    region = (0.45, 0.12, 0.90, 0.50)
    noise = np.random.default_rng(7).integers(
        0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
    assert menu_score(noise, template, region) < 0.55
    image = noise.copy()
    image[420:620, 1460:2080] = template
    assert menu_score(image, template, region) > 0.9


def test_menu_score_on_real_menu_capture():
    probe = ROOT / "artifacts" / "probe_pw_occluded.png"
    if not probe.is_file():
        pytest.skip("probe capture not available")
    image = cv2.imread(str(probe))
    template = cv2.imread(str(ROOT / "artifacts/templates/menu/ranked_banner.png"))
    config = CollectConfig.load(COLLECT_CONFIG)
    assert menu_score(image, template, config.menu_region) >= config.menu_threshold


# -------------------------------------------------------------- toggle panel

class FakeWindow:
    """Drawer simulation: triangle click toggles the panel; row clicks flip dots."""

    def __init__(self, rows):
        self.rows = dict(rows)
        self.on = {"auto_sort"}
        self.drawer_open = False
        self.clicks = []

    def window_rect(self):
        return (0, 0, SIZE[0], SIZE[1])

    def click(self, point, **kwargs):
        self.clicks.append(point)
        x, y = point
        near_closed_tab = abs(x - 0.025 * SIZE[0]) < 25 and abs(y - 0.761 * SIZE[1]) < 25
        near_open_tab = abs(x - 0.155 * SIZE[0]) < 25 and abs(y - 0.626 * SIZE[1]) < 25
        if (not self.drawer_open and near_closed_tab) or (self.drawer_open and near_open_tab):
            self.drawer_open = not self.drawer_open
            return
        if self.drawer_open:
            for label, row in self.rows.items():
                if abs(y - row * SIZE[1]) < 0.02 * SIZE[1]:
                    self.on.symmetric_difference_update({label})
                    return

    def capture(self):
        image = _image()
        if self.drawer_open:
            template = cv2.imread(str(ROOT / "artifacts/templates/collect/open_tab.png"))
            image[853:853+template.shape[0], 385:385+template.shape[1]] = template
            for label in self.rows:
                _draw_dot(image, self.rows[label], color=(80, 220, 120) if label in self.on else (102, 62, 52))
        else:
            _paste_closed_tab(image)
        return LiveFrame(image, 0.0, (0, 0, SIZE[0], SIZE[1]), (11, 45))


def _panel(fake, config):
    return TogglePanel(fake, config, sleep=lambda _: None)


def test_toggle_panel_enables_all_targets_and_closes_drawer():
    config = CollectConfig.load(COLLECT_CONFIG)
    fake = FakeWindow(config.toggle_rows)
    missing = _panel(fake, config).ensure()
    assert missing == []
    assert fake.on >= set(config.toggle_targets)
    assert not fake.drawer_open  # closed again at the end
    # triangle open + 4 toggle clicks + triangle close
    assert len(fake.clicks) == 6


def test_toggle_panel_skips_rows_already_on():
    config = CollectConfig.load(COLLECT_CONFIG)
    fake = FakeWindow(config.toggle_rows)
    fake.on.add("auto_win")
    fake.on.add("auto_kita")
    missing = _panel(fake, config).ensure()
    assert missing == []
    assert len(fake.clicks) == 2 + 2  # open + 2 missing toggles + close


def test_agent_turns_auto_options_off_without_touching_sort():
    config = CollectConfig.load(COLLECT_CONFIG)
    fake = FakeWindow(config.toggle_rows)
    fake.on.update(config.toggle_targets)
    assert _panel(fake, config).ensure(enabled=False) == []
    assert fake.on == {"auto_sort"}
    assert not fake.drawer_open
    assert len(fake.clicks) == 6
    assert _panel(fake, config).ensure(enabled=False) == []
    assert len(fake.clicks) == 8  # Recheck OFF without inverting the toggles.


def test_toggle_panel_reports_unverifiable_rows():
    config = CollectConfig.load(COLLECT_CONFIG)

    class StubbornWindow(FakeWindow):
        def click(self, point, **kwargs):
            super().click(point, **kwargs)
            self.on.discard("auto_kita")  # refuses to stay on

    fake = StubbornWindow(config.toggle_rows)
    missing = _panel(fake, config).ensure()
    assert missing == ["auto_kita"]


def test_toggle_panel_waits_through_occluded_switch_before_click():
    config = CollectConfig.load(COLLECT_CONFIG)

    class AnimationWindow(FakeWindow):
        reads = 0

        def capture(self):
            frame = super().capture()
            if self.drawer_open:
                self.reads += 1
                if self.reads <= 2:
                    _draw_dot(frame.image, self.rows["auto_win"], color=(230, 180, 230))
            return frame

    fake = AnimationWindow(config.toggle_rows)
    assert _panel(fake, config).ensure() == []
    assert fake.on >= set(config.toggle_targets)
    assert len(fake.clicks) == 6


def test_toggle_panel_refuses_alternating_switch_even_when_last_frame_is_off():
    config = CollectConfig.load(COLLECT_CONFIG)

    class FlickeringWindow(FakeWindow):
        reads = 0

        def capture(self):
            frame = super().capture()
            if self.drawer_open:
                self.reads += 1
                _draw_dot(frame.image, self.rows["auto_win"],
                          color=(80, 220, 120) if self.reads % 2 else (102, 62, 52))
            return frame

    fake = FlickeringWindow(config.toggle_rows)
    assert _panel(fake, config).ensure() == ["auto_win"]
    assert "auto_win" not in fake.on
    assert len(fake.clicks) == 5


def test_toggle_panel_missing_alignment_with_offset():
    config = CollectConfig.load(COLLECT_CONFIG)
    fake = FakeWindow(config.toggle_rows)
    panel = _panel(fake, config)
    # dots shifted by +0.01 still map onto rows
    dots = [row + 0.01 for row in config.toggle_rows.values()]
    assert panel.missing(dots) == []


# -------------------------------------------------------------- match ending

class EndScreenWindow:
    def __init__(self, frames):
        self.frames = list(frames)
        self.clicks = []

    def restore_if_minimized(self):
        pass

    def click(self, point, **kwargs):
        self.clicks.append(point)

    def capture(self):
        image = self.frames.pop(0) if self.frames else self.last
        self.last = image
        return LiveFrame(image, 0.0, (0, 0, SIZE[0], SIZE[1]), (11, 45))


def _collector(fake, tmp_path):
    config = CollectConfig.load(COLLECT_CONFIG)
    config = replace(config, dataset_root=tmp_path)
    live = LiveConfig.load(LIVE_CONFIG)
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    collector = MatchCollector(
        fake, config, live, sleep=sleep, now=lambda: clock[0],
        notifier=type("N", (), {"notify_text": lambda self, msg, image=None: True})())
    return collector


def _menu_image():
    template = cv2.imread(str(ROOT / "artifacts/templates/menu/ranked_banner.png"))
    image = _image()
    image[420:620, 1460:2080] = template
    return image


def _confirm_image():
    image = _image()
    cv2.rectangle(image, (int(0.85 * SIZE[0]), int(0.86 * SIZE[1])),
                  (int(0.96 * SIZE[0]), int(0.93 * SIZE[1])), (60, 205, 240), -1)
    return image


def test_return_to_menu_clicks_confirm_until_menu(tmp_path):
    fake = EndScreenWindow([_confirm_image()] * 6 + [_menu_image()])
    collector = _collector(fake, tmp_path)
    assert collector._return_to_menu()
    assert len(fake.clicks) == 2


@pytest.mark.parametrize('failures', [1, 3])
def test_match_confirm_focus_failure_reobserves_and_has_retry_limit(tmp_path, failures):
    from majsoul.live.capture import FocusUnavailable

    class FocusWindow(EndScreenWindow):
        attempts = 0

        def __init__(self):
            super().__init__([])

        def capture(self):
            self.last = _menu_image() if self.clicks else _confirm_image()
            return super().capture()

        def click(self, point, **kwargs):
            self.attempts += 1
            if self.attempts <= failures:
                raise FocusUnavailable('No input sent')
            super().click(point, **kwargs)

    fake = FocusWindow()
    collector = _collector(fake, tmp_path)
    if failures < collector.config.confirm_focus_attempts:
        collector._play_match()
        assert len(fake.clicks) == 1
        assert fake.attempts == failures+1
    else:
        with pytest.raises(CollectError, match='bounded retries'):
            collector._play_match()
        assert not fake.clicks
        assert fake.attempts == collector.config.confirm_focus_attempts


def test_resolve_roi_bakes_projection_for_frame(tmp_path):
    fake = EndScreenWindow([_menu_image()])
    collector = _collector(fake, tmp_path)
    frame = fake.capture()
    roi_eff = collector._resolve_roi(frame)
    assert roi_eff is collector._resolve_roi(frame)  # cached per size
    # The current Steam calibration matches the captured window exactly.
    assert tuple(roi_eff.reference_size) == tuple(collector._roi.reference_size)
    assert roi_eff.regions == collector._roi.regions


def test_return_to_menu_times_out_and_archives(tmp_path):
    fake = EndScreenWindow([_confirm_image()])
    collector = _collector(fake, tmp_path)
    assert not collector._return_to_menu()
    assert (tmp_path / collector.artifacts.name / "end_screen_stuck.jpg").is_file()


@pytest.mark.parametrize("confirmations", [1, 2, 3])
def test_result_confirms_wait_for_stable_table_return(confirmations):
    lifecycle = RoundLifecycle(ready_seconds=1)
    assert lifecycle.update(confirm=False, table=True, now=0) == ()
    assert lifecycle.update(confirm=False, table=True, now=1) == ("table_ready",)
    for n in range(confirmations):
        base = 2 + n * 4
        assert lifecycle.update(confirm=True, table=False, now=base) == ("confirm_appeared",)
        assert "table_ready" not in lifecycle.update(confirm=False, table=False, now=base+3)
    base += 4
    assert lifecycle.update(confirm=False, table=True, now=base) == ()
    assert lifecycle.update(confirm=False, table=True, now=base+1) == ("table_ready",)
    assert lifecycle.round_index == 2
    assert lifecycle.update(confirm=False, table=True, now=base+2) == ()


def test_brief_table_flash_between_confirms_does_not_enable_options():
    lifecycle = RoundLifecycle(ready_seconds=1)
    lifecycle.update(confirm=True, table=False, now=0)
    lifecycle.update(confirm=False, table=True, now=3)
    assert "table_ready" not in lifecycle.update(confirm=True, table=False, now=3.5)
    assert lifecycle.round_index == 0


def test_confirm_starting_at_zero_has_correct_duration():
    lifecycle = RoundLifecycle(ready_seconds=1)
    lifecycle.update(confirm=True, table=False, now=0)
    lifecycle.update(confirm=True, table=False, now=9)
    assert lifecycle.confirm_since == 0


def test_ui_anchors_scale_and_reject_dim_result_overlay():
    ui = CollectionUI(CollectConfig.load(COLLECT_CONFIG))
    image = _table_image()
    assert ui.table_visible(image)
    assert ui.table_visible(cv2.resize(image, (1291, 775)))
    assert not ui.table_visible((image * 0.2).astype(np.uint8))
    assert not ui.table_visible(_confirm_image())


def test_panel_works_when_auto_sort_is_disabled():
    config = CollectConfig.load(COLLECT_CONFIG)
    fake = FakeWindow(config.toggle_rows)
    fake.on.clear()
    assert _panel(fake, config).ensure() == []
    assert fake.on == set(config.toggle_targets)


def test_panel_refuses_unknown_switch_color():
    config = CollectConfig.load(COLLECT_CONFIG)

    class UnknownSwitch(FakeWindow):
        def capture(self):
            frame = super().capture()
            if self.drawer_open:
                _draw_dot(frame.image, self.rows["auto_kita"], color=(255, 255, 255))
            return frame

    fake = UnknownSwitch(config.toggle_rows)
    assert _panel(fake, config).ensure() == ["auto_kita"]
    assert "auto_kita" not in fake.on


def test_panel_does_not_click_when_confirm_appears_mid_setup():
    config = CollectConfig.load(COLLECT_CONFIG)

    class ResultDuringSetup(FakeWindow):
        def capture(self):
            if len(self.clicks) >= 2:
                return LiveFrame(_confirm_image(), 0, self.window_rect(), (11, 45))
            return super().capture()

    fake = ResultDuringSetup(config.toggle_rows)
    with pytest.raises(CollectError, match="result appeared"):
        _panel(fake, config).ensure()
    assert len(fake.clicks) == 2  # open and first switch only


def test_confirm_click_cancelled_when_countdown_dismisses(tmp_path):
    fake = EndScreenWindow([_confirm_image(), _menu_image()])
    collector = _collector(fake, tmp_path)
    collector._click_confirm(fake.capture())
    assert fake.clicks == []


def test_last_match_returns_to_lobby_before_counting_complete(tmp_path, monkeypatch):
    fake = EndScreenWindow([_menu_image()])
    collector = _collector(fake, tmp_path)
    collector.max_matches = 2
    calls = []
    monkeypatch.setattr(collector, "_enter_match", lambda: calls.append("enter"))
    monkeypatch.setattr(collector, "_play_match", lambda: calls.append("play_and_return"))
    assert collector.run() == 0
    assert calls == ["enter", "play_and_return", "enter", "play_and_return"]


def test_stop_file_interrupts_without_focus_or_click(tmp_path):
    fake = EndScreenWindow([_table_image()])
    collector = _collector(fake, tmp_path)
    collector.artifacts.mkdir()
    (collector.artifacts / "STOP").touch()
    with pytest.raises(KeyboardInterrupt):
        collector._capture()
    assert not fake.clicks


def test_enter_match_does_not_require_layout_ocr(tmp_path, monkeypatch):
    fake = EndScreenWindow([_table_image()])
    collector = _collector(fake, tmp_path)
    monkeypatch.setattr(collector, "_layout_resolver", lambda *args: pytest.fail("UI must not wait on OCR"))
    collector._enter_match()
    assert collector._active_slots is None
    assert not fake.clicks


def test_unknown_entry_screen_never_triggers_blind_menu_navigation(tmp_path, monkeypatch):
    fake = EndScreenWindow([_image()])
    collector = _collector(fake, tmp_path)
    collector.config = replace(collector.config, table_timeout_seconds=1)
    monkeypatch.setattr("majsoul.live.collector.MenuNavigator.run", lambda *args: pytest.fail("unknown UI cannot navigate"))
    with pytest.raises(CollectError, match="table screen not detected"):
        collector._enter_match()
    assert not fake.clicks


def test_failed_auto_options_stop_after_bounded_retries(tmp_path, monkeypatch):
    config = CollectConfig.load(COLLECT_CONFIG)

    class StubbornWindow(FakeWindow):
        def click(self, point, **kwargs):
            super().click(point, **kwargs)
            self.on.discard("auto_kita")

    fake = StubbornWindow(config.toggle_rows)
    collector = _collector(fake, tmp_path)
    monkeypatch.setattr(collector, "_track", lambda *args: None)
    with pytest.raises(CollectError, match="bounded retries"):
        collector._play_match()
    assert "auto_kita" not in fake.on


def test_visual_frames_survive_without_semantic_recognition(tmp_path):
    import json
    fake = EndScreenWindow([_table_image()])
    collector = _collector(fake, tmp_path)
    collector._visual_record(fake.capture())
    collector._visual_record(fake.capture())
    records = [json.loads(line) for line in (collector.artifacts / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["observation"] is None
    assert records[0]["player_count"] == 3
    assert (collector.artifacts / records[0]["file"]).is_file()


def test_small_discard_change_is_saved_despite_low_whole_image_difference(tmp_path):
    fake = EndScreenWindow([_table_image()])
    collector = _collector(fake, tmp_path)
    before = fake.capture()
    collector._visual_record(before)
    after = fake.capture()
    after.image = after.image.copy()
    x, y = int(.5 * SIZE[0]), int(.6 * SIZE[1])
    cv2.rectangle(after.image, (x,y), (x+50,y+80), (220,220,220), -1)
    assert np.abs(after.image.astype(float)-before.image).mean()/255 < collector.config.visual_change_threshold
    collector.sleep(collector.config.keyframe_min_seconds)
    collector._visual_record(after)
    assert collector._seq == 2


@pytest.mark.parametrize('operator_takeover',[False,True])
def test_match_lifecycle_enables_initial_and_next_hand_without_ocr(tmp_path, monkeypatch,operator_takeover):
    import json
    config = CollectConfig.load(COLLECT_CONFIG)
    clock = [0.0]

    class AutoplayWindow(FakeWindow):
        final_clicks = 0
        reset_done = False

        def capture(self):
            now = clock[0]
            if now >= 11.5 and not self.reset_done:
                self.on = set()  # All five reset; takeover must also restore auto-sort.
                self.reset_done = True
            if self.final_clicks >= 2:
                image = _menu_image()
            elif 5 <= now < 8 or 8.5 <= now < 11.5:
                image = _blue_confirm_image()
            elif now >= 17:
                image = _confirm_image()
            elif 8 <= now < 8.5:
                image = _image()
            else:
                return super().capture()
            return LiveFrame(image, now, self.window_rect(), (11, 45))

        def click(self, point, **kwargs):
            assert not (5 <= clock[0] < 11.5), "must leave auto confirms to countdown"
            if clock[0] >= 17:
                self.clicks.append(point)
                self.final_clicks += 1
            else:
                super().click(point, **kwargs)

    fake = AutoplayWindow(config.toggle_rows)
    kwargs=dict(now=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        notifier=type("N", (), {"notify_text": lambda *args: True})())
    if operator_takeover:
        from tools.live_test_supervisor import finish_with_client_takeover
        from majsoul.live.menu import MenuNavigator
        fake.restore_if_minimized=lambda:None
        monkeypatch.setattr(MenuNavigator,'run',lambda *a:pytest.fail('takeover must not match a new game'))
        monkeypatch.setattr(MatchCollector,'_pipeline',lambda *a:pytest.fail('takeover must not wait for CNN'))
        directory=tmp_path/'takeover'
        report=finish_with_client_takeover(fake,config,LiveConfig.load(LIVE_CONFIG),directory,**kwargs)
        assert report['returned_lobby'] and report['agent_acceptance'] is False
        assert fake.on == {'auto_sort','auto_win','no_call','auto_discard','auto_kita'}
    else:
        collector = MatchCollector(fake, replace(config, dataset_root=tmp_path),
            LiveConfig.load(LIVE_CONFIG), **kwargs)
        monkeypatch.setattr(collector, "_track", lambda *args: None)
        collector._play_match()
        assert collector._round_index == 2
        directory=collector.artifacts
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([event for event in events if event["event"] == "toggle_setup_done"]) == 2
    assert events[-1]['round_index'] == 2
    assert fake.on >= set(config.toggle_targets)
    assert fake.final_clicks == 2
    if operator_takeover:
        assert events[-1]['event']=='session_end'
        assert sum(e['event']=='match_complete' for e in events)==1
    else:
        assert events[-1]["event"] == "menu_reached"


def _blue_confirm_image():
    image = _image()
    body = cv2.cvtColor(np.uint8([[[100, 128, 150]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    x, y = int(.85 * SIZE[0]), int(.87 * SIZE[1])
    cv2.rectangle(image, (x,y), (int(.96 * SIZE[0]), int(.925 * SIZE[1])), body, -1)
    cv2.putText(image, "OK", (x+30,y+55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (60,190,235),4)
    return image


def test_toggle_panel_waits_for_a_slow_drawer_animation():
    """protected-20261018-151224: the handback was rejected because the drawer
    was checked once, 0.35 s after the click, under heavy CPU load."""
    config = CollectConfig.load(COLLECT_CONFIG)

    class SlowDrawer(FakeWindow):
        hidden_reads = 3

        def capture(self):
            if self.drawer_open and self.hidden_reads:
                self.hidden_reads -= 1
                self.drawer_open = False
                frame = super().capture()
                self.drawer_open = True
                return frame
            return super().capture()

    fake = SlowDrawer(config.toggle_rows)
    fake.on.update(config.toggle_targets)
    assert _panel(fake, config).ensure(enabled=False) == []
    assert fake.on == {"auto_sort"}
    assert len(fake.clicks) == 6  # no extra triangle click while it animates


def test_toggle_panel_reclicks_once_only_when_the_first_click_was_lost():
    config = CollectConfig.load(COLLECT_CONFIG)

    class LostFirstClick(FakeWindow):
        lost = False

        def click(self, point, **kwargs):
            if not self.lost:
                self.lost = True
                self.clicks.append(point)
                return
            super().click(point, **kwargs)

    fake = LostFirstClick(config.toggle_rows)
    assert _panel(fake, config).ensure() == []
    assert fake.on >= set(config.toggle_targets)
    assert len(fake.clicks) == 7  # lost open + re-open + 4 toggles + close


def test_toggle_panel_never_clicks_more_than_twice_to_open():
    config = CollectConfig.load(COLLECT_CONFIG)

    class DeadDrawer(FakeWindow):
        def click(self, point, **kwargs):
            self.clicks.append(point)

    fake = DeadDrawer(config.toggle_rows)
    with pytest.raises(CollectError, match="toggle drawer is not visibly open"):
        _panel(fake, config).ensure()
    assert len(fake.clicks) == 2
