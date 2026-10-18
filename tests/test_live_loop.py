from dataclasses import replace
import json
from pathlib import Path
from threading import Event
import time
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from majsoul.calibration.frames import write_image
from majsoul.calibration.roi_config import ROIConfig, default_layout
from majsoul.domain import Action, Box, Observation, State, TileView
from majsoul.live.buttons import ActionButtonReader, ButtonConfig
from majsoul.live.capture import LiveFrame
from majsoul.live.config import (DecisionConfig, ExecutionConfig, LiveConfig,
                                 NotifyConfig)
from majsoul.live.loop import (LiveAgentLoop, LiveError, decision_deadline,
                               preflight, verify_action)
from majsoul.live.menu import MenuConfig
from majsoul.live.notify import FeishuNotifier
from majsoul.perception.schemas import (MeldReading, RecognizedTile,
                                        SegmentedTile, TilePrediction,
                                        TileSequenceReading)
from majsoul.state import (FrameObservation, FrameObservationReader,
                           ObservedMeld, ObservedTile, PlayerObservation)
from majsoul.tiles import parse_hand

ROOT = Path(__file__).resolve().parents[1]
LIVE_CONFIG = ROOT / "configs" / "live.steam.yaml"
SLOTS = ("bottom", "top", "left")
SEATS = ("1z", "2z", "3z")
HAND14 = ("1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
          "1s", "1s", "2s", "3s", "9s")
HAND13 = HAND14[:-1]
WINDOW = (100, 100, 1000, 800)


# ---------------------------------------------------------------- LiveConfig

def test_live_config_loads_shipped_steam_config():
    config = LiveConfig.load(LIVE_CONFIG)
    assert config.window_title_regex == "雀魂"
    assert config.poll_seconds == 0.25
    assert config.stable_frames == 2
    assert config.layout_confirmations == 2
    assert config.projection == "auto"
    assert config.classifier == "cnn"
    assert config.cnn_checkpoint == (
        ROOT / "artifacts/tile-augment-20261002/runs/D_s42/checkpoint.pt")
    assert config.read_scores is False
    assert config.tile_templates == ROOT / "artifacts/templates/tiles"
    assert config.glyph_templates == ROOT / "artifacts/templates/glyphs"
    assert config.roi == ROOT / "configs/roi.steam.yaml"
    assert config.vision == ROOT / "configs/vision.steam.yaml"
    assert config.rules == ROOT / "configs/sanma.json"
    assert config.menu == ROOT / "configs/menu.steam.yaml"
    assert config.artifacts_root == ROOT / "artifacts/live"
    assert config.match_end_silence_seconds == 90
    assert config.persistent_issue_seconds == 10
    assert config.decision == DecisionConfig(
        policy="deepseek", jev_proxy=None,
        action_margin_seconds=1.0, max_decision_age=10.0,
        hedge_seconds=1.0, local_fallback_seconds=0.8)
    assert config.execution == ExecutionConfig(4.0, 0.02, hand_click_count=2)
    assert config.buttons == ButtonConfig(
        hsv_lower=(125, 10, 90), hsv_upper=(95, 255, 255),
        minimum_fill_ratio=0.25)
    assert config.button_templates == ROOT / "artifacts/templates/buttons"
    assert config.notify.enabled
    assert config.notify.chat_id == "oc_54dfa592aa8d0d5e5adefc017a969262"
    assert config.notify.feishu_helper.is_file()
    assert config.notify.node == "node"


def test_live_config_referenced_menu_loads():
    config = LiveConfig.load(LIVE_CONFIG)
    menu = MenuConfig.load(config.menu)
    assert [step.name for step in menu.steps] == [
        "main_ranked", "bronze", "sanma_east"]
    assert menu.ready


@pytest.mark.parametrize('proxy', [None, 'http://127.0.0.1:12345'])
def test_live_policy_route_overrides_stale_env_without_mutating_it(monkeypatch, proxy):
    import majsoul.live.loop as module
    monkeypatch.setenv('JEV_PROXY', 'http://retired.invalid:4567')
    monkeypatch.setattr(module.JevSettings, 'load', lambda _: module.JevSettings(
        'test', proxy='http://retired.invalid:4567'))
    monkeypatch.setattr(module, 'JevPolicy', lambda settings: settings)
    config = LiveConfig.load(LIVE_CONFIG)
    config = replace(config, decision=replace(config.decision, policy='jev', jev_proxy=proxy,
                                                       local_fallback_seconds=0))
    assert module.build_policy(config).proxy == proxy
    assert module.os.environ['JEV_PROXY'] == 'http://retired.invalid:4567'


def test_live_config_empty_mapping_uses_defaults(tmp_path):
    path = tmp_path / "live.yaml"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="feishu_helper"):
        LiveConfig.load(path)
    path.write_text(yaml.safe_dump({"notify": {"enabled": False}}),
                    encoding="utf-8")
    config = LiveConfig.load(path)
    assert config.window_title_regex == "雀魂"
    assert config.decision.jev_proxy is None
    assert config.decision.policy == "auto"
    assert config.buttons == ButtonConfig()
    assert not config.notify.enabled
    assert config.notify.feishu_helper is None


@pytest.mark.parametrize("data", [
    {"window_title_regex": "["},
    {"poll_seconds": 0},
    {"poll_seconds": 6},
    {"poll_seconds": -1},
    {"stable_frames": 0},
    {"layout_confirmations": 1},
    {"projection": "diagonal"},
    {"classifier": "soup"},
    {"match_end_silence_seconds": 0},
    {"persistent_issue_seconds": -5},
    {"decision": {"policy": "random"}},
    {"decision": {"jev_proxy": "ftp://proxy"}},
    {"decision": {"action_margin_seconds": 0}},
    {"decision": {"action_margin_seconds": float("inf")}},
    {"decision": {"max_decision_age": 0}},
    {"execution": {"verification_timeout": 0}},
    {"execution": {"target_tolerance": 0}},
    {"execution": {"target_tolerance": 0.5}},
    {"buttons": {"maximum_distance": 5}},
    {"notify": "nope"},
    {"notify": {"enabled": "yes"}},
    {"notify": {"enabled": True}},
    {"notify": {"enabled": True, "feishu_helper": "no/such/helper.mjs",
                "chat_id": "oc_test"}},
    {"notify": {"enabled": True, "feishu_helper": "configs/roi.yaml",
                "chat_id": ""}},
    {"notify": {"enabled": False, "node": "  "}},
])
def test_live_config_rejects_invalid_values(tmp_path, data):
    path = tmp_path / "live.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        LiveConfig.load(path)


# ---------------------------------------------------------- verify_action

def tile(label, orientation="upright"):
    return ObservedTile(label, orientation, 0.9)


def frame_observation(*, timestamp=1.0, frame_index=1, self_river=(),
                      self_melds=(), self_kita=0, action_buttons=None,
                      issues=(), round_id=("1z", 1, 0)):
    players = (
        PlayerObservation("bottom", "1z", True, 35000, river=tuple(self_river),
                          melds=tuple(self_melds), kita_count=self_kita),
        PlayerObservation("top", "2z", False, 35000, kita_count=0),
        PlayerObservation("left", "3z", False, 35000, kita_count=0),
    )
    return FrameObservation(
        timestamp, frame_index, SLOTS, round_id[0], round_id[1], round_id[2],
        0, 55, players, (), None, (),
        action_buttons=action_buttons, issues=tuple(issues))


def meld(*labels):
    return ObservedMeld(tuple(tile(label) for label in labels))


# Action evidence tests live in test_live_verification.py.


# ------------------------------------------------------- decision_deadline

def test_decision_deadline_uses_turn_budget_and_age_cap():
    assert decision_deadline(100.0, 102.0, turn_seconds=5, action_margin=1,
                             max_decision_age=10) == 104.0
    assert decision_deadline(None, 102.0, turn_seconds=5, action_margin=1,
                             max_decision_age=10) == 106.0
    assert decision_deadline(100.0, 100.5, turn_seconds=5, action_margin=1,
                             max_decision_age=2) == 102.5


# -------------------------------------------------------------- preflight

def domain_state(hand="19m123456p123s111z", **kwargs):
    defaults = dict(draw_index=13, phase="discard", round_id="1z:1:0",
                    seat_wind="1z", round_wind="1z", dora_indicators=("9s",))
    defaults.update(kwargs)
    tiles = parse_hand(hand)
    observation = Observation(
        10.0, WINDOW,
        hand=tuple(TileView(label, Box(20 + i * 40, 600, 50 + i * 40, 660))
                   for i, label in enumerate(tiles)),
        **defaults)
    return State(observation, True, ((), (), ()))


def discard_action(engine, domain, action_id="discard:13:1z"):
    return next(a for a in engine.legal_actions(domain) if a.id == action_id)


def test_preflight_accepts_and_returns_fresh_point(engine):
    config = LiveConfig.load(LIVE_CONFIG)
    domain = domain_state()
    action = discard_action(engine, domain)
    before = frame_observation(timestamp=10.0)
    fresh = frame_observation(timestamp=10.5, frame_index=2)
    point = preflight(action, before, fresh, domain, WINDOW, WINDOW, config,
                      engine, now=11.0)
    assert point == domain.observation.hand[13].box.center


def test_preflight_accepts_button_action(engine):
    config = LiveConfig.load(LIVE_CONFIG)
    box = Box(700, 600, 780, 640)
    domain = domain_state(hand="19m123456p123s11z", draw_index=None,
                          phase="response", buttons={"pass": box})
    action = next(a for a in engine.legal_actions(domain)
                  if a.id == "button:pass")
    before = frame_observation(timestamp=10.0)
    fresh = frame_observation(timestamp=10.5, frame_index=2)
    point = preflight(action, before, fresh, domain, WINDOW, WINDOW, config,
                      engine, now=11.0)
    assert point == box.center


def test_preflight_rejects_uncertain_fresh_observation(engine):
    config = LiveConfig.load(LIVE_CONFIG)
    domain = domain_state()
    action = discard_action(engine, domain)
    fresh = frame_observation(timestamp=10.5, frame_index=2,
                              issues=("hand: unknown",))
    with pytest.raises(LiveError, match="uncertain"):
        preflight(action, frame_observation(timestamp=10.0), fresh, domain,
                  WINDOW, WINDOW, config, engine, now=11.0)


def test_preflight_rejects_changed_own_hand(engine):
    config = LiveConfig.load(LIVE_CONFIG)
    domain = domain_state()
    action = discard_action(engine, domain)
    fresh = replace(frame_observation(timestamp=10.5, frame_index=2), hand=(tile('1p'),))
    with pytest.raises(LiveError, match="changed"):
        preflight(action, frame_observation(timestamp=10.0), fresh, domain,
                  WINDOW, WINDOW, config, engine, now=11.0)


def test_preflight_rejects_moved_window(engine):
    config = LiveConfig.load(LIVE_CONFIG)
    domain = domain_state()
    action = discard_action(engine, domain)
    before = frame_observation(timestamp=10.0)
    fresh = frame_observation(timestamp=10.5, frame_index=2)
    with pytest.raises(LiveError, match="Window geometry"):
        preflight(action, before, fresh, domain, WINDOW, (0, 0, 1000, 800),
                  config, engine, now=11.0)


def test_preflight_rejects_expired_observation(engine):
    config = LiveConfig.load(LIVE_CONFIG)
    domain = domain_state()
    action = discard_action(engine, domain)
    before = frame_observation(timestamp=10.0)
    fresh = frame_observation(timestamp=10.5, frame_index=2)
    with pytest.raises(LiveError, match="expired"):
        preflight(action, before, fresh, domain, WINDOW, WINDOW, config,
                  engine, now=30.0)


def test_preflight_rejects_no_longer_legal_action(engine):
    config = LiveConfig.load(LIVE_CONFIG)
    domain = domain_state()
    action = Action("button:ron", "ron", button="ron")
    before = frame_observation(timestamp=10.0)
    fresh = frame_observation(timestamp=10.5, frame_index=2)
    with pytest.raises(LiveError, match="no longer legal"):
        preflight(action, before, fresh, domain, WINDOW, WINDOW, config,
                  engine, now=11.0)


def test_preflight_rejects_moved_click_target(engine):
    config = LiveConfig.load(LIVE_CONFIG)
    domain = domain_state()
    action = discard_action(engine, domain)
    before = frame_observation(timestamp=10.0)
    fresh = frame_observation(timestamp=10.5, frame_index=2)
    with pytest.raises(LiveError, match="tolerance"):
        preflight(action, before, fresh, domain, WINDOW, WINDOW, config,
                  engine, point_before=(0.0, 0.0), now=11.0)


def test_preflight_rejects_disappeared_button(engine):
    config = LiveConfig.load(LIVE_CONFIG)
    box = Box(700, 600, 780, 640)
    domain = domain_state(hand="19m123456p123s11z", draw_index=None,
                          phase="response", buttons={"pass": box})
    action = next(a for a in engine.legal_actions(domain)
                  if a.id == "button:pass")
    vanished = domain_state(hand="19m123456p123s11z", draw_index=None,
                            phase="response", buttons={})
    before = frame_observation(timestamp=10.0)
    fresh = frame_observation(timestamp=10.5, frame_index=2)
    with pytest.raises(LiveError, match="no longer legal"):
        preflight(action, before, fresh, vanished, WINDOW, WINDOW, config,
                  engine, now=11.0)


# ------------------------------------------------- buttons last_buttons cache

DARK = (30, 30, 30)
BRIGHT = (170, 250, 170)
INK = (20, 20, 20)
BUTTON_A = (110, 120, 180, 170)
BUTTON_B = (210, 120, 280, 170)


def button_roi():
    slots, variants, self_slot = default_layout(3)
    return ROIConfig(reference_size=[400, 200], player_count=3,
                     screen_slots=slots, layout_variants=variants,
                     self_slot=self_slot, kita=False,
                     regions={"action_buttons": [0.25, 0.5, 0.75, 1.0]})


def draw_button(image, rect):
    x1, y1, x2, y2 = rect
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), BRIGHT, -1)
    cv2.rectangle(image, (x1 + 18, y1 + 12), (x1 + 30, y1 + 38), INK, -1)
    cv2.rectangle(image, (x1 + 34, y1 + 24), (x1 + 52, y1 + 34), INK, -1)


def write_button_templates(root, image):
    x1, y1, x2, y2 = BUTTON_A
    write_image(root / "ron" / "sample.png", image[y1:y2, x1:x2])


def test_read_caches_last_buttons(tmp_path):
    image = np.full((200, 400, 3), DARK, dtype=np.uint8)
    draw_button(image, BUTTON_A)
    write_button_templates(tmp_path / "templates", image)
    reader = ActionButtonReader(ButtonConfig(), tmp_path / "templates")
    assert reader.last_buttons == {}
    buttons = reader.read(image, button_roi())
    assert buttons == {"ron": Box(*BUTTON_A)}
    assert reader.last_buttons is buttons
    dark = np.full((200, 400, 3), DARK, dtype=np.uint8)
    assert reader.read(dark, button_roi()) == {}
    assert reader.last_buttons == {}


def test_read_caches_empty_result_without_templates(tmp_path):
    reader = ActionButtonReader(ButtonConfig(), tmp_path / "missing")
    image = np.full((200, 400, 3), DARK, dtype=np.uint8)
    draw_button(image, BUTTON_A)
    assert reader.read(image, button_roi()) == {}
    assert reader.last_buttons == {}


def test_instance_unknown_sink_is_used_by_reader_internal_call(tmp_path):
    image = np.full((200, 400, 3), DARK, dtype=np.uint8)
    draw_button(image, BUTTON_A)
    rng = np.random.default_rng(7)
    x1, y1, x2, y2 = BUTTON_B
    noise = rng.integers(170, 251, size=(y2 - y1, x2 - x1), dtype=np.uint8)
    image[y1:y2, x1:x2] = cv2.merge([noise, noise, noise])
    write_button_templates(tmp_path / "templates", image)
    reader = ActionButtonReader(ButtonConfig(), tmp_path / "templates")
    seen = []
    reader.unknown_sink = lambda *args: seen.append(args)
    buttons = reader.read(image, button_roi())
    assert buttons == {"ron": Box(*BUTTON_A)}
    assert len(seen) == 1
    assert seen[0][2] == Box(*BUTTON_B)


# ------------------------------------------------------- loop integration

class Clock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += max(float(seconds), 0.0)
        time.sleep(0)  # Let the real worker run while advancing the fake clock.


class FakeWindow:
    def __init__(self, clock, size=(400, 200)):
        self.clock = clock
        self.size = size
        self.clicks = []
        self.activated = False

    def activate(self):
        self.activated = True

    def capture(self):
        self.clock.sleep(0.01)
        image = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)
        return LiveFrame(image, self.clock(), (0, 0, *self.size), (0, 0))

    def click(self, point, **kwargs):
        self.clicks.append(point)

    def window_rect(self):
        return (0, 0, *self.size)


class FakePolicy:
    def __init__(self):
        self.calls = []
        self.last_timing = {"backend": "fake", "status": "ok",
                            "total_seconds": 0.01}
        self.last_result = {"backend": "fake", "confidence": 1.0}
        self.closed = False
        self.closed_event = Event()

    def choose(self, state, actions, *, deadline=None):
        self.calls.append((state, actions, deadline))
        return actions[0]

    def close(self):
        self.closed = True
        self.closed_event.set()


def recognized_tile(label, index):
    return RecognizedTile(
        SegmentedTile(Box(20 + index * 40, 600, 50 + index * 40, 660),
                      np.zeros((4, 4, 3), dtype=np.uint8)),
        TilePrediction(label, 0.99, 0.0, 0.1, "upright"))


def visible_reading(hand):
    return SimpleNamespace(
        active_slots=SLOTS,
        hand=TileSequenceReading(
            "hand", tuple(recognized_tile(label, i)
                          for i, label in enumerate(hand)),
            drawn_index=len(hand) - 1 if len(hand) == 14 else None,
            issues=()),
        rivers={slot: TileSequenceReading(f"river_{slot}", ()) for slot in SLOTS},
        melds={slot: MeldReading(f"melds_{slot}", ()) for slot in SLOTS},
        dora=TileSequenceReading("dora", (recognized_tile("9s", 0),)),
        kita={slot: TileSequenceReading(f"kita_{slot}", ()) for slot in SLOTS},
        issues=(),
    )


def mock_reader(hand=HAND14, buttons=None, unknowns=(), window=None):
    visible = visible_reading(hand)
    number = lambda value: SimpleNamespace(value=value, glyphs=(), issues=(),
                                           complete=True)
    metadata = SimpleNamespace(
        active_slots=SLOTS,
        round=SimpleNamespace(round_wind="1z", hand_number=1, glyphs=(),
                              issues=(), complete=True),
        remaining_tiles=number(55), honba=number(0), riichi_sticks=number(0),
        scores={slot: number(35000) for slot in SLOTS},
        seats={slot: SimpleNamespace(wind=seat, dealer=index == 0, glyph=None,
                                     issues=(), complete=True)
               for index, (slot, seat) in enumerate(zip(SLOTS, SEATS))},
        issues=(), complete=True)
    reader = object.__new__(FrameObservationReader)
    reader.visible_reader = SimpleNamespace(read=lambda *_: visible)
    reader.metadata_reader = SimpleNamespace(read=lambda *_: metadata)
    reader.button_reader = FakeButtonReader(buttons, unknowns, window)
    return reader


class FakeButtonReader:
    """Mimics the patched ActionButtonReader cache + instance unknown sink."""

    def __init__(self, buttons=None, unknowns=(), window=None):
        self._buttons = dict(buttons or {})
        self._unknowns = list(unknowns)
        self.window = window
        self.last_buttons = {}
        self.unknown_sink = None

    def read(self, image, roi, **_kwargs):
        if self.window is not None and self.window.clicks:
            self._buttons = {}
            self._unknowns = []
        for hint, candidate, box in self._unknowns:
            if callable(self.unknown_sink):
                self.unknown_sink(hint, candidate, box)
        self.last_buttons = dict(self._buttons)
        return self.last_buttons


class FakeNotifier:
    def __init__(self):
        self.sent = []

    def notify_unknown_button(self, case_dir, message):
        self.sent.append((case_dir, message))
        return True


def small_roi():
    slots, variants, self_slot = default_layout(3)
    return ROIConfig(reference_size=[400, 200], player_count=3,
                     screen_slots=slots, layout_variants=variants,
                     self_slot=self_slot, kita=True, regions={})


def test_loop_runs_dry_run_decision_once(tmp_path):
    config = replace(LiveConfig.load(LIVE_CONFIG), artifacts_root=tmp_path, lifecycle_ui=None)
    clock = Clock()
    window = FakeWindow(clock)
    policy = FakePolicy()
    loop = LiveAgentLoop(window, config, execute=False, skip_menu=True,
                         once=True, session="test", sleep=clock.sleep,
                         now=clock, roi=small_roi(), reader=mock_reader(),
                         button_reader=None, policy=policy)
    assert loop.run() == 0
    assert not window.activated
    assert window.clicks == []
    assert len(policy.calls) == 1
    assert policy.closed_event.wait(1)
    decisions = (tmp_path / "test" / "decisions.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert len(decisions) == 1
    record = json.loads(decisions[0])
    assert record["mode"] == "dry_run"
    assert record["action"] == policy.calls[0][1][0].for_model()
    events = (tmp_path / "test" / "events.jsonl").read_text(
        encoding="utf-8")
    assert "session_start" in events
    assert "layout_locked" in events
    assert "decision_dry_run" in events


def test_game_start_requires_a_valid_agent_observation_and_signals_once(tmp_path):
    config=replace(LiveConfig.load(LIVE_CONFIG),artifacts_root=tmp_path,lifecycle_ui=None,
                   notify=replace(LiveConfig.load(LIVE_CONFIG).notify,enabled=False))
    clock=Clock()
    reader=mock_reader()
    loop=LiveAgentLoop(FakeWindow(clock),config,execute=False,skip_menu=True,
                      session='recognition',sleep=clock.sleep,now=clock,roi=small_roi(),
                      reader=reader,policy=FakePolicy())
    loop._active_slots=SLOTS
    loop._recognizing_start=True
    try:
        frame,index=loop._capture()
        observation=reader.read(frame.image,small_roi(),SLOTS,timestamp=clock(),frame_index=index)
        invalid=replace(observation,hand=(replace(observation.hand[0],tile=None),*observation.hand[1:]))
        assert not loop._signal_game_started(invalid,frame)
        assert not loop._game_started and loop._recognizing_start
        assert loop._signal_game_started(observation,frame)
        assert loop._game_started and not loop._recognizing_start
        assert loop._signal_game_started(observation,frame)
        events=[json.loads(line) for line in (loop.artifacts/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        assert sum(event['event']=='agent_game_started' for event in events)==1
    finally:
        loop.decision_worker.close()


def test_loop_dedupes_repeated_fingerprints(tmp_path):
    config = replace(LiveConfig.load(LIVE_CONFIG), artifacts_root=tmp_path, lifecycle_ui=None)
    clock = Clock()
    window = FakeWindow(clock)
    policy = FakePolicy()
    loop = LiveAgentLoop(window, config, execute=False, skip_menu=True,
                         once=False, session="test2", sleep=clock.sleep,
                         now=clock, roi=small_roi(), reader=mock_reader(),
                         policy=policy)
    loop.artifacts.mkdir(parents=True, exist_ok=True)
    loop._last_activity = clock()
    loop._active_slots = SLOTS
    # Commit the round through the tracker (stable_frames = 2).
    frame, index = loop._capture()
    loop._track(frame.image, small_roi(), SLOTS, frame.captured_at, index)
    frame, index = loop._capture()
    loop._process_frame(frame, index)
    loop._pending_decision["future"].result(timeout=1)
    assert len(policy.calls) == 1
    # Identical follow-up frame: same fingerprint, no second decision.
    frame, index = loop._capture()
    loop._process_frame(frame, index)
    assert len(policy.calls) == 1


# ------------------------------------------------------------------ CLI smoke

def test_cli_help_exits_zero():
    from majsoul.live.__main__ import main
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0


@pytest.mark.parametrize("command", ["doctor", "calibrate-menu", "observe", "run"])
def test_cli_subcommand_help_exits_zero(command):
    from majsoul.live.__main__ import main
    with pytest.raises(SystemExit) as caught:
        main([command, "--help"])
    assert caught.value.code == 0


# ---------------------------------------------------------- FeishuNotifier

def _notify_config(tmp_path):
    helper = tmp_path / "feishu_send.mjs"
    helper.write_text("// test helper", encoding="utf-8")
    return NotifyConfig(True, helper, "oc_test", "node")


def _notify_case(tmp_path):
    case = tmp_path / "case" / "abc123def456"
    write_image(case / "crop.png", np.zeros((8, 8, 3), dtype=np.uint8))
    write_image(case / "frame.png", np.zeros((16, 16, 3), dtype=np.uint8))
    return case


def test_notifier_sends_preflight_text_and_images(tmp_path):
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        stdout = (json.dumps({"credentialsReady": True, "targetReady": True,
                              "ready": True})
                  if "check-config" in command else json.dumps({"ok": True}))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    notifier = FeishuNotifier(_notify_config(tmp_path), runner=runner)
    case = _notify_case(tmp_path)
    assert notifier.notify_unknown_button(case, "unrecognized interactive button found") is True
    assert [command[2] for command in calls] == [
        "check-config", "text", "image", "image"]
    assert str(case / "crop.png") in calls[2]
    assert str(case / "frame.png") in calls[3]
    assert "oc_test" in calls[0] and "oc_test" in calls[1]


@pytest.mark.parametrize('proxy', [None, 'http://127.0.0.1:12345'])
def test_notifier_uses_configured_route_and_does_not_inherit_stale_proxies(tmp_path, monkeypatch, proxy):
    names = ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY',
             'http_proxy','https_proxy','all_proxy','no_proxy')
    for name in names:
        monkeypatch.setenv(name, 'retired-route')
    environments=[]
    def runner(command, **kwargs):
        environments.append(kwargs['env'])
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({'ready':True}), stderr='')
    notifier=FeishuNotifier(replace(_notify_config(tmp_path),proxy=proxy),runner=runner)
    assert notifier.notify_text('routing test')
    for environment in environments:
        routes={k.lower():v for k,v in environment.items() if k.lower() in
                ('http_proxy','https_proxy','all_proxy','no_proxy')}
        assert routes == ({'http_proxy':proxy,'https_proxy':proxy} if proxy else {})
    import os
    assert os.environ['HTTP_PROXY']=='retired-route'


def test_notifier_sends_canonical_review_crop_and_full_frame(tmp_path):
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({'ready':True}), stderr='')
    case = tmp_path/'case_000001'
    write_image(case/'crop.png', np.zeros((8,8,3), dtype=np.uint8))
    write_image(case/'full_frame.png', np.zeros((16,16,3), dtype=np.uint8))
    (case/'metadata.json').write_text(json.dumps({'status':'pending'}))
    assert FeishuNotifier(_notify_config(tmp_path),runner=runner).notify_unknown_button(case,'pending case')
    images = [call[3] for call in calls if call[2]=='image']
    assert images == [str(case/'crop.png'), str(case/'full_frame.png')]


def test_notifier_returns_false_when_not_ready(tmp_path):
    calls = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"ready": False}), stderr="")

    notifier = FeishuNotifier(_notify_config(tmp_path), runner=runner)
    assert notifier.notify_unknown_button(_notify_case(tmp_path), "m") is False
    assert [command[2] for command in calls] == ["check-config"]


def test_notifier_helper_failure_returns_false_and_logs(tmp_path):
    logged = []

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

    notifier = FeishuNotifier(_notify_config(tmp_path), runner=runner,
                              logger=logged.append)
    assert notifier.notify_unknown_button(_notify_case(tmp_path), "m") is False
    assert logged and "boom" in logged[0]


def test_notifier_disabled_never_runs_helper(tmp_path):
    def runner(command, **_kwargs):
        raise AssertionError("helper must not run when disabled")

    notifier = FeishuNotifier(NotifyConfig(False, None, None), runner=runner)
    assert notifier.notify_unknown_button(tmp_path, "m") is False


# ------------------------------------------- unknown-button phase policy

PASS_BOX = Box(300, 150, 380, 180)
UNKNOWN_CROP = np.full((30, 70, 3), 200, dtype=np.uint8)
cv2.rectangle(UNKNOWN_CROP, (10, 5), (20, 25), (20, 20, 20), -1)
cv2.rectangle(UNKNOWN_CROP, (30, 15), (55, 20), (20, 20, 20), -1)


def make_unknown_loop(tmp_path, *, hand=HAND14, buttons=None, unknowns=(),
                      execute=False, session="uk"):
    config = replace(LiveConfig.load(LIVE_CONFIG), artifacts_root=tmp_path, lifecycle_ui=None)
    clock = Clock()
    window = FakeWindow(clock)
    policy = FakePolicy()
    notifier = FakeNotifier()
    reader = mock_reader(hand=hand, buttons=buttons, unknowns=unknowns,
                         window=window)
    loop = LiveAgentLoop(window, config, execute=execute, skip_menu=True,
                         session=session, sleep=clock.sleep, now=clock,
                         roi=small_roi(), reader=reader, policy=policy,
                         notifier=notifier)
    loop.artifacts.mkdir(parents=True, exist_ok=True)
    loop._last_activity = clock()
    loop._active_slots = SLOTS
    return loop, window, policy, notifier


def drive_committed_frame(loop):
    frame, index = loop._capture()
    loop._track(frame.image, small_roi(), SLOTS, frame.captured_at, index)
    frame, index = loop._capture()
    loop._process_frame(frame, index)
    if loop._pending_decision is not None:
        loop._pending_decision["future"].result(timeout=1)
        frame, index = loop._capture()
        loop._process_frame(frame, index)


@pytest.mark.parametrize('early_proof,stable_final,expected', [(True, True, 'verified'),
                        (False, True, 'verified'), (True, False, 'verified')])
def test_discard_verification_accepts_exact_hand_difference_without_stable_table(
        tmp_path, early_proof, stable_final, expected):
    loop, window, _, _ = make_unknown_loop(tmp_path, execute=True)
    players = tuple(PlayerObservation(slot, seat, index == 0, None,
                         kita_count=0, score_available=False)
                    for index, (slot, seat) in enumerate(zip(SLOTS, SEATS)))
    before = FrameObservation(10., 1, SLOTS, '1z', 1, 0, 0, 50, players,
                              tuple(ObservedTile(label) for label in HAND14), 13,
                              (ObservedTile('2z'),))
    own = players[0]  # The discarded tile need not appear in the river.
    anchor = replace(before, timestamp=11., frame_index=2, remaining_tiles=49,
                     players=(own, *players[1:]), hand=before.hand[:-1], drawn_index=None)
    final = replace(anchor, timestamp=12., frame_index=3, remaining_tiles=48)
    original_samples = ([(anchor, False)] if early_proof else []) + [(final, stable_final)]
    samples = iter(original_samples)
    last = original_samples[-1]
    def track(*args):
        nonlocal last
        last = next(samples, last)
        return last[0], None, SimpleNamespace(stable=last[1], validation_issues=())
    loop._track = track
    loop.session_ui = None
    action = Action('discard:13:9s', 'discard', '9s', 13)
    try:
        assert loop._verify(action, before, 1) == expected
        record = json.loads((loop.artifacts/'frames/00001-verification.json').read_text())
        if expected == 'verified':
            assert record['evidence']['proof'] == 'exact_requested_physical_tile_removed'
            assert record['evidence']['river_evidence_required'] is False
            assert record['assessed_frame_index'] == (2 if early_proof else 3)
        assert window.clicks == []
    finally:
        loop.decision_worker.close()


def test_loop_unknown_button_response_clicks_recognized_pass(tmp_path):
    loop, window, policy, notifier = make_unknown_loop(
        tmp_path, hand=HAND13, buttons={"pass": PASS_BOX},
        unknowns=[("kan", UNKNOWN_CROP, PASS_BOX)], execute=True)
    drive_committed_frame(loop)

    assert window.clicks == [PASS_BOX.center]
    assert policy.calls == []
    assert notifier.sent == []  # Response closed before four stable unknown frames.
    assert not (tmp_path / "unknown_buttons").exists()
    events = (tmp_path / "uk" / "events.jsonl").read_text(encoding="utf-8")
    assert "unknown_button_response" in events
    assert '"window_resolved"' in events
    assert 'response_window_closed' in events
    assert '"verified"' not in events
    assert loop.decision_budget.started_at is None
    assert loop._actionable_since is None

def test_unknown_pass_is_not_repeated_while_its_acceptance_remains_unknown(tmp_path):
    loop, window, policy, _ = make_unknown_loop(
        tmp_path, hand=HAND13, buttons={"pass": PASS_BOX},
        unknowns=[("kan", UNKNOWN_CROP, PASS_BOX)], execute=True)
    # Model a client which ignores input and leaves the same response visible.
    loop.reader.button_reader.window = None
    drive_committed_frame(loop)
    drive_committed_frame(loop)
    assert window.clicks == [PASS_BOX.center]
    assert policy.calls == []
    records = [json.loads(line) for line in (loop.artifacts / "decisions.jsonl").read_text().splitlines()]
    assert len(records) == 1 and records[0]["mode"] == "verification_unknown"


def test_unknown_pass_refuses_changed_response_even_at_the_same_button_point(tmp_path):
    loop, window, _, _ = make_unknown_loop(tmp_path, hand=HAND13,
                                         buttons={"pass": PASS_BOX}, execute=True)
    frame, index = loop._capture()
    loop._track(frame.image, small_roi(), SLOTS, frame.captured_at, index)
    frame, index = loop._capture()
    tracked = loop._track(frame.image, small_roi(), SLOTS, frame.captured_at, index)
    changed = visible_reading(HAND13)
    changed.rivers["left"] = TileSequenceReading("river_left", (recognized_tile("1p", 0),))
    loop.reader.visible_reader.read = lambda *_: changed
    loop._pass_unknown_response(frame, {"pass": PASS_BOX}, tracked[0])
    assert window.clicks == []
    assert "response changed before pass" in (loop.artifacts / "events.jsonl").read_text()


def test_unknown_pass_refuses_expired_response_budget(tmp_path):
    loop, window, _, _ = make_unknown_loop(
        tmp_path, hand=HAND13, buttons={"pass": PASS_BOX},
        unknowns=[("kan", UNKNOWN_CROP, PASS_BOX)], execute=True)
    loop._actionable_since = loop.now()-20
    drive_committed_frame(loop)
    assert window.clicks == []
    assert "response deadline passed before pass" in (loop.artifacts / "events.jsonl").read_text()


def test_stop_after_pass_preserves_cancelled_execution_and_verification(tmp_path):
    loop, window, _, _ = make_unknown_loop(
        tmp_path, hand=HAND13, buttons={"pass": PASS_BOX},
        unknowns=[("kan", UNKNOWN_CROP, PASS_BOX)], execute=True)
    def click_then_stop(point, *, expected, hold_seconds):
        assert expected.window == window.window_rect()
        assert hold_seconds == loop.config.execution.mouse_press_seconds
        window.clicks.append(point)
        (loop.artifacts / "STOP").touch()
    window.click = click_then_stop
    with pytest.raises(KeyboardInterrupt):
        drive_committed_frame(loop)
    assert window.clicks == [PASS_BOX.center]
    record = json.loads((loop.artifacts / "decisions.jsonl").read_text())
    assert record["mode"] == "cancelled"
    proof = next((loop.artifacts / "frames").glob("*-verification.json"))
    verification = json.loads(proof.read_text())
    assert verification["status"] == "cancelled"
    assert verification["reason"] == "stop_requested_after_click"
    assert (loop.artifacts / verification["before_image"]).is_file()


def test_loop_unknown_button_response_without_pass_does_not_click(tmp_path):
    loop, window, policy, notifier = make_unknown_loop(
        tmp_path, hand=HAND13, buttons={"ron": PASS_BOX},
        unknowns=[("kan", UNKNOWN_CROP, PASS_BOX)], execute=True)
    drive_committed_frame(loop)

    assert window.clicks == []
    assert policy.calls == []
    assert notifier.sent == []
    for _ in range(3):
        frame, index = loop._capture()
        loop._process_frame(frame, index)
    assert len(notifier.sent) == 1
    events = (tmp_path / "uk" / "events.jsonl").read_text(encoding="utf-8")
    assert "unknown_button_no_pass" in events
    assert (tmp_path / "unknown_buttons" / 'case_000001' / 'metadata.json').is_file()


def test_loop_unknown_button_discard_phase_continues_decision(tmp_path):
    loop, window, policy, notifier = make_unknown_loop(
        tmp_path, hand=HAND14, buttons={},
        unknowns=[("riichi", UNKNOWN_CROP, PASS_BOX)], execute=True)
    drive_committed_frame(loop)

    assert len(policy.calls) == 1
    assert len(window.clicks) == 1  # normal discard execution, never "pass"
    assert notifier.sent == []  # One transient glyph cannot become a review case.
    events = (tmp_path / "uk" / "events.jsonl").read_text(encoding="utf-8")
    assert "unknown_button" in events
    assert "unknown_button_response" not in events


@pytest.mark.parametrize('hand,buttons,expected_count',[
    (HAND14,{},2),(HAND13,{'pass':PASS_BOX},1)])
def test_hand_double_click_does_not_double_click_response_buttons(tmp_path,hand,buttons,expected_count):
    loop,window,_,_=make_unknown_loop(tmp_path,hand=hand,buttons=buttons,
        unknowns=[('kan',UNKNOWN_CROP,PASS_BOX)],execute=True)
    counts=[]
    original=window.click
    def click(point,**kwargs):
        counts.append(kwargs.get('click_count',1))
        original(point,**kwargs)
    window.click=click
    drive_committed_frame(loop)
    assert counts==[expected_count]


def test_loop_unknown_button_response_dry_run_does_not_click(tmp_path):
    loop, window, policy, notifier = make_unknown_loop(
        tmp_path, hand=HAND13, buttons={"pass": PASS_BOX},
        unknowns=[("kan", UNKNOWN_CROP, PASS_BOX)], execute=False)
    drive_committed_frame(loop)

    assert window.clicks == []
    assert policy.calls == []
    assert notifier.sent == []
    for _ in range(3):
        frame, index = loop._capture()
        loop._process_frame(frame, index)
    assert len(notifier.sent) == 1
    assert not (tmp_path / "uk" / "decisions.jsonl").exists()
    case_dir = tmp_path / 'unknown_buttons' / 'case_000001'
    meta = json.loads((case_dir / 'metadata.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'pending' and meta['kind'] == 'action'
    assert meta['player_count'] == 3 and meta['mode'] == 'sanma'
    assert meta['context']['hint'] == 'kan' and meta['context']['hint_is_label'] is False
    assert meta['context']['control_state'] == 'HUMAN_REVIEW'
    assert len(meta['context']['samples']) == 4
    assert {'crop.png','full_frame.png','previous_frame.png','state.json','metadata.json'} == {p.name for p in case_dir.iterdir()}
    message = notifier.sent[0][1]
    assert 'phase=response' in message and 'hint=kan' in message
    assert 'click acceptance is verified separately' in message and '--kind action' in message
    # A new live session reuses the canonical case and delivery dedup ledger.
    reloaded, _, _, new_notifier = make_unknown_loop(tmp_path, hand=HAND13, buttons={'pass': PASS_BOX},
        unknowns=[('kan', UNKNOWN_CROP, PASS_BOX)], execute=False, session='uk2')
    drive_committed_frame(reloaded)
    for _ in range(3):
        frame, index = reloaded._capture()
        reloaded._process_frame(frame, index)
    assert new_notifier.sent == []
    assert len(list((tmp_path/'unknown_buttons').glob('case_*'))) == 1
    assert 'case_000001' in reloaded._notified
