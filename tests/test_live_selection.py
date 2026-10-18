"""Input-bound selection evidence; native yonma images test shared UI only."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json

import cv2
import numpy as np
import pytest

from majsoul.calibration.roi_config import ROIConfig
from majsoul.domain import Action, Box, State
from majsoul.live.bridge import build_domain_state, infer_phase
from majsoul.live.capture import LiveFrame
from majsoul.live.config import LiveConfig
from majsoul.live.loop import LiveAgentLoop, LiveError, action_point
from majsoul.live.selection import RiichiSelection, RiichiSelectionConfig
from majsoul.perception import VisionConfig
from majsoul.perception.hand_reader import HandReader
from majsoul.perception.roi_extractor import extract_roi
from majsoul.state.schemas import FrameObservation, ObservedTile, PlayerObservation
from conftest import observation
from test_live_bridge import make_state, make_observation, visible_reading
from test_live_loop import Clock, FakeWindow, FakePolicy, FakeNotifier, mock_reader, small_roi
from test_live_verification import discard_case, check, ui_hidden

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/live/workflow-audit/selection-source-20261009/riichi-probe'
# Sanma fixture: source yonma tiles are never fed to the rule engine.
HAND = ('6s', '7s', '8s', '1p', '2p', '3p', '5p', '6p', '7p', '8p',
        '4z', '4z', '5z', '4p')


def capture(observation, visible, *, enabled=None, value=222, window=(0, 0, 1000, 800)):
    image = np.full((window[3], window[2], 3), value, np.uint8)
    if enabled is not None:
        image[:] = 133
        for index in enabled:
            b = visible.hand.tiles[index].detection.box
            image[int(b.y1):int(b.y2), int(b.x1):int(b.x2)] = 222
    return LiveFrame(image, observation.timestamp, window, (0, 0))


def baseline():
    obs = make_observation(HAND, action_buttons=('riichi',))
    visible = visible_reading(HAND)
    frame = capture(obs, visible)
    reader = RiichiSelection(RiichiSelectionConfig())
    reader.begin(frame, obs, visible, 16.)
    after = replace(obs, timestamp=11., frame_index=101, action_buttons=())
    return reader, obs, visible, after


def test_selection_requires_sent_intent_and_two_brightness_groups():
    reader, _, visible, after = baseline()
    frame = capture(after, visible, enabled=(12,))
    assert RiichiSelection(reader.config).read(frame, after, visible)[0] is None
    mask, evidence = reader.read(frame, after, visible)
    assert mask == (12,)
    assert evidence['before_frame_index'] == 100
    for enabled in ((), tuple(range(14))):
        assert reader.read(capture(after, visible, enabled=enabled), after, visible)[0] is None
    ambiguous = capture(after, visible, enabled=(12,))
    b = visible.hand.tiles[0].detection.box
    ambiguous.image[int(b.y1):int(b.y2), int(b.x1):int(b.x2)] = 180
    assert reader.read(ambiguous, after, visible)[1]['reason'] == 'selection_brightness_ambiguous'
    reader.reset()
    assert reader.deadline is None and reader.read(frame, after, visible)[0] is None


@pytest.mark.parametrize('change', [dict(timestamp=10.), dict(timestamp=16.),
    dict(frame_index=100), dict(hand_number=2),
    dict(action_buttons=('riichi',)), dict(issues=('hand: unknown',)),
    dict(active_slots=('bottom', 'left', 'right'))])
def test_selection_rejects_late_cross_round_or_changed_table(change):
    reader, _, visible, after = baseline()
    after = replace(after, **change)
    assert reader.read(capture(after, visible, enabled=(12,)), after, visible)[0] is None


def test_selection_checks_capture_labels_public_fields_and_whole_target_boxes():
    reader, _, visible, after = baseline()
    frame = capture(after, visible, enabled=(12,))
    assert reader.read(replace(frame, captured_at=12.), after, visible)[0] is None
    assert reader.read(replace(frame, window=(1, 0, 1000, 800)), after, visible)[0] is None
    changed = visible_reading((*HAND[:-1], '9p'))
    assert reader.read(frame, after, changed)[0] is None
    player = replace(after.players[1], river=(ObservedTile('1z'),))
    assert reader.read(frame, replace(after, players=(after.players[0], player, after.players[2]),
                                    remaining_tiles=None, dora_indicators=(),
                                    issues=('river_left: animation',)), visible)[0] == (12,)
    tile = visible.hand.tiles[0]
    enlarged = replace(tile, detection=replace(tile.detection, box=Box(0, 590, 90, 670)))
    moved = replace(visible, hand=replace(visible.hand, tiles=(enlarged, *visible.hand.tiles[1:])))
    assert reader.read(frame, after, moved)[1]['reason'] == 'selection_hand_geometry_changed'
    with pytest.raises(ValueError):
        reader.begin(capture(after, visible, value=133), replace(after, action_buttons=('riichi',)), visible, 16.)
    with pytest.raises(ValueError):
        reader.begin(capture(after, changed), replace(after, action_buttons=('riichi',)), changed, 16.)


@pytest.mark.parametrize('config', [{'enabled_ratio': 1}, {'disabled_ratio': [0.8, 1]},
    {'bright_minimum': True}, {'target_tolerance': '0.02'}, {'brightness_percentile': float('nan')},
    {'enabled_ratio': [1]}, {'unexpected': 1}])
def test_selection_configuration_rejects_malformed_thresholds(config):
    with pytest.raises(ValueError):
        RiichiSelectionConfig.from_dict(config)


def test_client_mask_intersects_tenpai_and_binds_the_exact_hand_target(engine):
    obs = observation('123456789p123s11z', phase='riichi_select', riichi_discard_indices=(12,))
    state = State(obs, True, ((), (), ()))
    actions = engine.legal_actions(state)
    assert len(actions) == 1
    assert actions[0].kind == 'riichi' and actions[0].hand_index == 12
    assert actions[0].button is None and actions[0].features['shanten'] == 0
    assert action_point(actions[0], state) == obs.hand[12].box.center
    obs.riichi = (None, False, False)
    assert [a.id for a in engine.legal_actions(state)] == [actions[0].id]
    obs.riichi = (True, False, False)
    assert engine.legal_actions(state) == []


@pytest.mark.parametrize('mask', [None, (), (True,), (-1,), (14,), (12, 12), ('12',), [12]])
def test_unproved_or_malformed_mask_never_authorizes_selection(engine, mask):
    obs = observation('123456789p123s11z', phase='riichi_select', riichi_discard_indices=mask)
    assert engine.legal_actions(State(obs, True, ((), (), ()))) == []


def test_bridge_never_infers_selection_from_a_disappearing_button():
    obs, state = make_observation(HAND, action_buttons=()), make_state(HAND)
    assert infer_phase(obs, state, pending_riichi=True) == 'unknown'
    proven = replace(obs, riichi_discard_indices=(12,))
    assert infer_phase(proven, state, pending_riichi=True) == 'riichi_select'
    domain = build_domain_state(state, proven, visible_reading(HAND), {}, pending_riichi=True, window=(0, 0, 1000, 800))
    assert domain.for_model()['riichi_discard_indices'] == (12,)
    assert replace(proven, selection_evidence={'ratios': [1.0]}).fingerprint == proven.fingerprint
    assert obs.fingerprint != proven.fingerprint


def test_riichi_button_alone_never_proves_a_declaration_even_with_horizontal_river():
    action, before, after = discard_case()
    own = replace(after.players[0], river=(*after.players[0].river[:-1],
                  replace(after.players[0].river[-1], orientation='horizontal')))
    after = replace(after, players=(own, *after.players[1:]))
    assert check(Action('button:riichi', 'riichi', button='riichi'), before, after).reason == 'riichi_discard_target_unbound'
    bound = replace(action, kind='riichi', id='riichi_discard:13:9s')
    assert check(bound, before, after).status == 'verified'
    assert check(replace(bound, hand_index=0), before, after).status == 'unknown'
    # Public blind spots can retain exact discard proof outside the old UI.
    assert check(bound, ui_hidden(before, ('river_left',)), ui_hidden(after, ('river_left',))).status == 'verified'
    assert check(Action('button:riichi', 'riichi', button='riichi'), ui_hidden(before), ui_hidden(after)).status == 'unknown'


def test_native_yonma_selection_with_real_gold_classifier(gold_tile_classifier):
    expected_hashes = {
        106440: 'b7783eb8c45106293aa67d9b0193d0c0ab2a8937683359b37ea3b9d41e0a8a0d',
        106448: '3770178216f290e404d2a42cc72e66dd61804dd170e02ec7a804c45429c56210',
        106450: 'c72bf3f7c9033c2d2bbac1b7efd62b4185ac281856ce6252068d6d5dfb14ed83',
        106458: 'b16aa4ac4ae90a1401c8c0a8d2d151e7550ddbe822118e9745925e46e8238663'}
    config = LiveConfig.load(ROOT / 'configs/live.steam.yaml')
    roi = ROIConfig.load(config.roi)
    vision = VisionConfig.load(config.vision)
    assert vision.value_min == 130
    service = RiichiSelection(config.riichi_selection)
    labels = ('6m','7m','8m','1p','2p','3p','5p','6p','7p','8p','4s','4s','5z','4p')
    slots = ('bottom', 'top', 'left', 'right')
    players = tuple(PlayerObservation(s, f'{i+1}z', i == 0, None, kita_count=0, score_available=False)
                    for i, s in enumerate(slots))
    for index, expected_hash in expected_hashes.items():
        path = SOURCE / f'{index:07d}.png'
        assert sha256(path.read_bytes()).hexdigest() == expected_hash
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        # Reviewed native historical source projection; production ROI is unchanged.
        source_roi = replace(roi, reference_size=[image.shape[1], image.shape[0]])
        hand = HandReader(vision, gold_tile_classifier).read(extract_roi(image, source_roi, 'self_hand'))
        assert hand.complete and hand.labels == labels
        obs = FrameObservation(index/30, index, slots, '1z', 1, 0, 0, 50, players,
                               tuple(ObservedTile(t.prediction.tile) for t in hand.tiles),
                               hand.drawn_index, (ObservedTile('1z'),),
                               action_buttons=('riichi',) if index == 106440 else ())
        frame = LiveFrame(image, obs.timestamp, (0, 0, image.shape[1], image.shape[0]), (0, 0))
        visible = SimpleNamespace(hand=hand)
        if index == 106440:
            service.begin(frame, obs, visible, obs.timestamp+6)
        else:
            mask, evidence = service.read(frame, obs, visible)
            assert mask == (12,) and evidence['source'] == 'client_bright_dim_partition'


class RiichiPolicy(FakePolicy):
    """Declares riichi together with the tile at index 12 in one choice."""
    def choose(self, state, actions, *, deadline=None):
        self.calls.append((state, actions, deadline))
        return next(a for a in actions if a.kind == 'riichi_declare' and a.hand_index == 12)


@pytest.mark.parametrize('acknowledged', [True, False])
def test_riichi_is_one_decision_with_two_clicks_and_no_third_input(tmp_path, acknowledged):
    """live-20261018-164552 f330: the old flow asked the model again for the
    tile and the selection expired fatally. Now the tile is chosen with the
    declaration and selected without a second model call; a lost
    acknowledgement is not fatal and never causes a third input."""
    config = replace(LiveConfig.load(ROOT / 'configs/live.steam.yaml'), artifacts_root=tmp_path,
                     lifecycle_ui=None, countdown=None)
    if not acknowledged:
        config = replace(config, execution=replace(config.execution, verification_timeout=0.8))
    clock = Clock()

    class SelectionWindow(FakeWindow):
        after_count = 0
        def capture(self):
            frame = super().capture()
            frame.image[:] = 222
            if len(self.clicks) == 1:
                frame.image[:] = 133
                frame.image[600:660, 20+12*40:50+12*40] = 222
            if len(self.clicks) >= 2:
                self.after_count += 1
                if not acknowledged:
                    frame.image[:] = 133
                    for index in (0, 12):
                        frame.image[600:660, 20+index*40:50+index*40] = 222
                if self.after_count > (6 if acknowledged else 40):
                    raise KeyboardInterrupt
            if clock() > 1020:
                raise KeyboardInterrupt
            return frame

    window = SelectionWindow(clock, (1000, 800))
    reader = mock_reader(HAND, buttons={'riichi': Box(700, 500, 800, 550)}, window=window)
    original = reader.read_with_visible
    def read(*args, **kwargs):
        obs, visible = original(*args, **kwargs)
        if len(window.clicks) >= 2 and acknowledged:
            own = replace(obs.players[0], river=(ObservedTile('5z', 'horizontal'),))
            obs = replace(obs, hand=(*obs.hand[:12], *obs.hand[13:]), drawn_index=None,
                          players=(own, *obs.players[1:]))
            visible = visible_reading((*HAND[:12], *HAND[13:]))
        return obs, visible
    reader.read_with_visible = read
    policy = RiichiPolicy()
    loop = LiveAgentLoop(window, config, execute=True, skip_menu=True, session='selection',
                         now=clock, sleep=clock.sleep, roi=replace(small_roi(), reference_size=[1000, 800]),
                         reader=reader, policy=policy, notifier=FakeNotifier())
    assert loop.run() == 130
    records = [json.loads(line) for line in (tmp_path/'selection/decisions.jsonl').read_text().splitlines()]
    assert len(policy.calls) == 1
    assert len(window.clicks) == len(records) == 2
    assert records[0]['action']['id'] == 'riichi:12:5z'
    assert [r['mode'] for r in records] == ['selection_open', 'verified' if acknowledged else 'verification_unknown']
    assert records[1]['action']['hand_index'] == 12 and records[1]['action']['tile'] == '5z'
    assert records[1]['backend'] == {'backend': 'riichi_plan', 'action_id': records[1]['action']['id']}
    # The tile selection may use the hand's reserve beyond the decision budget.
    assert records[1]['deadline'] >= records[0]['deadline']
    assert window.clicks[0] == (750, 525) and window.clicks[1] == (515, 630)
    if acknowledged:
        assert loop._pending_riichi is False and loop.riichi_selection.deadline is None
        assert loop._own_riichi_round == '1z:1:0'
    else:
        # The unconfirmed selection expires gracefully instead of stopping the Agent.
        events = (tmp_path/'selection/events.jsonl').read_text()
        assert 'riichi_selection_expired' in events and 'fatal_error' not in events
        assert loop._pending_riichi is False


def test_unresolved_kan_is_not_fatal_and_never_clicks_into_the_same_window(tmp_path):
    config = replace(LiveConfig.load(ROOT / 'configs/live.steam.yaml'), artifacts_root=tmp_path,
                     lifecycle_ui=None, countdown=None)
    clock = Clock()

    class LimitedWindow(FakeWindow):
        def capture(self):
            if clock() > 1015:
                raise KeyboardInterrupt
            return super().capture()

    window = LimitedWindow(clock)
    policy = FakePolicy()
    loop = LiveAgentLoop(window, config, execute=True, skip_menu=True, session='unresolved-kan',
                         now=clock, sleep=clock.sleep, roi=small_roi(), policy=policy,
                         reader=mock_reader(HAND, buttons={'kan': Box(250, 100, 300, 150)}, window=window),
                         notifier=FakeNotifier())
    assert loop.run() == 130
    records = [json.loads(line) for line in (tmp_path/'unresolved-kan/decisions.jsonl').read_text().splitlines()]
    assert len(policy.calls) == len(window.clicks) == len(records) == 1
    assert records[0]['action']['kind'] == 'kan' and records[0]['mode'] == 'verification_unknown'
    events = (tmp_path/'unresolved-kan/events.jsonl').read_text()
    assert 'kan_unresolved' in events and 'fatal_error' not in events
