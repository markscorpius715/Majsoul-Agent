"""Real Steam button evidence and the boundary between UI and tile evidence."""
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.live.buttons import ActionButtonReader, extract_button_candidates
from majsoul.live.capture import roi_for_frame
from majsoul.live.classifier import build_tile_classifier
from majsoul.live.config import LiveConfig
from majsoul.live.bridge import build_domain_state
from majsoul.rules import RuleEngine, Rules
from majsoul.perception import GlyphTemplateClassifier, VisionConfig
from majsoul.state import FrameObservationReader, MahjongStateTracker


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/live/unlocked-test-20261008/current-frame.jpg'


@pytest.mark.parametrize('scale', [1, .75, .5])
def test_independent_steam_nuki_frame_uses_complete_source_templates(scale):
    config = LiveConfig.load(ROOT / 'configs/live.steam.yaml')
    source = read_image(SOURCE)
    image = cv2.resize(source, None, fx=scale, fy=scale)
    roi = roi_for_frame(ROIConfig.load(config.roi), (image.shape[1], image.shape[0]), 'auto')
    reader = ActionButtonReader(config.buttons, config.button_templates)
    unknowns = []
    buttons = reader.read(image, roi, unknown_sink=lambda *args: unknowns.append(args))
    assert set(buttons) == {'nuki', 'pass'}
    assert not unknowns
    assert not any(candidate.clipped for candidate in reader.last_candidates)
    assert reader.last_occlusion_boxes
    assert config.buttons.maximum_distance == .25
    assert config.buttons.minimum_margin == .05


def test_candidate_extraction_reports_old_import_clipping():
    config = LiveConfig.load(ROOT / 'configs/live.steam.yaml')
    source = read_image(ROOT / 'datasets/拔北.jpg')
    roi = roi_for_frame(ROIConfig.load(config.roi), (source.shape[1], source.shape[0]), 'stretch')
    candidates = extract_button_candidates(source, roi, config.buttons)
    assert any(candidate.clipped for candidate in candidates)
    regions = dict(roi.regions)
    regions['action_buttons'] = [regions['action_buttons'][0], .65,
                                 regions['action_buttons'][2], regions['action_buttons'][3]]
    assert not any(candidate.clipped for candidate in
                   extract_button_candidates(source, replace(roi, regions=regions), config.buttons))


def test_unknown_or_unconfigured_ui_still_excludes_tile_evidence(tmp_path):
    config = LiveConfig.load(ROOT / 'configs/live.steam.yaml')
    reader = ActionButtonReader(config.buttons, tmp_path / 'no_templates')
    assert not reader.read(read_image(SOURCE), ROIConfig.load(config.roi))
    assert reader.last_candidates
    assert reader.last_occlusion_boxes
    # A subsequent frame with no UI must clear exclusion evidence.
    reader.read(read_image(ROOT / 'artifacts/live/observe-collect-debug/frame.png'),
                ROIConfig.load(config.roi))
    assert not reader.last_candidates
    assert not reader.last_occlusion_boxes


def test_steam_button_text_never_reaches_gold_tile_classifier(tmp_path):
    config = LiveConfig.load(ROOT / 'configs/live.steam.yaml')
    vision = VisionConfig.load(config.vision)
    roi = ROIConfig.load(config.roi)
    classifier = build_tile_classifier('cnn', vision, cnn_checkpoint=config.cnn_checkpoint,
                                      expected_metadata=config.cnn_expected)
    original = classifier.classify_many
    classified_regions = []
    def classify(candidates):
        candidates = tuple(candidates)
        classified_regions.extend(candidate.region for candidate in candidates)
        return original(candidates)
    classifier.classify_many = classify
    reader = FrameObservationReader(
        vision, classifier, GlyphTemplateClassifier(config.glyph_templates, vision),
        button_reader=ActionButtonReader(config.buttons, config.button_templates), read_scores=False)
    observation, visible = reader.read_with_visible(
        read_image(SOURCE), roi, ('bottom', 'top', 'right'), timestamp=1, frame_index=1)
    assert 'river_bottom' not in classified_regions
    assert 'kita_bottom' not in classified_regions
    assert 'melds_bottom' not in classified_regions
    assert len(observation.hand) == 14
    assert all(tile.tile is not None for tile in observation.hand)
    assert set(observation.action_buttons) == {'nuki', 'pass'}
    assert not visible.rivers['bottom'].complete
    assert not visible.kita['bottom'].complete
    assert observation.player('bottom').kita_count is None
    assert all(player.score is None for player in observation.players)
    assert observation.ui_occlusions
    # The detected row also intersects the upper edge of the configured meld
    # ROI. Preserve that blind spot even though this source has no melds.
    assert set(observation.unavailable_regions) == {'river_bottom', 'melds_bottom', 'kita_bottom'}
    assert observation.confidence['river_bottom'] == 0
    assert observation.confidence['kita_bottom'] == 0
    assert observation.to_dict()['unavailable_regions'] == observation.unavailable_regions
    # Pixel-level UI box jitter is diagnostic, not a new semantic decision.
    assert replace(observation, ui_occlusions=((1, 2, 3, 4),)).fingerprint == observation.fingerprint
    assert not observation.complete
    assert any('UI occlusion' in issue for issue in observation.issues)
    tracker = MahjongStateTracker.load(config.rules, stable_frames=2)
    for index in (1, 2, 3):
        update = tracker.update(replace(observation, timestamp=index, frame_index=index))
        if index == 1:
            assert not update.committed
            assert tracker.state is None
        else:
            assert update.stable
            assert not update.validation_issues
            assert all(event.type.value == 'RESYNC' for event in update.events)
            assert not tracker.state.history_complete
            assert tracker.state.player('bottom').kita_count is None
    # Full-table completeness remains false. The validated current hand can
    # nevertheless support ordinary legal discards without fabricated fields.
    assert tracker.state.decision_ready
    domain = build_domain_state(tracker.state, observation, visible,
        reader.button_reader.last_buttons, pending_riichi=False, window=(0,0,2582,1550))
    assert domain.observation.nuki[0] is None
    assert domain.observation.own_meld_count == 0
    assert set(domain.observation.unavailable_fields) == {'rivers_0', 'melds_0', 'nuki_0'}
    actions = RuleEngine(Rules.load(config.rules)).legal_actions(domain)
    discards = [action for action in actions if action.kind == 'discard']
    assert len(discards) == 14
    assert any(action.kind == 'nuki' for action in actions)
    assert all(action.features['effective_tile_count_is_upper_bound'] for action in discards)
    assert all(action.features['retained_dora'] is None for action in discards)
    assert all(action.features['hand_value_if_next_tsumo'] == {} for action in discards)
    # Run the production observation -> stable state -> bridge -> legal choices
    # -> asynchronous decision path on the saved source, without live input.
    from majsoul.live.loop import LiveAgentLoop
    from majsoul.live.capture import LiveFrame
    from test_live_loop import Clock, FakeWindow, FakePolicy, FakeNotifier
    image = read_image(SOURCE)
    clock = Clock()
    class SavedWindow(FakeWindow):
        def capture(self):
            clock.sleep(.01)
            return LiveFrame(image.copy(), clock(), (0, 0, 2582, 1550), (0, 0))
    window, policy = SavedWindow(clock, (2582, 1550)), FakePolicy()
    offline_config = replace(config, artifacts_root=tmp_path, lifecycle_ui=None)
    loop = LiveAgentLoop(window, offline_config, execute=False, skip_menu=True,
                         once=True, session='saved-ui', sleep=clock.sleep, now=clock,
                         reader=reader, policy=policy, notifier=FakeNotifier())
    assert loop.run() == 0
    assert not window.clicks and not window.activated
    assert len(policy.calls) == 1
    requested_state, requested_actions, _ = policy.calls[0]
    assert not requested_state.history_complete
    assert requested_state.observation.nuki[0] is None
    # Without a visible riichi threat only best-shanten discards reach the
    # model (decision.shanten_guard); kita stays available.
    discards = [a for a in requested_actions if a.kind == 'discard']
    assert discards and len({a.features['shanten'] for a in discards}) == 1
    assert any(a.kind == 'nuki' for a in requested_actions)
    assert policy.closed_event.wait(1)


def test_partial_river_keeps_uncovered_real_candidates_before_classification():
    from majsoul.perception import RiverReader, RiverSegmenter, extract_roi
    from majsoul.perception.schemas import TilePrediction
    config = VisionConfig.load(ROOT / 'configs/vision.yaml')
    roi = ROIConfig.load(ROOT / 'configs/roi.yaml')
    image = read_image(ROOT / 'artifacts/roi/review/frame_1590.png')
    crop = extract_roi(image, roi, 'river_bottom')
    candidates = RiverSegmenter(config).segment(crop).tiles
    assert len(candidates) == 10
    excluded = candidates[4].box
    captured = []
    class SpyClassifier:
        def classify_many(self, detections):
            captured.extend(detections)
            return tuple(TilePrediction('1p', .99, 0., 1., 'upright') for _ in detections)
    reading = RiverReader(config, SpyClassifier()).read(crop, excluded_boxes=(excluded,))
    assert 0 < len(captured) < len(candidates)
    assert candidates[4] not in captured
    assert all(tile.box.x2 < excluded.x1 or excluded.x2 < tile.box.x1 or
               tile.box.y2 < excluded.y1 or excluded.y2 < tile.box.y1 for tile in captured)
    assert len(reading.tiles) == len(captured)
    assert not reading.complete
    assert reading.issues == ('UI occlusion; region unavailable',)


@pytest.mark.parametrize('frame', ['00000020.jpg', '00000100.jpg'])
def test_own_north_tiles_cut_by_the_button_roi_are_not_a_ui_row(frame):
    """protected-20261018-151224: our second North tile reached into the
    action_buttons ROI; its clipped edge used to hide river_bottom/kita_bottom
    for the rest of the hand and blocked the rejoin Agent's handback."""
    config = LiveConfig.load(ROOT / 'configs/live.steam.yaml')
    image = read_image(ROOT / 'artifacts/live/protected-20261018-151224/client-002/frames' / frame)
    roi = roi_for_frame(ROIConfig.load(config.roi), (image.shape[1], image.shape[0]), 'auto')
    reader = ActionButtonReader(config.buttons, config.button_templates)
    unknowns = []
    assert reader.read(image, roi, unknown_sink=lambda *args: unknowns.append(args)) == {}
    assert reader.last_candidates and all(c.clipped for c in reader.last_candidates)
    assert {p['rejection'] for p in reader.last_predictions.values()} == {'ROI_edge_clutter'}
    assert not unknowns
    assert not reader.last_occlusion_boxes


@pytest.mark.parametrize('frame', ['emergency-000258-before.png', 'emergency-000312-before.png'])
def test_ron_button_reaching_above_the_row_is_read_in_full(frame):
    """live-20261018-164552: the 和 glyph reaches about 5 px above action_buttons;
    the cut candidate was rejected and the fallback declined the ron twice."""
    config = LiveConfig.load(ROOT / 'configs/live.steam.yaml')
    image = read_image(ROOT / 'artifacts/live/live-20261018-164552-978100/agent/frames' / frame)
    roi = roi_for_frame(ROIConfig.load(config.roi), (image.shape[1], image.shape[0]), 'auto')
    reader = ActionButtonReader(config.buttons, config.button_templates)
    unknowns = []
    buttons = reader.read(image, roi, unknown_sink=lambda *args: unknowns.append(args))
    assert set(buttons) == {'pon', 'ron', 'pass'} and not unknowns
    assert not any(c.clipped for c in reader.last_candidates)
