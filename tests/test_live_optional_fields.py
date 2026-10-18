"""Noncritical failures recover independently; local legality stays strict."""
from dataclasses import replace
import json
from pathlib import Path
from threading import Event
from time import monotonic

import cv2
import pytest

from majsoul.calibration.roi_config import ROIConfig
from majsoul.live.riichi import RiichiConfig, RiichiDetector
from majsoul.state import MahjongStateTracker, StateValidator
from majsoul.state.decision import decision_observation
from majsoul.live.bridge import build_domain_state
from majsoul.domain import Action
from majsoul.live.execution_verification import assess_execution_action
from test_live_single_frame import saved_observations
from test_live_loop import visible_reading
from test_live_decisions import make_loop, process, close, SlowPolicy
from test_live_verification import snapshot, advance, FILLER, tiles
from test_numeric_ocr import reader as make_ocr, patch

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / 'artifacts/live/acceptance-20261010/agent-recheck-154019'


@pytest.mark.parametrize('field', ['remaining_tiles', 'honba', 'riichi_sticks', 'dora'])
def test_optional_field_enters_prompt_as_unknown_then_recovers_without_pausing(tmp_path, field):
    loop, _, clock = make_loop(tmp_path, SlowPolicy())
    original = loop.reader.read_with_visible
    failed = [True]
    def read(*args, **kwargs):
        obs, visible = original(*args, **kwargs)
        if not failed[0]:
            return obs, visible
        values = {field: None} if field != 'dora' else {}
        return replace(obs, issues=(field + ': rejected',), **values), visible
    loop.reader.read_with_visible = read
    try:
        process(loop)
        assert loop.policy.entered.wait(1)
        pending = loop._pending_decision
        payload = loop.policy.calls[0][0].for_model()
        name = 'dora_indicators' if field == 'dora' else field
        assert name in payload['unavailable_fields']
        assert payload[name] == (() if field == 'dora' else None)
        clock.sleep(.1)
        failed[0] = False
        process(loop)
        assert not loop._paused and loop._issue_since is None
        assert loop._pending_decision is pending and not pending['future'].cancelled()
        # Recovery must not mutate the frozen request already in flight.
        assert loop.policy.calls[0][0].for_model() == payload
        log = [json.loads(s) for s in (loop.artifacts / 'events.jsonl').read_text().splitlines()]
        assert any(field in e.get('restored', ()) for e in log)
    finally:
        close(loop)


def test_real_bad_09_frame_is_decision_ready_with_unknown_remaining_count():
    obs = saved_observations(LIVE / 'observations.jsonl')[445]
    assert 'remaining_tiles: numeric_ocr_rejected' in obs.issues
    projected = decision_observation(obs)
    tracker = MahjongStateTracker(StateValidator())
    state = tracker.decision_snapshot(projected)
    domain = build_domain_state(state, projected, visible_reading(tuple(t.tile for t in obs.hand)),
                                {}, pending_riichi=False, window=(0,0,2582,1550))
    assert state.decision_ready and tracker.state is None
    assert domain.for_model()['remaining_tiles'] is None
    assert 'remaining_tiles' in domain.for_model()['unavailable_fields']
    assert tracker.validator.validate_observation(projected, allow_ui_occlusion=True)


def test_second_row_sideways_false_positive_cannot_override_empty_stick_roi():
    samples = saved_observations(LIVE / 'observations.jsonl')
    roi = ROIConfig.load(ROOT / 'configs/roi.steam.yaml')
    detector = RiichiDetector(RiichiConfig(enabled=True))
    # Use the actual immutable empty patches saved at the start of this match.
    for slot in samples[1241].active_slots:
        path = LIVE / 'riichi-baseline' / (slot + '.png')
        patch_image = cv2.imread(str(path))
        assert patch_image is not None, path
        detector.baselines[slot] = patch_image.astype('float32') / 255
    image = cv2.imread(str(LIVE / 'frames/2d3a33b771e248ee8e029399d0e5054e-request.png'))
    detector.geometry = (image.shape[:2], tuple((s, tuple(roi.regions['riichi_'+s]))
                                              for s in samples[1241].active_slots))
    obs = detector.apply(image, roi, samples[1241])
    assert obs.player('top').riichi is False
    assert obs.player('top').riichi_evidence['sideways_candidate'] is True
    tracker = MahjongStateTracker(StateValidator())
    assert tracker.decision_snapshot(decision_observation(obs)).player('top').riichi is False


def test_nuki_increment_and_restored_count_do_not_require_exact_replacement_faces():
    before = snapshot(FILLER + ('1s', '2s', '3s', '4z'), kita=1, buttons=('nuki',))
    after = advance(before, hand=FILLER + ('2s', '2s', '3s', '6s'), kita=2)
    result = assess_execution_action(Action('button:nuki', 'nuki', button='nuki'), before, after, 'bottom')
    assert result.status == 'verified' and result.evidence['replacement_tile'] is None
    assert result.evidence['replacement_proof'] == 'concealed_count_restored_after_kita_increment'
    pending = replace(after, hand=after.hand[:-1])
    assert assess_execution_action(Action('button:nuki', 'nuki', button='nuki'), before, pending, 'bottom').status == 'unknown'


def test_async_numeric_ocr_is_bounded_and_never_applies_a_previous_crop(tmp_path):
    ocr, _ = make_ocr(tmp_path)
    ocr.config = replace(ocr.config, asynchronous=True)
    entered, release = Event(), Event()
    def engine(image, **kwargs):
        from types import SimpleNamespace
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(txts=['09'], scores=[.99])
    ocr.engine = engine
    ocr.enable_async()
    image = patch()
    try:
        start = monotonic()
        assert ocr.read_for_field(image, 'remaining_tiles')[2] == 'numeric_ocr_pending'
        assert monotonic() - start < .2 and entered.wait(1)
        changed = image.copy()
        changed[0,0] = 10
        for _ in range(20):
            assert ocr.read_for_field(changed, 'remaining_tiles')[0] is None
        with ocr._condition:
            assert len(ocr._pending) == 1
        release.set()
        deadline = monotonic() + 2
        while monotonic() < deadline:
            with ocr._condition:
                if ocr._results.get('remaining_tiles'):
                    break
                ocr._condition.wait(.01)
        assert ocr.read_for_field(changed, 'remaining_tiles') == ('09', .99, None)
        assert ocr.read_for_field(image, 'remaining_tiles')[0] is None
    finally:
        release.set()
        ocr.close()
