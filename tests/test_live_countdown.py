from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.live.capture import roi_for_frame
from majsoul.live.config import LiveConfig
from majsoul.live.countdown import CountdownReader, CountdownReading, DecisionBudget


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reader():
    config = LiveConfig.load(ROOT/"configs/live.steam.yaml")
    return CountdownReader.load(config.countdown, ROOT, json.loads(config.rules.read_text()))


@pytest.mark.parametrize(("source", "base", "reserve", "mode"), [
    ("datasets/拔北.jpg", 1, 20, "base_plus_reserve"),
    ("datasets/自摸.png", 3, 19, "base_plus_reserve"),
    ("datasets/立直.jpg", 0, 18, "reserve_only"),
    ("artifacts/roi/action-survey/frame_615.png", 4, 5, "base_plus_reserve"),
    ("artifacts/roi/action-survey/frame_6765.png", 3, 20, "base_plus_reserve"),
    ("artifacts/live/workflow-audit/countdown-survey/video-frame-30.png", 0, 6, "reserve_only"),
])
@pytest.mark.parametrize("scale", [1., .75])
def test_reviewed_real_clock_formats_and_nearby_foreign_objects(reader, source, base, reserve, mode, scale):
    image = read_image(ROOT/source)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    roi = roi_for_frame(ROIConfig.load(ROOT/"configs/roi.steam.yaml"),
                        (image.shape[1], image.shape[0]), "stretch")
    reading = reader.read(image, roi)
    assert (reading.status, reading.mode, reading.base_seconds, reading.reserve_seconds) == ("complete", mode, base, reserve)
    assert reading.total_seconds == base+reserve
    assert all(g["label"] is not None and not g["clipped"] for g in reading.glyphs)
    assert reading.reader_version == reader.manifest["version_sha256"]


@pytest.mark.parametrize("source", ["datasets/暗杠.jpg", "artifacts/live/async-readonly-observe-20261008/frame.png",
                                     "artifacts/live/observe-collect-debug/frame.png"])
def test_non_clock_boards_or_results_never_yield_a_usable_budget(reader, source):
    image = read_image(ROOT/source)
    roi = ROIConfig.load(ROOT/"configs/roi.steam.yaml")
    assert reader.read(image, roi).total_seconds is None


@pytest.mark.parametrize('frame',[411,445])
def test_base_clock_is_not_vetoed_by_right_wall_gold_above_digit_row(reader,frame):
    image=read_image(ROOT/f'artifacts/live/acceptance-20261010/agent-recheck-172934/diagnostics/{frame:06d}.jpg')
    result=reader.read(image,ROIConfig.load(ROOT/'configs/roi.steam.yaml'))
    assert (result.status,result.mode,result.total_seconds)==('complete','base_only',5)


def test_unknown_gold_on_the_clock_baseline_still_blocks_base_only(reader,monkeypatch):
    roi=ROIConfig.load(ROOT/'configs/roi.steam.yaml')
    image=read_image(ROOT/'artifacts/live/acceptance-20261010/agent-recheck-172934/diagnostics/000411.jpg')
    pieces=list(reader.segment(image,roi))
    gold=next(p for p in pieces if p['color']=='gold')
    gold['box']={**gold['box'],'y1':1150,'y2':1220}
    monkeypatch.setattr(reader,'segment',lambda *_:tuple(pieces))
    assert reader.read(image,roi).status=='unknown'


def test_real_initial_eight_plus_twenty_clock_uses_the_existing_rounding_allowance(reader):
    image=read_image(ROOT/'artifacts/live/protected-20261013-round06/observer/000008.jpg')
    roi=ROIConfig.load(ROOT/'configs/roi.steam.yaml')
    result=reader.read(image,roi)
    assert (result.status,result.base_seconds,result.reserve_seconds)==('complete',8,20)
    b=DecisionBudget(turn_seconds=7,reserve_seconds=20,action_margin=1,
                     confirmations=1,quantization_margin=1)
    b.start(100)
    b.update(result,100)
    assert b.deadline(100,50)==126  # Displayed 28 minus rounding and action margins.
    exact=CountdownReader(replace(reader.config,quantization_margin_seconds=0),
                          turn_seconds=7,reserve_seconds=20)
    assert exact.read(image,roi).status=='unknown'


def test_clock_out_of_rules_range_or_partial_glyph_is_unknown(reader):
    image = read_image(ROOT/"datasets/拔北.jpg")
    roi = ROIConfig.load(ROOT/"configs/roi.steam.yaml")
    constrained = CountdownReader(reader.config, turn_seconds=7, reserve_seconds=5)
    assert constrained.read(image, roi).status == "unknown"
    # Crop through the last digit: the remaining visible prefix is not 2 seconds.
    roi.regions["countdown"][2] = .88
    assert reader.read(image, roi).status == "unknown"


def test_an_extra_leading_digit_cannot_be_silently_parsed_as_a_short_clock(reader, monkeypatch):
    image = read_image(ROOT/"datasets/立直.jpg")
    roi = ROIConfig.load(ROOT/"configs/roi.steam.yaml")
    glyphs = list(reader.read(image, roi).glyphs)
    first = glyphs[0]
    extra = {**first, "label": "0", "box": {**first["box"],
             "x1": first["box"]["x1"]-50, "x2": first["box"]["x1"]-10}}
    monkeypatch.setattr(reader, "segment", lambda *_: tuple([extra, *glyphs]))
    assert reader.read(image, roi).status == "unknown"


def reading(base, reserve):
    return CountdownReading("complete", "base_plus_reserve", base, reserve)


def budget():
    return DecisionBudget(turn_seconds=7, reserve_seconds=20, action_margin=1)


def test_unknown_clock_uses_five_second_limit_and_only_confirmed_visual_time_can_extend_it():
    b = DecisionBudget(turn_seconds=7, reserve_seconds=20, action_margin=1,
                       fallback_turn_seconds=5)
    b.start(100.)
    assert b.deadline(100., 30) == 104.
    b.update(reading(7, 20), 100.)
    assert b.pending and b.deadline(100., 30) == 104.
    b.update(reading(6, 20), 100.5)
    assert b.deadline(100.5, 30) == 124.5
    b.reset(); b.start(200.)
    b.update(CountdownReading('unknown'), 200.)
    assert b.deadline(200., 30) == 204.
    b.update(reading(5, 0), 200.25)
    b.update(reading(4, 0), 201.)
    assert b.deadline(201., 30) == 203.


@pytest.mark.parametrize('value', [0, 1, 8, True, float('nan')])
def test_invalid_fallback_turn_limit_is_rejected(value):
    with pytest.raises(ValueError):
        DecisionBudget(turn_seconds=7, reserve_seconds=20, action_margin=1,
                       fallback_turn_seconds=value)


def test_clock_requires_distinct_frames_then_can_use_only_observed_reserve():
    b = budget(); b.start(100.)
    b.update(reading(3, 19), 100.)
    assert b.pending
    assert b.deadline(100., 30) == 106.
    b.update(reading(3, 19), 100.)
    assert b.samples == 1
    b.update(reading(3, 19), 100.25)
    assert not b.pending
    assert b.deadline(100.25, 30) == 120.
    assert b.deadline(100.25, 10) == 110.25  # Independent observation-age cap.
    b.update(reading(3, 19), 100.5)
    assert b.deadline(100.5, 30) == 120.  # Repeated integer does not add time.


def test_lower_clock_shortens_the_same_window_and_larger_clock_cannot_refill_it():
    b = budget(); b.start(100.)
    b.update(reading(3, 19), 100.)
    b.update(reading(3, 19), 100.25)
    b.update(reading(2, 19), 100.5)
    assert b.deadline(100.5, 30) == 119.5
    b.update(reading(7, 20), 100.75)
    b.update(reading(7, 20), 101.)
    assert b.deadline(101., 30) == 119.5
    b.update(CountdownReading("unknown"), 102.)
    assert b.deadline(102., 30) == 119.5


def test_unconfirmed_larger_reading_and_gaps_do_not_erase_an_earlier_bound():
    b = budget(); b.start(100.)
    b.update(reading(0, 2), 100.)
    b.update(reading(7, 20), 100.25)
    b.update(reading(7, 20), 100.5)
    assert b.deadline(100.5, 30) <= 100.
    b.reset(); b.start(200.)
    b.update(reading(7, 20), 200.)
    b.update(reading(7, 20), 204.)
    assert b.pending
    assert b.deadline(204., 30) == 206.


@pytest.mark.parametrize("value", [None, CountdownReading("absent"), CountdownReading("unknown"),
                                  reading(9, 20), reading(1, 21)])
def test_unavailable_or_invalid_clock_uses_elapsed_base_time_without_reserve(value):
    b = budget(); b.start(100.)
    b.update(value, 102.)
    assert b.deadline(102., 30) == 106.
    b.update(value, 104.)
    assert b.deadline(104., 30) == 106.


def test_zero_clock_expires_immediately_and_new_window_must_reset_explicitly():
    b = budget(); b.start(100.)
    b.update(reading(0, 0), 100.)
    assert b.deadline(100., 30) < 100.
    b.start(200.)
    assert b.deadline(200., 30) < 100.
    b.reset(); b.start(200.)
    assert b.deadline(200., 30) == 206.


def test_clock_ticks_do_not_change_semantic_state_fingerprint():
    from test_live_loop import frame_observation
    before = replace(frame_observation(), countdown=reading(3, 19).to_dict())
    after = replace(before, countdown=reading(2, 19).to_dict())
    assert before.fingerprint == after.fingerprint
    assert after.to_dict()["countdown"]["total_seconds"] == 21


def test_live_window_uses_one_complete_clock_and_does_not_resubmit_on_ticks(tmp_path):
    from types import SimpleNamespace
    from test_live_decisions import SlowPolicy, make_loop, process, close
    policy = SlowPolicy()
    loop, window, clock = make_loop(tmp_path, policy)
    loop.config = replace(loop.config, decision=replace(loop.config.decision, max_decision_age=30))
    current = [reading(3, 19)]
    loop.reader.countdown_reader = SimpleNamespace(read=lambda *_: current[0])
    try:
        process(loop)
        assert policy.entered.wait(1)
        assert not loop.decision_budget.pending
        pending = loop._pending_decision
        assert pending["deadline"] > clock()+7
        assert pending["budget"]["source"] == "confirmed_visual_countdown"
        current[0] = reading(2, 19)
        process(loop)
        assert len(policy.calls) == 1
        assert not pending["future"].cancelled()
        assert pending["deadline"] <= 1020.
        clock.sleep(pending["deadline"]-clock()+.1)
        current[0] = CountdownReading("unknown")
        process(loop)
        assert pending["future"].cancelled()
        assert policy.cancelled.wait(1)
        assert not window.clicks
    finally:
        close(loop)


def test_fresh_preclick_zero_clock_rejects_an_otherwise_valid_model_answer(tmp_path):
    from types import SimpleNamespace
    from test_live_decisions import SlowPolicy, make_loop, process, close
    policy = SlowPolicy()
    loop, window, _ = make_loop(tmp_path, policy, execute=True)
    reads = []
    def read(*_):
        reads.append(True)
        return reading(0, 0) if len(reads) >= 5 else reading(3, 19)
    loop.reader.countdown_reader = SimpleNamespace(read=read)
    try:
        process(loop); process(loop); process(loop)
        assert policy.entered.wait(1)
        policy.release.set()
        loop._pending_decision["future"].result(timeout=1)
        process(loop)
        assert len(reads) >= 5
        assert window.clicks == []
        record = json.loads((loop.artifacts/"decisions.jsonl").read_text())
        assert record["mode"] == "preflight_abort"
        assert record["countdown"]["total_seconds"] == 22
        assert "deadline passed before click" in (loop.artifacts/"events.jsonl").read_text()
    finally:
        close(loop)


@pytest.mark.parametrize('kind', ['DRAW', 'DISCARD'])
def test_observed_own_boundary_releases_expired_clock_despite_missing_waiting_frame(tmp_path, kind):
    from types import SimpleNamespace
    from majsoul.state.schemas import GameEvent, GameEventType
    from test_live_decisions import SlowPolicy, make_loop, process, close
    policy = SlowPolicy()
    loop, window, clock = make_loop(tmp_path, policy)
    try:
        process(loop); process(loop)
        assert policy.entered.wait(1)
        pending = loop._pending_decision
        old_deadline = pending['deadline']
        clock.value = old_deadline + 5
        loop._advance_decision_window(SimpleNamespace(events=(GameEvent(
            GameEventType[kind], clock(), 20, player='bottom', tile='1z'),)))
        assert pending['future'].cancelled()
        assert policy.cancelled.wait(1)
        assert loop._last_request_key is None
        assert loop._actionable_since is None
        assert loop.decision_budget.started_at is None
        observation = replace(pending['observation'], timestamp=clock(),
                              countdown=reading(5, 0).to_dict())
        loop._window_deadline(observation)
        observation = replace(observation, timestamp=clock()+.5)
        assert loop._window_deadline(observation) > old_deadline
        assert loop.decision_budget.absolute_deadline == clock()+3
        assert window.clicks == []
    finally:
        close(loop)


@pytest.mark.parametrize(('kind', 'player', 'certainty'), [
    ('RESYNC', None, 'observed'), ('DRAW', 'top', 'observed'),
    ('DISCARD', 'bottom', 'inferred'),
])
def test_uncertainty_or_foreign_turn_never_renews_own_clock(tmp_path, kind, player, certainty):
    from types import SimpleNamespace
    from majsoul.state.schemas import GameEvent, GameEventType
    from test_live_decisions import SlowPolicy, make_loop, process, close
    policy = SlowPolicy()
    loop, _, clock = make_loop(tmp_path, policy)
    try:
        process(loop); process(loop)
        assert policy.entered.wait(1)
        pending = loop._pending_decision
        clock.value = pending['deadline']+5
        loop._advance_decision_window(SimpleNamespace(events=(GameEvent(
            GameEventType[kind], clock(), 20, player=player, certainty=certainty),)))
        assert loop._pending_decision is pending
        assert loop._window_deadline(replace(pending['observation'], timestamp=clock())) <= pending['deadline']
    finally:
        close(loop)
