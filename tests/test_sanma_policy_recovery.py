"""Regression for counter-OCR stalls and missing riichi/nuki model context."""
from dataclasses import replace
import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

from majsoul.calibration.roi_config import ROIConfig
from majsoul.domain import Box
from majsoul.jev import JevPolicy, JevSettings
from majsoul.strategy import DeepSeekPolicy, ModelSettings
from majsoul.live.emergency import EmergencyConfig, TurnWatchdog
from majsoul.live.riichi import RiichiConfig, RiichiDetector
from majsoul.state import MahjongStateTracker
from test_live_single_frame import saved_observations
from test_live_decisions import make_loop, close, process, SlowPolicy
from test_live_loop import visible_reading
from tools.live_test_supervisor import AcceptanceGuard, AcceptanceStopped

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'artifacts/live/acceptance-20261010/agent-recheck-001254'


@pytest.fixture(scope='module')
def samples():
    return saved_observations(DIRECTORY / 'observations.jsonl')


def test_both_wire_payloads_use_same_rule_card_and_active_room(engine, state):
    state.rules_context = dict(engine.rules.data, nuki_rinshan_yaku=False)
    actions = engine.legal_actions(state)
    jev = JevPolicy(JevSettings('test'), httpx.AsyncClient())
    ds = DeepSeekPolicy(ModelSettings('test'), httpx.AsyncClient())
    try:
        a, b = jev.payload(state, actions), ds.payload(state, actions)
        card = a['questions']['action']['instructions']
        assert b['messages'][0]['content'].startswith(card)
        assert 'ONLY among candidates with\nthe SAME shanten' in card
        assert 'Chiitoitsu is fixed 25' in card and 'fourth payer' in card
        assert json.loads(card.split('unknown):\n')[-1])['nuki_rinshan_yaku'] is False
        assert json.loads(b['messages'][1]['content'])['state']['rules_context']['nuki_rinshan_yaku'] is False
        assert set(a['questions']['action']['criteria']) == {x.id for x in actions}
    finally:
        jev.close()
        ds.close()


def test_nuki_has_explicit_effects_and_keeps_other_candidates(engine, state):
    state.observation.hand = state.observation.hand[:-1] + (replace(state.observation.hand[-1], tile='4z'),)
    state.observation.buttons = {'nuki': Box(1, 1, 2, 2)}
    actions = engine.legal_actions(state)
    nuki = next(a for a in actions if a.kind == 'nuki')
    assert nuki.features['replacement_draw'] is True
    assert nuki.features['bonus_han_on_legal_win'] == 1
    assert nuki.features['breaks_menzen'] is False and nuki.features['is_yaku'] is False
    assert any(a.kind == 'discard' and a.tile == '4z' for a in actions)


def test_real_stick_attribution_retests_and_survives_hand_restart(samples):
    roi = ROIConfig.load(ROOT / 'configs/roi.steam.yaml')
    detector = RiichiDetector(RiichiConfig(enabled=True))
    initial = cv2.imread(str(DIRECTORY / 'diagnostics/000048.jpg'))
    current = cv2.imread(str(DIRECTORY / 'diagnostics/000897.jpg'))
    baseline = detector.apply(initial, roi, samples[48])
    assert all(p.riichi is False for p in baseline.players)
    frozen = {k: v.copy() for k, v in detector.baselines.items()}
    for i in range(3):
        obs = replace(samples[897], timestamp=100+i*.3, frame_index=100+i)
        result = detector.apply(current, roi, obs)
        assert result.player('left').riichi is (True if i == 2 else None), result.player('left').riichi_evidence
        assert result.player('right').riichi is (True if i == 2 else None), result.player('right').riichi_evidence
    assert result.player('bottom').riichi is False
    tracker = MahjongStateTracker.load(ROOT / 'configs/sanma.json')
    # Repair only the fixture's known rejected counter to test attribution.
    ready = replace(result, riichi_sticks=2,
                    issues=tuple(i for i in result.issues if not i.startswith('riichi_sticks:')))
    from majsoul.state.decision import decision_observation
    state = tracker.decision_snapshot(decision_observation(ready))
    assert state.player('left').riichi is True and state.player('right').riichi is True
    detector.new_hand()
    assert not detector.latched
    assert all(np.array_equal(detector.baselines[k], v) for k, v in frozen.items())
    empty = detector.apply(initial, roi, replace(samples[48], hand_number=2))
    assert all(p.riichi is False for p in empty.players)


def test_riichi_never_learns_mid_hand_or_latches_animation_or_duplicate_frames(samples):
    roi = ROIConfig.load(ROOT / 'configs/roi.steam.yaml')
    detector = RiichiDetector(RiichiConfig(enabled=True))
    initial = cv2.imread(str(DIRECTORY / 'diagnostics/000048.jpg'))
    current = cv2.imread(str(DIRECTORY / 'diagnostics/000897.jpg'))
    detector.apply(current, roi, samples[897])
    assert not detector.baselines
    detector.apply(initial, roi, samples[48])
    for n in range(3):
        detector.apply(current, roi, replace(samples[897], timestamp=100+n, frame_index=100+n))
        detector.apply(initial, roi, replace(samples[48], timestamp=100+n+.4, frame_index=200+n))
    assert not detector.latched
    for _ in range(5):
        detector.apply(current, roi, replace(samples[897], timestamp=200, frame_index=300))
    assert not detector.latched
    assert all(count == 1 for _, _, count in detector.samples.values())


def local(samples, *, timestamp=0, hand=None, drawn_index=7, buttons=(), total=24, issues=()):
    obs = samples[897]
    return replace(obs, timestamp=timestamp, frame_index=round(timestamp*10)+1,
                   hand=obs.hand if hand is None else hand, drawn_index=drawn_index,
                   action_buttons=buttons, issues=issues,
                   countdown={'status': 'complete', 'total_seconds': total} if total is not None else {'status': 'unknown'})


def response(samples, **values):
    """A call window: 13 tiles, no drawn tile, response buttons."""
    values.setdefault('buttons', ('pon', 'pass'))
    return local(samples, hand=samples[897].hand[:-1], drawn_index=None, **values)


def test_counter_ocr_failure_still_allows_exact_draw_at_the_turn_end(samples):
    """7+20 hand timer: 24 s on screen; an ordinary turn may spend 2 s of the
    shared reserve, so the rung fires at 7 - 1 + 2 - 1.5 = 6.5 s."""
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    start = local(samples, issues=('riichi_sticks: numeric_ocr_rejected', 'kita_right: unknown'))
    assert not watchdog.observe(start)
    assert not watchdog.observe(replace(start, timestamp=6.4))
    current = replace(start, timestamp=6.6, frame_index=102)
    assert watchdog.observe(current)
    action, reason = watchdog.select(current, {})
    assert reason is None and action.kind == 'discard' and action.tile == '2z' and action.hand_index == 7
    after = replace(current, timestamp=7.0, frame_index=103, hand=current.hand[:-1], drawn_index=None)
    result = watchdog.verification(action, current, after)
    assert result.status == 'verified' and result.evidence['river_evidence_required'] is False


def test_short_clock_fires_lead_seconds_before_the_client_deadline(samples):
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    assert not watchdog.observe(local(samples, total=5))
    # 5 s - 1 s rounding - 1.5 s lead
    assert not watchdog.observe(local(samples, timestamp=2.4, total=None))
    assert watchdog.observe(local(samples, timestamp=2.5, total=None))
    assert watchdog.select(local(samples, timestamp=2.5, total=None), {})[0].kind == 'discard'
    assert watchdog.select(local(samples, timestamp=3.6, total=None), {})[1] == 'operation_window_expired'


@pytest.mark.parametrize('change,reason', [
    ({'drawn_index': None}, 'fresh_draw_not_proven_including_post_pon'),
    ({'issues': ('hand: unknown',)}, 'own_hand_unrecognized_or_illegal'),
    ({'action_buttons': None}, 'action_buttons_unrecognized'),
])
def test_local_fallback_rejects_unsafe_windows(samples, change, reason):
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    watchdog.observe(local(samples))
    action, rejected = watchdog.select(replace(local(samples, timestamp=6.6), **change), {})
    assert action is None and rejected == reason


def test_own_turn_discards_the_drawn_tile_despite_riichi_or_pass_buttons(samples):
    """live-20261018-164552 f275/f320: PASS on our own turn skipped the discard."""
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    current = local(samples, buttons=('pass', 'riichi'))
    watchdog.observe(current)
    action, _ = watchdog.select(replace(current, timestamp=6.6), {'pass': Box(1, 1, 2, 2), 'riichi': Box(3, 1, 4, 2)})
    assert action.kind == 'discard' and action.hand_index == 7


def test_unreadable_clock_frame_does_not_block_the_counted_down_window(samples):
    """The clock is read once; the reserve-only display misread later is fine."""
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    watchdog.observe(local(samples))
    current = local(samples, timestamp=6.6, total=None)
    assert watchdog.observe(current)
    assert watchdog.select(current, {})[0].kind == 'discard'


def test_window_not_open_or_expired_is_never_clicked(samples):
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    assert watchdog.select(local(samples), {})[1] == 'operation_window_not_open'
    watchdog.observe(local(samples, total=5))
    assert watchdog.select(local(samples, timestamp=3.6, total=None), {})[1] == 'operation_window_expired'


def test_recognized_win_is_taken_and_a_possible_win_is_never_passed(samples):
    """live-20261018-164552 f258/f312: PASS declined a ron twice."""
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    watchdog.observe(response(samples, buttons=('pon', 'ron', 'pass')))
    current = response(samples, timestamp=6.6, buttons=('pon', 'ron', 'pass'))
    boxes = {'pon': Box(1, 1, 2, 2), 'ron': Box(3, 1, 4, 2), 'pass': Box(5, 1, 6, 2)}
    action, _ = watchdog.select(current, boxes)
    assert action.kind == 'ron' and action.button == 'ron'
    hidden = response(samples, timestamp=6.6, buttons=('pon', 'pass'),
                      issues=('action_buttons: unknown candidates',))
    action, reason = watchdog.select(hidden, {'pon': Box(1, 1, 2, 2), 'pass': Box(5, 1, 6, 2)}, win_hint=True)
    assert action is None and reason == 'possible_win_button_unrecognized'


def test_call_window_and_following_draw_are_separate_turns(samples):
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    watchdog.observe(response(samples, total=5))
    current = response(samples, timestamp=2.5, total=None)
    assert watchdog.observe(current)
    action, _ = watchdog.select(current, {'pass': Box(5, 1, 6, 2)})
    assert action.kind == 'pass'
    draw = local(samples, timestamp=3.0, total=None)
    # The draw is a new turn with its own budget, not the expired call window.
    assert not watchdog.observe(draw) and watchdog.since == 3.0 and watchdog.kind == 'own'


def test_expired_clock_and_new_turn_dont_reuse_old_timeout(samples):
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    watchdog.observe(local(samples, total=5))
    assert watchdog.select(local(samples, timestamp=11, total=1), {})[0] is None
    waiting = local(samples, timestamp=12, hand=samples[897].hand[:-1], drawn_index=None, total=None)
    assert not watchdog.observe(waiting) and watchdog.since is None
    assert not watchdog.observe(local(samples, timestamp=13, total=5))
    assert watchdog.since == 13


def test_riichi_selection_discards_a_selectable_tile(samples):
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    watchdog.observe(local(samples))
    obs = replace(local(samples, timestamp=6.6), riichi_discard_indices=(0, 1))
    action, _ = watchdog.select(obs, {}, pending_riichi=True)
    assert action.kind == 'discard' and action.hand_index == 0
    unseen = replace(local(samples, timestamp=6.6), riichi_discard_indices=None)
    assert watchdog.select(unseen, {}, pending_riichi=True)[1] == 'riichi_selection_not_observed'


def test_unreadable_timer_uses_the_hand_timer_from_the_last_closed_window(samples):
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    waiting = local(samples, timestamp=0, hand=samples[897].hand[:-1], drawn_index=None, total=None)
    assert not watchdog.observe(waiting)
    # Opened unseen after 0: 7 s base, 2 s of reserve, 1 s rounding, 1.5 s lead.
    assert not watchdog.observe(local(samples, timestamp=1, total=None))
    assert not watchdog.observe(local(samples, timestamp=6.4, total=None))
    current = local(samples, timestamp=6.6, total=None)
    assert watchdog.observe(current)
    assert watchdog.select(current, {})[0].kind == 'discard'


def test_spent_reserve_leaves_only_timeout_turns(samples):
    from majsoul.live.turn_timer import MatchTimer
    timer = MatchTimer()
    timer.new_hand()
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True), timer=timer)
    timer.spend(20)
    assert timer.timeout and timer.turn_base == 5 and timer.turn_reserve == 0
    watchdog.observe(local(samples, total=None))
    # 5 s time-out turn: 5 - 1 - 1.5
    assert watchdog.commit_at == pytest.approx(2.5)


def test_recognized_pass_is_usable_despite_other_unknown_buttons(samples):
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    watchdog.observe(response(samples, buttons=('pass',)))
    current = response(samples, timestamp=6.6, buttons=('pass',), issues=('action_buttons: unknown candidates',))
    assert watchdog.select(current, {'pass': Box(1, 1, 2, 2)})[0].kind == 'pass'


def test_wrong_tile_removal_never_counts_as_emergency_success(samples):
    watchdog = TurnWatchdog(EmergencyConfig(enabled=True))
    before = local(samples)
    watchdog.observe(before)
    action, _ = watchdog.select(local(samples, timestamp=6.6), {})
    wrong = replace(before, timestamp=7.0, frame_index=121, hand=before.hand[1:], drawn_index=None)
    assert watchdog.verification(action, before, wrong).status == 'failed'


@pytest.mark.parametrize('values', [{'lead_seconds': 0.4, 'click_seconds': 0.5},
                                    {'base_seconds': 2}, {'maximum_turn_seconds': 0}])
def test_invalid_tsumogiri_timing_is_rejected(values):
    with pytest.raises(ValueError):
        EmergencyConfig.from_dict({'enabled': True, **values})


def test_rebuild_waits_for_opponent_turn_and_runs_once(tmp_path, samples):
    loop, window, clock = make_loop(tmp_path, SlowPolicy(), execute=True)
    loop.session_ui = None
    loop.reader.supports_local_input = True
    loop.turn_watchdog.rebuild_pending = True
    waiting = [False]
    def read_local(image, roi, slots, *, timestamp, frame_index):
        obs = local(samples, timestamp=timestamp)
        if waiting[0]:
            obs = replace(obs, hand=obs.hand[:-1], drawn_index=None, countdown={'status': 'unknown'})
        obs = replace(obs, frame_index=frame_index, active_slots=tuple(slots))
        return obs, visible_reading(tuple(t.tile for t in obs.hand)), {}
    loop.reader.read_local = read_local
    try:
        initial_epoch = loop._decision_epoch
        frame, index = loop._capture()
        loop._emergency_tick(frame, index)
        assert loop._decision_epoch == initial_epoch
        waiting[0] = True
        frame, index = loop._capture()
        loop._emergency_tick(frame, index)
        assert loop._decision_epoch == initial_epoch + 1
        assert loop.turn_watchdog.rebuild_pending is False
        frame, index = loop._capture()
        loop._emergency_tick(frame, index)
        assert loop._decision_epoch == initial_epoch + 1
    finally:
        close(loop)


def test_local_watchdog_preempts_global_reader_and_cancels_model(tmp_path, samples):
    policy = SlowPolicy()
    loop, window, clock = make_loop(tmp_path, policy, execute=True)
    loop.session_ui = None
    process(loop)
    assert policy.entered.wait(1)
    old_future = loop._pending_decision['future']
    loop.roi = ROIConfig.load(ROOT / 'configs/roi.steam.yaml')
    loop.reader.supports_local_input = True
    clicked = []
    def read_local(image, roi, slots, *, timestamp, frame_index):
        obs = local(samples, timestamp=timestamp, total=24)
        if clicked:
            obs = replace(obs, hand=obs.hand[:-1], drawn_index=None, countdown={'status': 'unknown'})
        obs = replace(obs, active_slots=tuple(slots), frame_index=frame_index)
        return obs, visible_reading(tuple(t.tile for t in obs.hand)), {}
    loop.reader.read_local = read_local
    loop.reader.read_with_visible = lambda *a, **k: (_ for _ in ()).throw(AssertionError('global reader must not run'))
    window.click = lambda point, **kwargs: clicked.append((point, kwargs))
    try:
        loop.turn_watchdog.observe(local(samples, timestamp=clock()))
        clock.sleep(8.2)
        window.size = (2582, 1550)
        frame, index = loop._capture()
        assert loop._process_frame(frame, index) is False
        assert len(clicked) == 1 and clicked[0][1]['click_count'] == 2
        assert old_future.cancelled() and loop._pending_decision is None
        assert loop.turn_watchdog.rebuild_pending
        events = [json.loads(x)['event'] for x in (loop.artifacts / 'events.jsonl').read_text().splitlines()]
        assert 'emergency_verified' in events
    finally:
        close(loop)


def test_protected_runner_allows_local_timeout_before_client_takeover():
    guard = AcceptanceGuard(17, local_fallback=True)
    guard.observe('decision_gate', {'countdown': {'status': 'complete'}}, 0)
    guard.observe('decision_failed', {'error': 'transport'}, 3)
    guard.check(10.1)
    guard.observe('emergency_verified', {}, 12)
    guard.check(18)
    with pytest.raises(AcceptanceStopped, match='emergency_failed'):
        guard.observe('emergency_failed', {'reason': 'no local proof'}, 20)
