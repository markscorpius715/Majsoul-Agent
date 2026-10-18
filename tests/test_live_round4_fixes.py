"""Hidden bugs found in the logs of live-20261018-164552/204720/212426."""
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from majsoul.calibration.frames import read_image
from majsoul.calibration.roi_config import ROIConfig
from majsoul.domain import Action
from majsoul.live.capture import LiveFrame, roi_for_frame
from majsoul.live.config import LiveConfig
from majsoul.live.decisions import ChoiceResult

from test_live_play_fixes import make_loop  # noqa: F401  (fixture)
import test_live_watchdog as watchdog_tests
import tools.watchdog_fault_probe as probe

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'artifacts/live'


def finished(loop, action, outcome, monkeypatch):
    monkeypatch.setattr(loop, '_execute_action', lambda *a, **k: outcome)
    loop.execute = True
    monkeypatch.setattr(loop, '_input_allowed', lambda: True)
    future = Future()
    future.set_result(ChoiceResult(action, {'rung': 'model'}, None))
    observation = SimpleNamespace(timestamp=loop.now(), frame_index=1, fingerprint='x', countdown=None)
    pending = {'request_id': 'r', 'key': ('k',), 'future': future,
               'frame': SimpleNamespace(window=(0, 0, 10, 10)), 'observation': observation,
               'domain': SimpleNamespace(for_model=lambda: {}), 'actions': [action], 'phase': 'discard',
               'deadline': loop.now()+5, 'snapshot': None, 'budget': {}, 'source_image': None}
    loop._finish_decision(pending)


def test_verified_kita_restarts_the_fallback_turn_clock(make_loop, monkeypatch):
    """7 of 8 'model click then fallback discard' cases: after every kita the
    fallback kept the pre-kita clock and tsumogiri'd the replacement tile."""
    loop = make_loop()
    watchdog = loop.turn_watchdog
    watchdog.since, watchdog.kind = loop.now()-4, 'own'
    loop._last_click_at = loop.now()-0.5
    finished(loop, Action('button:nuki', 'nuki', button='nuki'), 'verified', monkeypatch)
    assert watchdog.since is None and watchdog.idle_at == loop._last_click_at


def test_ready_answer_may_be_clicked_until_the_client_deadline(make_loop):
    """Five model answers were dropped 0.3 s after the decision deadline with
    the whole 20 s reserve left; the fallback then discarded instead."""
    loop = make_loop()
    loop.match_timer.new_hand('1z:1:0')
    clock = loop.operation_clock
    clock.start(100.)
    clock.observe(100., {'status': 'complete', 'mode': 'base_plus_reserve',
                         'base_seconds': 4, 'reserve_seconds': 20, 'total_seconds': 24})
    assert clock.decision_due < loop._click_deadline(clock.decision_due) == pytest.approx(clock.last_safe_input)
    assert clock.last_safe_input == pytest.approx(100.+24-1-0.5)
    clock.reset()
    assert loop._click_deadline(50.) == 50.


@pytest.mark.parametrize('frame,expected', [
    ('live-20261018-212426-175400/agent-004/frames/emergency-000290-before.png', True),
    ('live-20261018-164552-978100/agent-002/frames/00008-before.png', False),
    ('live-20261018-164552-978100/agent/frames/emergency-000275-before.png', False)])
def test_client_riichi_selection_mode_is_detected_before_a_plain_discard(make_loop, frame, expected):
    """212426 agent-004: the client was already selecting a riichi tile (1z
    raised, cancel marker shown); blind double-clicks removed the wrong tile."""
    loop = make_loop()
    config = LiveConfig.load(ROOT/'configs/live.steam.yaml')
    image = read_image(RUNS/frame)
    roi = roi_for_frame(ROIConfig.load(config.roi), image.shape[1::-1], 'auto')
    assert loop._riichi_mode_visible(LiveFrame(image, 0., (0, 0, *image.shape[1::-1]), (0, 0)), roi) is expected


def test_lobby_found_after_an_outage_ends_the_match_instead_of_restarting(tmp_path):
    """204720 match 2: the match ended during a lock screen; the replacement
    found the lobby and was restarted as if it had crashed."""
    code, result, events, starts = watchdog_tests.run(tmp_path, probe.lobby_after_outage, warm_standby=False)
    assert result['matches_completed'] == 1
    assert len(result['agent_restarts']) == 1  # only the real lock-screen failure
    assert result['reason'] is None and 'restart_budget_exhausted' not in watchdog_tests.names(events)
