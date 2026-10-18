"""The watchdog keeps the Agent alive and never plays; spawned fake children."""
from dataclasses import replace
import json
from pathlib import Path
import threading

import pytest

from majsoul.live.config import LiveConfig
from majsoul.live.watchdog import WatchdogConfig, run_watchdog
from tools.protection_fault_probe import OfflineNotifier
import tools.watchdog_fault_probe as probe

ROOT = Path(__file__).resolve().parents[1]


def config(tmp_path, **watchdog):
    base = LiveConfig.load(ROOT/'configs/live.steam.yaml')
    fast = dict(heartbeat_seconds=.1, stall_seconds=1.0, startup_seconds=30, poll_seconds=.02,
                restart_backoff_seconds=.05, restart_backoff_max_seconds=.1, standby_delay_seconds=.1,
                stop_grace_seconds=.5, max_matches=1)
    fast.update(watchdog)
    return replace(base, artifacts_root=tmp_path, watchdog=replace(base.watchdog, **fast))


def run(tmp_path, child, **watchdog):
    cfg = config(tmp_path, **watchdog)
    code = run_watchdog(cfg, 1, session='run', notifier=OfflineNotifier(), _targets={'agent': child})
    root = tmp_path/'run'
    result = json.loads((root/'watchdog-result.json').read_text(encoding='utf-8'))
    events = [json.loads(line) for line in (root/'watchdog.jsonl').read_text(encoding='utf-8').splitlines()]
    starts = {p.stem: json.loads(p.read_text()) for p in root.glob('agent*.json')}
    return code, result, events, starts


def names(events):
    return [e['event'] for e in events]


def test_agent_completing_the_match_is_accepted_without_restarts(tmp_path):
    code, result, events, starts = run(tmp_path, probe.completing_agent, warm_standby=False)
    assert code == 0 and result['agent_acceptance'] and result['returned_lobby']
    assert result['agent_restarts'] == [] and result['mixed_control'] is False
    assert starts == {'agent': {'join': False, 'standby': False}}


def test_crash_promotes_the_loaded_standby_at_once(tmp_path):
    code, result, events, starts = run(tmp_path, probe.crash_then_complete)
    assert 'standby_promoted' in names(events) and 'restart_scheduled' not in names(events)
    assert result['returned_lobby'] and not result['agent_acceptance']
    assert len(result['agent_restarts']) == 1
    assert starts['agent-002'] == {'join': True, 'standby': True}
    assert code == 1


def test_silent_agent_is_fenced_and_a_cold_agent_joins_the_match(tmp_path):
    code, result, events, starts = run(tmp_path, probe.hang_then_complete, warm_standby=False)
    reason = result['agent_restarts'][0]['reason']
    assert reason['kind'] == 'agent_stalled'
    assert starts['agent-002'] == {'join': True, 'standby': False}
    assert result['returned_lobby']


def test_closed_window_stops_without_restart_storm(tmp_path):
    code, result, events, starts = run(tmp_path, probe.window_closed)
    assert result['reason']['kind'] == 'window_gone'
    assert list(starts) == ['agent'] and result['agent_restarts'] == []
    assert code == 1


def test_restart_budget_is_bounded(tmp_path):
    code, result, events, starts = run(tmp_path, probe.always_crash, restart_limit=2, warm_standby=False)
    assert result['reason']['kind'] == 'restart_budget_exhausted'
    assert len(starts) == 3 and len(result['agent_restarts']) == 3


def test_failure_after_matchmaking_began_restarts_by_joining_not_by_menu(tmp_path):
    code, result, events, starts = run(tmp_path, probe.crash_while_matching, warm_standby=False)
    assert starts['agent'] == {'join': False, 'standby': False}
    assert starts['agent-002']['join'] is True
    assert result['returned_lobby']


def test_operator_stop_ends_the_run_and_the_agent(tmp_path):
    cfg = config(tmp_path, warm_standby=False)
    timer = threading.Timer(2.5, lambda: (tmp_path/'run'/'STOP').write_text('stop'))
    timer.start()
    try:
        code = run_watchdog(cfg, 1, session='run', notifier=OfflineNotifier(),
                            _targets={'agent': probe.endless_agent})
    finally:
        timer.cancel()
    result = json.loads((tmp_path/'run'/'watchdog-result.json').read_text(encoding='utf-8'))
    assert code == 1 and result['reason']['kind'] == 'operator_stop'


def test_watchdog_module_never_imports_the_takeover_supervisor():
    source = (ROOT/'majsoul/live/watchdog.py').read_text(encoding='utf-8')
    assert 'protection' not in source.replace('modals', '')
    assert '.click(' not in source


@pytest.mark.parametrize('values', [{'stall_seconds': 1, 'heartbeat_seconds': 2}, {'restart_limit': -1},
                                    {'warm_standby': 1}, {'surprise': 1}, {'poll_seconds': 0}])
def test_invalid_watchdog_configuration_is_rejected(values):
    with pytest.raises(ValueError):
        WatchdogConfig.from_dict(values)


def test_matches_are_played_back_to_back_until_the_limit(tmp_path):
    """User request 2026-10-18: keep playing after returning to the lobby."""
    code, result, events, starts = run(tmp_path, probe.completing_agent, warm_standby=False, max_matches=2)
    assert code == 0 and result['matches_completed'] == 2
    assert [m['match'] for m in result['matches']] == [1, 2]
    # Each match gets a fresh Agent that enters it through the menus.
    assert starts == {'agent': {'join': False, 'standby': False},
                      'agent-002': {'join': False, 'standby': False}}
    assert names(events).count('match_complete') == 2
