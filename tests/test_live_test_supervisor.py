from dataclasses import replace
from pathlib import Path

import pytest

from majsoul.live.collector import CollectConfig, TOGGLE_LABELS
from tools.live_test_supervisor import AcceptanceGuard, AcceptanceStopped, enable_client_takeover


@pytest.mark.parametrize('event',['verification_unknown','verification_failed','decision_expired','decision_failed','safe_pause','interrupted'])
def test_acceptance_stops_at_first_failure(event):
    guard=AcceptanceGuard()
    with pytest.raises(AcceptanceStopped):
        guard.observe(event,{'action_id':'discard:0:9m'},10)
    assert guard.failure['event']==event


def test_unknown_turn_stalls_but_opponent_wait_does_not():
    guard=AcceptanceGuard()
    guard.observe('decision_gate',{'countdown':{'status':'complete'}},10)
    guard.check(17.9)
    with pytest.raises(AcceptanceStopped):
        guard.check(18)
    other=AcceptanceGuard()
    other.observe('decision_gate',{'countdown':{'status':'absent'}},10)
    other.check(100)
    assert other.failure is None


def test_takeover_adds_auto_sort_without_modifying_collect_defaults(tmp_path,monkeypatch):
    config=CollectConfig.load(Path(__file__).resolve().parents[1]/'configs/collect.steam.yaml')
    original=config.toggle_targets
    calls=[]
    class Panel:
        def __init__(self,window,actual,**kwargs):
            assert set(actual.toggle_targets)==set(TOGGLE_LABELS)
            assert actual.toggle_targets[0]=='auto_discard'
        def ensure(self,*,enabled):
            calls.append(enabled)
            return []
    monkeypatch.setattr('tools.live_test_supervisor.TogglePanel',Panel)
    report=enable_client_takeover(object(),config,tmp_path)
    assert report['verified'] and calls==[True]
    assert config.toggle_targets==original and 'auto_sort' not in original
    assert (tmp_path/'takeover.json').exists()


def test_takeover_reobserves_after_transient_drawer_failure(tmp_path,monkeypatch):
    config=CollectConfig.load(Path(__file__).resolve().parents[1]/'configs/collect.steam.yaml')
    attempts=[]
    class Panel:
        def __init__(self,*args,**kwargs):
            pass
        def ensure(self,*,enabled):
            attempts.append(enabled)
            if len(attempts)==1:
                raise RuntimeError('toggle drawer is not visibly open')
            return []
    monkeypatch.setattr('tools.live_test_supervisor.TogglePanel',Panel)
    report=enable_client_takeover(object(),config,tmp_path)
    assert report['verified']
    assert attempts==[True,True]
    assert 'error' in report['attempts'][0]
    assert report['attempts'][1]=={'missing':[]}
