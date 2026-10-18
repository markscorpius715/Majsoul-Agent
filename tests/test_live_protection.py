from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from majsoul.live.config import LiveConfig
from majsoul.live.protection import (ProtectionConfig,ProtectionMonitor,WindowLease,
                                    run_protected,stop_input_owner,_drain,_input_window)
from tools.protection_fault_probe import (OfflineNotifier,observer,hung_agent,
    responsive_but_blocked_agent,verification_failure_agent,protector,first_protector_hangs,
    waiting_observer,input_hung_agent,delayed_standby,no_agent_expected,prestart_protector,
    lobby_observer,lobby_failure_agent,stopping_agent,generous_observer,successful_agent,
    prestart_observer,unrecognized_agent,never_ready_rejoin,table_observer,own_turn_observer,
    failing_once_agent,handback_protector,ready_rejoin)

ROOT=Path(__file__).resolve().parents[1]


def test_stall_diagnostics_follow_actual_capture_heartbeats(monkeypatch):
    from queue import Queue
    import majsoul.live.capture as capture
    class Window:
        def __init__(self, hwnd):self.blocked=False
        def capture(self):
            if self.blocked:raise RuntimeError('capture failed')
            return 'frame'
    monkeypatch.setattr(capture,'LiveWindow',Window)
    queue=Queue()
    stages=[]
    window=_input_window(1,queue,None,on_heartbeat=lambda fields:stages.append(fields['stage']))
    assert window.capture()=='frame'
    assert window.capture()=='frame'
    window.blocked=True
    with pytest.raises(RuntimeError,match='capture failed'):window.capture()
    assert stages==['capture_begin','capture_done','capture_begin','capture_done','capture_begin']
    assert [queue.get_nowait()['fields']['stage'] for _ in stages]==stages


def sample(at,**values):
    return {**dict(captured_at=at,table=True,lobby=False,actionable=True,waiting=False,
                   total_seconds=5),**values}


def test_unavailable_desktop_observation_does_not_latch_or_renew_safety_budgets():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.observe(sample(0),0)
    m.observe({'captured_at':.2,'desktop_status':{'status':'unknown'}},.2)
    assert m.reason is None and m.last_observation==0
    assert m.operation_due==1.5 and not m.lobby_verified
    m.observe(sample(.4),.4)
    m.event('heartbeat',{},1.5)
    assert m.check(1.5)['kind']=='operation_deadline'


def test_persistent_desktop_unknown_expires_existing_observer_timeout():
    m=ProtectionMonitor(ProtectionConfig(),0,require_agent_recognition=True)
    for at in (.2,1,2.1):
        m.observe({'captured_at':at,'desktop_status':{'status':'unknown'}},at)
    assert m.last_observation==0
    assert m.check(2.1)['kind']=='observer_stalled'


def test_confirmed_desktop_lock_still_trips_immediately():
    m=ProtectionMonitor(ProtectionConfig(),0,require_agent_recognition=True)
    m.observe({'captured_at':.2,'desktop_status':{'status':'blocked'},
               'error':'input_desktop_blocked'},.2)
    assert m.check(.2)['kind']=='observation_error'


def test_input_owner_never_sends_input_with_unknown_or_locked_desktop(monkeypatch):
    from queue import Queue
    from types import SimpleNamespace
    import majsoul.live.capture as capture
    import majsoul.live.modals as modals
    sent=[]
    class Window:
        def __init__(self,hwnd):pass
        def click(self,*args,**kwargs):sent.append(args)
    monkeypatch.setattr(capture,'LiveWindow',Window)
    window=_input_window(1,Queue(),SimpleNamespace(value=False))
    monkeypatch.setattr(modals,'input_desktop_status',lambda:{'status':'unknown'})
    with pytest.raises(capture.FocusUnavailable):window.click((1,1))
    monkeypatch.setattr(modals,'input_desktop_status',lambda:{'status':'blocked'})
    with pytest.raises(RuntimeError,match='input_desktop_blocked'):window.click((1,1))
    assert not sent


def test_five_second_turn_is_protected_before_timeout_not_after_ten_seconds():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.observe(sample(0),0)
    assert m.operation_due==1.5
    m.event('heartbeat',{},1.4);m.observe(sample(1.4),1.4)
    assert m.check(1.49) is None
    assert m.check(1.5)['kind']=='operation_deadline'


def test_agent_recognition_retries_for_configured_thirty_seconds_despite_ui_unknowns():
    m=ProtectionMonitor(ProtectionConfig(),0,require_agent_recognition=True)
    m.event('worker_ready',{},0)
    m.event('table_wait_started',{},1)
    for at in (1,2,10,20,30.9):
        x=sample(at)
        x.update(total_seconds=None,uncertain=True)
        m.observe(x,at)
        assert m.check(at) is None
    m.event('table_wait_started',{},30.9)
    assert m.check(31)['kind']=='agent_recognition_timeout'
    assert m.operation_due is None and not m.clock.open


def test_agent_recognition_enables_fresh_operation_protection_without_old_animation_budget():
    m=ProtectionMonitor(ProtectionConfig(),0,require_agent_recognition=True)
    m.event('table_wait_started',{},0)
    x=sample(1);x['total_seconds']=None
    m.observe(x,1)
    assert m.check(1.5) is None
    m.event('agent_game_started',{},2)
    assert m.seen_match
    m.observe(sample(2.1),2.1)
    assert m.operation_since==2.1 and m.operation_due==pytest.approx(3.6)
    m.event('heartbeat',{},3.6)
    assert m.check(3.6)['kind']=='operation_deadline'


def test_window_without_any_clock_reading_assumes_base_time_only():
    m=ProtectionMonitor(ProtectionConfig(),0)
    x=sample(0);x['total_seconds']=None
    m.observe(x,0)
    # 5 s base - 1 s quantization - 2.5 s takeover margin; no 0.4 s clock trip.
    assert m.check(1.49) is None
    reason=m.check(1.5)
    assert reason['kind']=='operation_deadline' and reason['clock']['source']=='base_time_fallback'


def test_one_confirmed_reading_is_counted_down_locally_and_unknown_frames_never_trip():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.observe(sample(0,total_seconds=25),0)
    assert m.operation_due==8  # maximum_operation_seconds still caps the window
    for at in (.5,1,2,4,6,7.9):
        x=sample(at);x['total_seconds']=None
        m.event('heartbeat',{},at);m.observe(x,at)
        assert m.check(at) is None
    m.event('heartbeat',{},8)
    assert m.check(8)['kind']=='operation_deadline'


def test_later_readings_only_shorten_the_locally_counted_deadline():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.observe(sample(0,total_seconds=25),0)
    m.event('heartbeat',{},1);m.observe(sample(1,total_seconds=30),1)
    assert m.operation_due==8
    m.event('heartbeat',{},2);m.observe(sample(2,total_seconds=6),2)
    assert m.operation_due==pytest.approx(4.5)


@pytest.mark.parametrize('total,expected',[(None,True),(25,False)])
def test_observer_gap_counts_base_fallback_from_last_closed_window(total,expected):
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.observe(sample(0,actionable=False,waiting=True),0)
    x=sample(1.9,total_seconds=total)
    m.event('heartbeat',{},1.9)
    m.observe(x,1.9)  # The window may have opened anywhere after 0.
    assert (m.check(1.9) is not None)==expected


def test_brief_clock_animation_can_recover_before_protection_debounce():
    m=ProtectionMonitor(ProtectionConfig(),0)
    x=sample(0);x['total_seconds']=None
    m.observe(x,0)
    x=sample(.2);x['total_seconds']=25
    m.observe(x,.2)
    assert m.operation_due==8 and m.check(.4) is None


def test_unknown_clock_failure_preserves_the_initial_actionable_evidence():
    m=ProtectionMonitor(ProtectionConfig(),0)
    initial=sample(0)
    initial.update(total_seconds=None,evidence='observer/first.jpg',hand_count=14)
    m.observe(initial,0)
    waiting=sample(.2)
    waiting.update(total_seconds=None,actionable=False,waiting=True,hand_count=13)
    m.observe(waiting,.2)
    reason=m.check(.5)
    assert reason['kind']=='unverified_operation_ended'
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.observe(initial,0)
    later=sample(.2);later.update(total_seconds=None)
    m.observe(later,.2)
    reason=m.check(1.5)
    assert reason['kind']=='operation_deadline'
    assert reason['operation_origin']==initial


def test_later_animation_or_longer_clock_never_extends_budget():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.observe(sample(0),0)
    x=sample(.5);x.update(actionable=False,waiting=False,total_seconds=None)
    m.observe(x,.5)
    x=sample(1);x['total_seconds']=27
    m.observe(x,1)
    assert m.operation_due==1.5


def test_unverified_waiting_does_not_erase_a_missed_operation():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.observe(sample(0),0)
    x=sample(.5);x.update(actionable=False,waiting=True,total_seconds=None)
    m.observe(x,.5)
    assert m.operation_since==0
    assert m.check(.81)['kind']=='unverified_operation_ended'


def test_input_then_early_waiting_resolves_window_but_is_not_tile_verification():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.observe(sample(0),0)
    m.event('heartbeat',{'stage':'input_end','input_completed':True},.4)
    x=sample(.5);x.update(actionable=False,waiting=True,total_seconds=None)
    m.observe(x,.5)
    assert m.operation_due is None and m.reason is None
    m.event('verification_failed',{'reason':'wrong tile'},.7)
    assert m.reason['event']=='verification_failed'


@pytest.mark.parametrize('event',['verification_failed','agent_stopped','worker_exception','fatal_error','emergency_failed'])
def test_any_failed_stage_trips_immediately(event):
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.event(event,{'reason':'test'},.2)
    assert m.check(.2)['event']==event


@pytest.mark.parametrize('event',['decision_failed','decision_expired','verification_unknown','preflight_abort','safe_pause'])
def test_single_no_input_outcome_is_left_to_the_agent_and_the_operation_deadline(event):
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.event(event,{'reason':'test'},.2)
    m.event('heartbeat',{},.3)
    assert m.check(.3) is None


def test_repeated_soft_failures_escalate_once_per_dedupe_interval():
    m=ProtectionMonitor(ProtectionConfig(soft_failure_limit=3),0)
    for at in (.1,.2,.3,.4):  # One missed window logs several lines.
        m.event('decision_expired',{},at)
    assert m.reason is None and m.soft_failures==1
    m.event('preflight_abort',{},1.2)
    m.event('verified',{},1.5)  # A completed operation clears the run.
    m.event('preflight_abort',{},2.6)
    m.event('preflight_abort',{},3.7)
    assert m.reason is None
    m.event('safe_pause',{},4.8)
    assert m.reason['kind']=='repeated_soft_failures' and m.reason['count']==3


def test_stalled_global_reader_is_detected_without_its_callbacks():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.event('worker_ready',{},0)
    x=sample(1.9);x.update(actionable=False,waiting=True,total_seconds=None)
    m.observe(x,1.9)
    assert m.check(2.01)['kind']=='agent_stalled'


def test_child_heartbeat_age_starts_at_launch_without_extending_operation_budget():
    m=ProtectionMonitor(ProtectionConfig(),0)
    x=sample(1.7);x['total_seconds']=27
    m.observe(x,1.7)
    due=m.operation_due
    m.event('input_owner_started',{},1.8)
    x=sample(2.1);x['total_seconds']=27
    m.observe(x,2.1)
    assert m.check(2.1) is None and m.operation_due==due
    x=sample(3.81);x['total_seconds']=27
    m.observe(x,3.81)
    assert m.check(3.81)['kind']=='agent_stalled'


def test_critical_unknown_does_not_wait_for_global_safe_pause():
    m=ProtectionMonitor(ProtectionConfig(),0)
    x=sample(0);x.update(actionable=False,waiting=False,total_seconds=None,uncertain=True)
    m.observe(x,0)
    m.event('heartbeat',{},.5)
    assert m.check(1)['kind']=='critical_local_ui_uncertain'


def test_dealing_count_animation_can_finish_without_false_uncertainty_takeover():
    m=ProtectionMonitor(ProtectionConfig(),0)
    for at,count in ((0,8),(.4,12),(.8,8)):
        x=sample(at)
        x.update(actionable=False,waiting=False,uncertain=True,total_seconds=None,hand_count=count)
        m.observe(x,at)
        m.event('heartbeat',{},at)
    assert m.check(1.05) is None
    x=sample(1.2)
    x.update(actionable=False,waiting=True,total_seconds=None,hand_count=13)
    m.observe(x,1.2)
    assert m.check(1.3) is None


def test_changing_bad_hand_counts_cannot_hide_uncertainty_until_a_turn_expires():
    config=ProtectionConfig()
    m=ProtectionMonitor(config,0)
    for at,count in ((0,8),(.4,12),(.8,8),(1.2,12)):
        x=sample(at)
        x.update(actionable=False,waiting=False,uncertain=True,total_seconds=None,hand_count=count)
        m.observe(x,at)
        m.event('heartbeat',{},at)
    cap=config.shortest_turn_seconds-config.quantization_seconds-config.takeover_margin_seconds
    assert m.check(cap)['kind']=='critical_local_ui_uncertain'


def test_saved_live_dealing_trace_recovers_before_the_takeover_safety_cap():
    directory=ROOT/'artifacts/live/protected-20261013-round04'
    rows=[json.loads(line) for line in (directory/'observer/samples.jsonl').read_text(encoding='utf-8').splitlines()]
    events=[json.loads(line) for line in (directory/'protection.jsonl').read_text(encoding='utf-8').splitlines()]
    fence=next(row['monotonic'] for row in events if row['event']=='protection_requested')
    first=next(i for i,row in enumerate(rows) if row['table'])
    m=ProtectionMonitor(ProtectionConfig(),rows[first]['captured_at'])
    counts=[]
    for row in rows[first:]:
        at=row['captured_at']
        if at>fence:
            assert m.check(fence) is None
        m.event('heartbeat',{},at)  # Replay UI timing independently of child work.
        m.observe(row,at)
        assert m.check(at) is None
        counts.append(row['hand_count'])
        if row['waiting']:break
    assert counts==[8,12,8,13]
    assert m.uncertain_since is None


def test_model_initialization_and_matching_in_lobby_do_not_count_as_a_turn():
    m=ProtectionMonitor(ProtectionConfig(),0)
    x=sample(.5);x.update(table=False,actionable=False,uncertain=True,total_seconds=None)
    m.observe(x,.5)
    assert m.check(1.5) is None


def test_pre_match_heartbeat_cannot_hide_a_stuck_menu_or_loading_stage():
    m=ProtectionMonitor(ProtectionConfig(),0)
    x=sample(29);x.update(table=False,actionable=False,lobby=True,total_seconds=None)
    m.observe(x,29)
    m.event('heartbeat',{},30)
    assert m.check(30)['kind']=='pre_match_timeout'


def test_initialization_does_not_spend_the_bounded_menu_and_matching_wait():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.event('worker_ready',{},20)
    for at in (21, 29, 40, 49):
        x=sample(at)
        x.update(table=False,actionable=False,uncertain=True,total_seconds=None)
        m.observe(x,at)
        m.event('heartbeat',{},at)
        assert m.check(at) is None
    # Repeated readiness cannot extend the matching deadline.
    m.event('worker_ready',{},49)
    m.event('heartbeat',{},50)
    assert m.check(50)=={'kind':'pre_match_timeout','since':20}


def test_matching_uses_menu_timeout_without_extending_initialization_or_turns():
    m=ProtectionMonitor(ProtectionConfig(),0,matching_seconds=120)
    m.event('worker_ready',{},10)
    m.event('table_wait_started',{},17)
    for at in (29,40,70,136):
        x=sample(at)
        x.update(table=False,actionable=False,uncertain=True,total_seconds=None)
        m.observe(x,at)
        m.event('heartbeat',{},at)
        assert m.check(at) is None
    m.event('table_wait_started',{},136)  # A repeated callback cannot renew it.
    m.event('heartbeat',{},137)
    assert m.check(137)=={'kind':'pre_match_timeout','since':17}
    init=ProtectionMonitor(ProtectionConfig(),0,matching_seconds=120)
    init.event('heartbeat',{},30)
    init.last_observation=30
    assert init.check(30)['kind']=='pre_match_timeout'
    turn=ProtectionMonitor(ProtectionConfig(),0,matching_seconds=120)
    turn.event('table_wait_started',{},0)
    turn.observe(sample(0),0)
    turn.event('heartbeat',{},1.5)
    assert turn.check(1.5)['kind']=='operation_deadline'


def test_a_stalled_observer_also_trips_without_waiting_for_agent_errors():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.event('heartbeat',{},2)
    assert m.check(2.01)['kind']=='observer_stalled'


def test_late_ipc_frame_cannot_reopen_a_completed_window():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.observe(sample(0),0)
    m.event('verified',{},.5)
    m.observe(sample(.4),.6)
    assert m.operation_due is None


def test_stale_lobby_frames_cannot_certify_match_completion():
    m=ProtectionMonitor(ProtectionConfig(),0)
    x=sample(0);x.update(table=False,actionable=False,lobby=True)
    m.observe(x,0)
    x['captured_at']=1.1
    m.observe(x,5)
    assert not m.lobby_verified
    assert m.reason['kind']=='stale_or_invalid_observer_frame'


def test_lobby_observation_gap_requires_a_new_continuous_proof():
    m=ProtectionMonitor(ProtectionConfig(),0)
    x=sample(0);x.update(table=False,actionable=False,lobby=True)
    m.observe(x,0)
    x['captured_at']=3
    m.observe(x,3)
    assert not m.lobby_verified
    x['captured_at']=4.1
    m.observe(x,4.1)
    assert m.lobby_verified


@pytest.mark.parametrize('latched',[False,True])
def test_observer_stall_invalidates_lobby_even_after_takeover_is_latched(latched):
    m=ProtectionMonitor(ProtectionConfig(),0)
    x=sample(0);x.update(table=False,actionable=False,lobby=True)
    m.observe(x,0)
    x['captured_at']=1.1
    m.observe(x,1.1)
    assert m.lobby_verified
    if latched:m.trip({'kind':'operator_stop'})
    m.check(3.2)
    assert not m.lobby_verified and m.lobby_since is None


@pytest.mark.parametrize('total',[float('inf'),float('nan'),True,-1])
def test_invalid_clock_never_receives_a_full_operation_budget(total):
    m=ProtectionMonitor(ProtectionConfig(),0)
    x=sample(0);x['total_seconds']=total
    m.observe(x,0)
    assert m.check(1.49) is None
    assert m.check(1.5)['kind']=='operation_deadline'


def test_visible_zero_clock_requests_immediate_takeover():
    m=ProtectionMonitor(ProtectionConfig(),0)
    x=sample(0);x['total_seconds']=0
    m.observe(x,0)
    assert m.check(0)['kind']=='operation_deadline'


def test_busy_ipc_producer_cannot_prevent_monitor_deadline_checks():
    class Flood:
        calls=0
        def get_nowait(self):
            self.calls+=1
            return {'kind':'heartbeat'}
    queue=Flood()
    assert len(list(_drain(queue,128)))==queue.calls==128


def test_late_event_does_not_move_heartbeat_time_backwards():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.event('heartbeat',{},1)
    m.event('heartbeat',{},.5)
    assert m.last_progress==1


def test_window_session_lease_rejects_a_second_controller(tmp_path):
    with WindowLease(tmp_path,123):
        with pytest.raises(RuntimeError,match='Another controller'):
            with WindowLease(tmp_path,123):pass
    with WindowLease(tmp_path,123):pass


def test_replacement_owner_is_forbidden_if_old_process_cannot_be_stopped():
    class Stuck:
        def is_alive(self):return True
        def terminate(self):pass
        def kill(self):pass
        def join(self,timeout):pass
    with pytest.raises(RuntimeError,match='replacement input forbidden'):
        stop_input_owner(Stuck(),.01)


@pytest.mark.parametrize('agent_target,client_target,maximum,observer_target',[
    (hung_agent,protector,8,waiting_observer),
    (responsive_but_blocked_agent,protector,.8,observer),
    (verification_failure_agent,protector,8,observer),
    (responsive_but_blocked_agent,first_protector_hangs,.8,observer),
    (input_hung_agent,protector,8,waiting_observer),
    (stopping_agent,protector,8,waiting_observer),
])
def test_real_spawned_faults_fence_agent_then_protect_until_lobby(tmp_path,agent_target,client_target,maximum,observer_target):
    cfg=LiveConfig.load(ROOT/'configs/live.steam.yaml')
    cfg=replace(cfg,artifacts_root=tmp_path,notify=replace(cfg.notify,enabled=False,shutdown_seconds=.1),
        protection=replace(cfg.protection,heartbeat_seconds=2,maximum_operation_seconds=maximum))
    started=time.monotonic()
    result=run_protected(cfg,998877,session='probe',notifier=OfflineNotifier(),
        _targets={'observer':observer_target,'agent':agent_target,'client':client_target,
                  'rejoin':never_ready_rejoin})
    assert result==1
    report=json.loads((tmp_path/'probe/protection-result.json').read_text())
    assert report['returned_lobby'] and report['mixed_control'] and not report['agent_acceptance']
    assert not report['second_match_started'] and not (tmp_path/'probe/late-input').exists()
    assert not (tmp_path/'probe/probe-timeout').exists()
    events=[json.loads(s) for s in (tmp_path/'probe/protection.jsonl').read_text().splitlines()]
    exit_index=next(i for i,e in enumerate(events) if e['event']=='input_owner_exited')
    starts=[i for i,e in enumerate(events) if e['event']=='input_owner_started' and e['owner']=='client']
    assert starts and min(starts)>exit_index
    assert time.monotonic()-started<20
    if client_target is first_protector_hangs:
        assert len(starts)>=2
    if agent_target is input_hung_agent:
        assert (tmp_path/'probe/client-002/release-mouse').read_text()=='True'


def fault_config(tmp_path,**values):
    cfg=LiveConfig.load(ROOT/'configs/live.steam.yaml')
    return replace(cfg,artifacts_root=tmp_path,
        notify=replace(cfg.notify,enabled=False,shutdown_seconds=.1),
        protection=replace(cfg.protection,**values))


def test_unknown_clock_before_agent_recognition_gets_bounded_recognition_retry(tmp_path):
    cfg=fault_config(tmp_path,recognition_seconds=.8)
    result=run_protected(cfg,998877,session='probe',notifier=OfflineNotifier(),
        _targets={'observer':prestart_observer,'agent':unrecognized_agent,'client':protector,
                  'standby':delayed_standby,'rejoin':never_ready_rejoin})
    report=json.loads((tmp_path/'probe/protection-result.json').read_text())
    assert result==1 and report['returned_lobby'] and not report['agent_acceptance']
    assert report['reason']['kind']=='agent_recognition_timeout'
    assert (tmp_path/'probe/agent-started').exists()
    assert not (tmp_path/'probe/probe-timeout').exists()


def test_failure_in_lobby_restarts_the_agent_then_fences_without_client_input(tmp_path):
    result=run_protected(fault_config(tmp_path,prematch_retries=2,rejoin_delay_seconds=.1),998877,
        session='probe',notifier=OfflineNotifier(),
        _targets={'observer':lobby_observer,'agent':lobby_failure_agent,'client':prestart_protector,
                  'rejoin':never_ready_rejoin})
    report=json.loads((tmp_path/'probe/protection-result.json').read_text())
    assert result==1 and report['returned_lobby'] and not report['agent_acceptance']
    assert report['reason']['kind'] in ('agent_failure','agent_process_exited')
    assert (tmp_path/'probe/agent-termination.json').exists()
    assert not (tmp_path/'probe/client-002').exists()
    events=[json.loads(s) for s in (tmp_path/'probe/protection.jsonl').read_text().splitlines()]
    agents=[e for e in events if e['event']=='input_owner_started' and e['owner']=='agent']
    assert len(agents)==3 and report['recovery']['prematch_restarts']==2
    assert report['recovery']['latched']=='prematch_retries_exhausted'


def test_successful_simulated_match_keeps_agent_acceptance_without_client_input(tmp_path):
    result=run_protected(fault_config(tmp_path),998877,session='probe',notifier=OfflineNotifier(),
        _targets={'observer':generous_observer,'agent':successful_agent,'client':prestart_protector,
                  'rejoin':never_ready_rejoin})
    report=json.loads((tmp_path/'probe/protection-result.json').read_text())
    assert result==0 and report['returned_lobby'] and report['agent_acceptance']
    assert report['reason'] is None and not report['mixed_control']
    assert not (tmp_path/'probe/client-002').exists()
    assert not (tmp_path/'probe/probe-timeout').exists()


def test_steam_execute_cannot_accidentally_use_the_old_unguarded_runner(monkeypatch):
    from majsoul.live.capture import LiveWindow
    from majsoul.live.loop import LiveAgentLoop,LiveError
    window=LiveWindow.__new__(LiveWindow)
    cfg=LiveConfig.load(ROOT/'configs/live.steam.yaml')
    with pytest.raises(LiveError,match='runs under the watchdog'):
        LiveAgentLoop(window,cfg,execute=True,skip_menu=True)


@pytest.mark.parametrize('values',[{'heartbeat_seconds':float('nan')},{'enabled':1},
    {'shortest_turn_seconds':3},{'surprise':1},{'recovery_enabled':1},{'handback_stable_frames':1.5},
    {'handbacks_per_match':0},{'prematch_retries':-1},{'rejoin_delay_seconds':0}])
def test_invalid_protection_configuration_is_rejected(values):
    with pytest.raises(ValueError):ProtectionConfig.from_dict(values,ROOT)


@pytest.mark.parametrize('source,expected',[
    ('artifacts/live/acceptance-20261010/agent-recheck-172934/diagnostics/000411.jpg','actionable'),
    ('artifacts/live/acceptance-20261010/agent-recheck-172934/diagnostics/000623.jpg','afk'),
    ('artifacts/live/acceptance-20261010/agent-recheck-172934/operator-takeover-through-match/frames/00000018.jpg','connection'),
    ('artifacts/live/workflow-audit/recovery-live-20261011-172420/preflight.png','logout'),
    ('artifacts/live/workflow-audit/tile-audit-20261011-172934/closure-lobby-2.png','lobby'),
])
def test_lightweight_observer_replays_actual_table_and_lifecycle_without_cnn(source,expected):
    from types import SimpleNamespace
    from majsoul.calibration.frames import read_image
    from majsoul.live.protection_ui import ProtectionUI
    cfg=LiveConfig.load(ROOT/'configs/live.steam.yaml')
    reader=ProtectionUI(cfg)
    result=reader.read(SimpleNamespace(image=read_image(ROOT/source),captured_at=1))
    if expected in ('actionable','lobby'):
        assert result[expected]
    else:
        assert result['modal']==expected and not result['lobby']
    if expected=='actionable':assert result['total_seconds']==5


@pytest.mark.parametrize('count,drawn,actionable,waiting',[
    (14,None,False,False),(14,13,True,False),(13,None,False,True),
    (11,None,False,False),(11,10,True,False),
])
def test_hand_count_alone_without_clock_or_draw_geometry_is_uncertain(count,drawn,actionable,waiting):
    from types import SimpleNamespace
    from majsoul.calibration.frames import read_image
    from majsoul.live.protection_ui import ProtectionUI
    reader=ProtectionUI(LiveConfig.load(ROOT/'configs/live.steam.yaml'))
    reader.clock=None
    reader.buttons.read=lambda *_:{}
    reader.hand.segment=lambda *_:SimpleNamespace(tiles=tuple(range(count)),drawn_index=drawn,issues=())
    image=read_image(ROOT/'artifacts/live/protected-20261013-round03/client-002/frames/00000001.jpg')
    result=reader.read(SimpleNamespace(image=image,captured_at=1))
    assert result['actionable']==actionable and result['waiting']==waiting
    assert result['uncertain']==(not actionable and not waiting)


def test_unseparated_call_discard_still_trips_the_existing_uncertainty_watchdog():
    m=ProtectionMonitor(ProtectionConfig(),0)
    x=sample(0)
    x.update(actionable=False,uncertain=True,total_seconds=None)
    m.observe(x,0)
    m.event('heartbeat',{},.9)
    assert m.check(1)['kind']=='critical_local_ui_uncertain'


# ------------------------------------------------------------ temporary takeover

from majsoul.live.protection import RecoveryLedger,recovery_class,read_checkpoint


@pytest.mark.parametrize('reason,seen,lobby,expected',[
    ({'kind':'operator_stop'},True,False,'fatal'),
    ({'kind':'operator_interrupt'},False,True,'fatal'),
    ({'kind':'observation_error','error':'input_desktop_blocked'},True,False,'fatal'),
    ({'kind':'agent_failure','event':'worker_exception'},False,True,'prematch'),
    ({'kind':'pre_match_timeout'},False,True,'prematch'),
    ({'kind':'operation_clock_unrecognized'},True,False,'recoverable'),
    ({'kind':'critical_local_ui_uncertain'},True,False,'recoverable'),
    ({'kind':'agent_stalled'},True,False,'recoverable'),
    ({'kind':'agent_recognition_timeout'},False,False,'recoverable'),
    ({'kind':'modal','name':'table_connection'},True,False,'recoverable'),
])
def test_only_operator_stops_and_lock_screens_latch_protection(reason,seen,lobby,expected):
    assert recovery_class(reason,seen_match=seen,lobby_verified=lobby)==expected


def test_rearm_after_handback_starts_a_fresh_budget_without_old_samples():
    m=ProtectionMonitor(ProtectionConfig(),0,require_agent_recognition=True)
    m.event('agent_game_started',{},0)
    m.observe(sample(.1),.1)
    m.event('heartbeat',{},1.6)
    assert m.check(1.6)['kind']=='operation_deadline'
    m.observe(sample(1.7),1.7)
    m.rearm(2.0)
    assert m.reason is None and m.operation_due is None and m.agent_recognized
    m.observe(sample(1.9),2.1)  # Captured before the handback: ignored.
    assert m.operation_due is None
    m.observe(sample(2.2),2.2)
    assert m.operation_since==2.2 and m.operation_due==pytest.approx(3.7)
    m.event('heartbeat',{},3.6)
    assert m.check(3.6) is None


def test_result_overlays_advance_the_hand_epoch_once_each():
    m=ProtectionMonitor(ProtectionConfig(),0)
    for at,confirm in ((.1,None),(.2,'blue'),(.3,'blue'),(.4,None),(.5,'gold')):
        x=sample(at);x.update(confirm=confirm,actionable=False)
        m.observe(x,at)
    assert m.hand_epoch==2


def test_ledger_backs_off_flaps_and_bounds_handbacks():
    config=ProtectionConfig(rejoin_delay_seconds=1,rejoin_backoff_max_seconds=5,handbacks_per_hand=1,
                            handbacks_per_match=3,same_reason_limit=3,flap_window_seconds=10)
    ledger=RecoveryLedger(config)
    reason={'kind':'agent_stalled'}
    ledger.record_trip(reason,0,'recoverable')
    assert ledger.delay()==1 and ledger.stable_frames()==config.handback_stable_frames
    assert ledger.record_handback(4,hand=1)['recovery_seconds']==4
    assert not ledger.hand_allows(1) and ledger.hand_allows(2)
    ledger.record_trip(reason,6,'recoverable')  # Within the flap window.
    assert ledger.delay()==2 and ledger.stable_frames()==config.handback_stable_frames+2
    assert ledger.exhausted() is None
    ledger.record_trip(reason,30,'recoverable')
    assert ledger.exhausted()=='repeated_same_reason'
    assert RecoveryLedger(replace(config,recovery_enabled=False)).exhausted()=='recovery_disabled'


def test_ungranted_window_refuses_input_until_the_supervisor_grants_it(monkeypatch):
    from queue import Queue
    from types import SimpleNamespace
    import majsoul.live.capture as capture
    import majsoul.live.modals as modals
    sent=[]
    class Window:
        def __init__(self,hwnd):pass
        def click(self,*args,**kwargs):sent.append(args)
    monkeypatch.setattr(capture,'LiveWindow',Window)
    monkeypatch.setattr(modals,'input_desktop_status',lambda:{'status':'available'})
    granted=SimpleNamespace(value=False)
    window=_input_window(1,Queue(),SimpleNamespace(value=False),granted=granted)
    with pytest.raises(capture.FocusUnavailable,match='not granted'):window.click((1,1))
    assert not sent
    granted.value=True
    window.click((1,1))
    assert sent==[((1,1),)]


def test_checkpoint_is_only_a_well_formed_layout_hint(tmp_path):
    assert read_checkpoint(tmp_path) is None
    (tmp_path/'agent-checkpoint.json').write_text('{"active_slots": ["bottom", 3]}')
    assert read_checkpoint(tmp_path) is None
    (tmp_path/'agent-checkpoint.json').write_text('{"active_slots": ["bottom", "left", "right"]}')
    assert read_checkpoint(tmp_path)==('bottom','left','right')


def handback_config(tmp_path,**values):
    values={'heartbeat_seconds':2,'rejoin_delay_seconds':.2,'handback_stable_seconds':.1,**values}
    return fault_config(tmp_path,**values)


def protection_events(tmp_path):
    return [json.loads(s) for s in (tmp_path/'probe/protection.jsonl').read_text().splitlines()]


def test_temporary_takeover_hands_back_to_a_rejoined_agent_after_verified_release(tmp_path):
    result=run_protected(handback_config(tmp_path),998877,session='probe',notifier=OfflineNotifier(),
        _targets={'observer':table_observer,'agent':failing_once_agent,'client':handback_protector,
                  'rejoin':ready_rejoin})
    report=json.loads((tmp_path/'probe/protection-result.json').read_text())
    assert result==1 and report['returned_lobby'] and report['mixed_control']
    assert report['agent_in_control_at_end'] and not report['agent_acceptance']
    assert report['reason']['event']=='verification_failed'
    recovery=report['recovery']
    assert len(recovery['trips'])==1 and len(recovery['handbacks'])==1 and recovery['latched'] is None
    assert recovery['handbacks'][0]['recovery_seconds']<15
    assert not (tmp_path/'probe/late-input').exists() and not (tmp_path/'probe/probe-timeout').exists()
    granted=list((tmp_path/'probe').glob('granted-rejoin-*'))
    assert len(granted)==1 and 'MISSING' not in granted[0].read_text()
    events=protection_events(tmp_path)
    names=[e['event'] for e in events]
    requested=names.index('handback_requested')
    complete=names.index('handback_complete')
    # The protector is fenced (exit acknowledged) before the Agent is granted.
    fence=next(i for i,e in enumerate(events) if e['event']=='input_owner_exited' and e.get('reason')=='handback')
    assert requested<fence<complete
    assert events[complete]['owner']=='agent'


def test_no_handback_while_the_observer_shows_an_own_operation_window(tmp_path):
    (tmp_path/'own-turn').write_text('scenario')
    run_protected(handback_config(tmp_path,maximum_operation_seconds=1),998877,session='probe',
        notifier=OfflineNotifier(),
        _targets={'observer':own_turn_observer,'agent':failing_once_agent,'client':handback_protector,
                  'rejoin':ready_rejoin})
    report=json.loads((tmp_path/'probe/protection-result.json').read_text())
    assert report['returned_lobby'] and not report['recovery']['handbacks']
    assert not (tmp_path/'probe/released').exists()
    assert 'handback_requested' not in [e['event'] for e in protection_events(tmp_path)]


def test_rejected_release_keeps_protection_then_retries_the_handback(tmp_path):
    (tmp_path/'reject-first').write_text('scenario')
    run_protected(handback_config(tmp_path),998877,session='probe',notifier=OfflineNotifier(),
        _targets={'observer':table_observer,'agent':failing_once_agent,'client':handback_protector,
                  'rejoin':ready_rejoin})
    report=json.loads((tmp_path/'probe/protection-result.json').read_text())
    assert report['returned_lobby'] and len(report['recovery']['handbacks'])==1
    events=protection_events(tmp_path)
    names=[e['event'] for e in events]
    assert names.count('handback_requested')==2
    rejected=next(i for i,e in enumerate(events)
                  if e['event']=='worker_event' and e['source_event']=='handback_rejected')
    assert rejected<names.index('handback_complete')


def test_repeated_same_failure_after_handback_latches_protection_to_the_lobby(tmp_path):
    (tmp_path/'flap').write_text('scenario')
    run_protected(handback_config(tmp_path,same_reason_limit=2),998877,session='probe',
        notifier=OfflineNotifier(),
        _targets={'observer':table_observer,'agent':failing_once_agent,'client':handback_protector,
                  'rejoin':ready_rejoin})
    report=json.loads((tmp_path/'probe/protection-result.json').read_text())
    recovery=report['recovery']
    assert report['returned_lobby'] and not report['agent_in_control_at_end']
    assert len(recovery['trips'])==2 and len(recovery['handbacks'])==1
    assert recovery['latched']=='repeated_same_reason'
    assert recovery['trips'][1]['flap_streak']==1
    assert not (tmp_path/'probe/probe-timeout').exists()


def test_recovery_disabled_restores_the_previous_latched_behaviour(tmp_path):
    (tmp_path/'no-handback').write_text('scenario')
    run_protected(handback_config(tmp_path,recovery_enabled=False),998877,session='probe',
        notifier=OfflineNotifier(),
        _targets={'observer':table_observer,'agent':failing_once_agent,'client':handback_protector,
                  'rejoin':ready_rejoin})
    report=json.loads((tmp_path/'probe/protection-result.json').read_text())
    assert report['returned_lobby'] and report['recovery']['latched']=='recovery_disabled'
    assert not report['recovery']['handbacks'] and not list((tmp_path/'probe').glob('granted-*'))


def test_run_151224_replay_does_not_take_over_a_confirmed_turn():
    """protected-20261018-151224 tripped at 9465.2 on one unreadable reserve-only
    frame although the confirmed 25 s reading at 9460.36 still covered the turn."""
    run=ROOT/'artifacts/live/protected-20261018-151224'
    samples=[json.loads(line) for line in (run/'observer/samples.jsonl').read_text(encoding='utf-8').splitlines()]
    m=ProtectionMonitor(ProtectionConfig(),9404.6,require_agent_recognition=True)
    m.event('worker_ready',{},9420.8)
    m.event('table_wait_started',{},9429.6)
    agent=[(9454.328,'agent_game_started',{}),
           (9458.9,'heartbeat',{'stage':'input_end','input_completed':True}),  # nuki click
           (9460.796,'verified',{})]
    for item in samples:
        at=item['captured_at']
        if at>9465.3:break
        while agent and agent[0][0]<=at:
            when,event,fields=agent.pop(0)
            m.event(event,fields,when)
        m.event('heartbeat',{},at)
        m.observe(item,at)
        assert m.check(at) is None,item
    m.event('heartbeat',{},9465.3)
    assert m.check(9465.3) is None
    assert m.operation_due==pytest.approx(9460.359+8)


def test_reading_seen_before_a_late_completion_carries_into_the_next_window():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.observe(sample(0,total_seconds=25),0)
    m.event('heartbeat',{'stage':'input_end','input_completed':True},1)
    m.observe(sample(1.5,total_seconds=25),1.5)  # post-nuki window, fresh base
    m.event('verified',{},2)  # nuki verification arrives late
    x=sample(4);x['total_seconds']=None
    m.event('heartbeat',{},4);m.observe(x,4)
    assert m.check(4) is None
    assert m.operation_due==pytest.approx(9.5)  # cap counts from the carried first sight


def test_carried_reading_is_dropped_once_an_opponent_turn_is_seen():
    m=ProtectionMonitor(ProtectionConfig(),0)
    m.observe(sample(0,total_seconds=25),0)
    m.event('heartbeat',{'stage':'input_end','input_completed':True},1)
    m.observe(sample(1.5,total_seconds=25),1.5)
    m.event('verified',{},2)
    m.event('heartbeat',{},3);m.observe(sample(3,actionable=False,waiting=True),3)
    x=sample(4);x['total_seconds']=None
    m.event('heartbeat',{},4);m.observe(x,4)
    assert m.operation_due==pytest.approx(4.5)  # base-only from the waiting frame

