"""Deterministic subprocess fault injection. No window, network or game input."""
from pathlib import Path
import time

from majsoul.live.protection import _emit, _standby


class OfflineNotifier:
    def notify_text(self,*args,**kwargs):return True


def observer(config,hwnd,output,stop):
    root=config.artifacts_root/'probe'
    deadline=time.monotonic()+15
    while not stop.is_set():
        lobby=_probe_lobby(root,deadline)
        # These cases test failures AFTER the Agent owns input. A separate
        # prestart case exposes the operation during standby initialization.
        actionable=not lobby and (root/'agent-started').exists()
        _emit(output,kind='sample',sample={'captured_at':time.monotonic(),
            'table':not lobby,'lobby':lobby,'actionable':actionable,'waiting':not actionable and not lobby,
            'total_seconds':(None if config.protection.maximum_operation_seconds<1 else 5) if actionable else None})
        stop.wait(.02)


def _probe_lobby(root,deadline):
    if time.monotonic()>deadline and not (root/'lobby').exists():
        # Bound a broken offline harness. Tests explicitly reject this marker;
        # this is never reported as a successful simulated return.
        (root/'probe-timeout').write_text('fault injection failed to finish')
        return True
    return (root/'lobby').exists()


def hung_agent(config,hwnd,session,skip_menu,once,output,active=None):
    _emit(output,kind='event',event='worker_ready',fields={})
    _emit(output,kind='event',event='agent_game_started',fields={})
    (config.artifacts_root/'agent-started').write_text(str(time.monotonic()))
    time.sleep(30)
    (config.artifacts_root/'late-input').write_text('must never happen')


def responsive_but_blocked_agent(config,hwnd,session,skip_menu,once,output,active=None):
    _emit(output,kind='event',event='worker_ready',fields={})
    _emit(output,kind='event',event='agent_game_started',fields={})
    (config.artifacts_root/'agent-started').write_text(str(time.monotonic()))
    while True:
        _emit(output,kind='event',event='heartbeat',fields={'stage':'capture_done'})
        time.sleep(.02)


def verification_failure_agent(config,hwnd,session,skip_menu,once,output,active=None):
    _emit(output,kind='event',event='worker_ready',fields={})
    _emit(output,kind='event',event='agent_game_started',fields={})
    (config.artifacts_root/'agent-started').write_text(str(time.monotonic()))
    _emit(output,kind='event',event='verification_unknown',fields={'reason':'response_still_open'})
    time.sleep(30)


def protector(config,hwnd,directory,output,release_mouse=False,active=None,release=None):
    root=directory.parent
    assert (root/'agent-termination.json').exists(),'replacement input before exit acknowledgment'
    directory.mkdir(parents=True)
    (directory/'input-owner-started').write_text(str(time.monotonic()))
    (directory/'release-mouse').write_text(str(release_mouse))
    _emit(output,kind='event',event='worker_ready',fields={})
    (root/'lobby').write_text('simulated verified return, no new match')
    while True:
        _emit(output,kind='event',event='heartbeat',fields={'stage':'protect_capture'})
        time.sleep(.02)


def first_protector_hangs(config,hwnd,directory,output,release_mouse=False,active=None,release=None):
    if directory.name=='client-002':
        directory.mkdir(parents=True)
        (directory/'hung').write_text('simulated PrintWindow hang')
        _emit(output,kind='event',event='worker_ready',fields={})
        time.sleep(30)
    else:
        protector(config,hwnd,directory,output,release_mouse,active)


def waiting_observer(config,hwnd,output,stop):
    root=config.artifacts_root/'probe'
    deadline=time.monotonic()+15
    while not stop.is_set():
        lobby=_probe_lobby(root,deadline)
        _emit(output,kind='sample',sample={'captured_at':time.monotonic(),
            'table':not lobby,'lobby':lobby,'actionable':False,'waiting':not lobby,'total_seconds':None})
        stop.wait(.02)


def input_hung_agent(config,hwnd,session,skip_menu,once,output,active=None):
    _emit(output,kind='event',event='worker_ready',fields={})
    _emit(output,kind='event',event='agent_game_started',fields={})
    active.value=True
    time.sleep(30)


def delayed_standby(*args):
    time.sleep(.7)
    _standby(*args)


def prestart_observer(config,hwnd,output,stop):
    root=config.artifacts_root/'probe'
    deadline=time.monotonic()+15
    while not stop.is_set():
        lobby=_probe_lobby(root,deadline)
        _emit(output,kind='sample',sample={'captured_at':time.monotonic(),
            'table':not lobby,'lobby':lobby,'actionable':not lobby,'waiting':False,
            'total_seconds':None})
        stop.wait(.02)


def no_agent_expected(config,hwnd,session,skip_menu,once,output,active=None):
    (config.artifacts_root/'forbidden-agent').write_text('Agent started after protection deadline')
    time.sleep(30)


def prestart_protector(config,hwnd,directory,output,release_mouse=False,active=None,release=None):
    directory.mkdir(parents=True)
    (directory.parent/'lobby').write_text('current match returned without Agent input')
    while True:
        _emit(output,kind='event',event='heartbeat',fields={'stage':'protect_capture'})
        time.sleep(.02)


def lobby_observer(config,hwnd,output,stop):
    while not stop.is_set():
        _emit(output,kind='sample',sample={'captured_at':time.monotonic(),
            'table':False,'lobby':True,'actionable':False,'waiting':False,'total_seconds':None})
        stop.wait(.02)


def lobby_failure_agent(config,hwnd,session,skip_menu,once,output,active=None):
    _emit(output,kind='event',event='worker_ready',fields={})
    time.sleep(.8)
    _emit(output,kind='event',event='worker_exception',fields={'error':'initialization failed in lobby'})


def stopping_agent(config,hwnd,session,skip_menu,once,output,active=None):
    _emit(output,kind='event',event='worker_ready',fields={})
    _emit(output,kind='event',event='agent_game_started',fields={})
    (config.artifacts_root/'STOP').write_text('operator requests protected stop')
    time.sleep(30)


def generous_observer(config,hwnd,output,stop):
    root=config.artifacts_root/'probe'
    deadline=time.monotonic()+15
    while not stop.is_set():
        lobby=_probe_lobby(root,deadline)
        _emit(output,kind='sample',sample={'captured_at':time.monotonic(),
            'table':not lobby,'lobby':lobby,'actionable':not lobby,'waiting':False,
            'total_seconds':None if lobby else 27})
        stop.wait(.02)


def successful_agent(config,hwnd,session,skip_menu,once,output,active=None):
    _emit(output,kind='event',event='worker_ready',fields={})
    _emit(output,kind='event',event='agent_game_started',fields={})
    _emit(output,kind='event',event='verified',fields={})
    (config.artifacts_root/'lobby').write_text('successful simulated match')
    while True:
        _emit(output,kind='event',event='heartbeat',fields={'stage':'capture_done'})
        time.sleep(.02)


def unrecognized_agent(config,hwnd,session,skip_menu,once,output,active=None):
    _emit(output,kind='event',event='worker_ready',fields={})
    _emit(output,kind='event',event='table_wait_started',fields={})
    (config.artifacts_root/'agent-started').write_text('recognition in progress')
    while True:
        _emit(output,kind='event',event='heartbeat',fields={'stage':'capture_done'})
        time.sleep(.02)


# ------------------------------------------------------- temporary takeover
# These fakes exercise the handback protocol: trip, protect, observe-only
# rejoin, protector release at an opponent turn, fence, then grant. Scenario
# flags live beside the (initially empty) session directory.

def never_ready_rejoin(config,hwnd,session,output,active,granted,activation,ready,stable_frames):
    _emit(output,kind='event',event='worker_ready',fields={'warm':True})
    ready.set();activation.wait()
    while True:
        if granted.value:
            (config.artifacts_root/'late-input').write_text('never-ready rejoin was granted')
        _emit(output,kind='event',event='heartbeat',fields={'stage':'capture_done'})
        time.sleep(.02)


def table_observer(config,hwnd,output,stop):
    """An opponent turn on a live table until the lobby marker appears."""
    root=config.artifacts_root/'probe'
    deadline=time.monotonic()+20
    while not stop.is_set():
        lobby=_probe_lobby(root,deadline)
        _emit(output,kind='sample',sample={'captured_at':time.monotonic(),
            'table':not lobby,'lobby':lobby,'actionable':False,'waiting':not lobby,'total_seconds':None})
        stop.wait(.02)


def own_turn_observer(config,hwnd,output,stop):
    """An own operation window with a generous clock; never a safe handback point."""
    root=config.artifacts_root/'probe'
    deadline=time.monotonic()+20
    while not stop.is_set():
        lobby=_probe_lobby(root,deadline)
        _emit(output,kind='sample',sample={'captured_at':time.monotonic(),
            'table':not lobby,'lobby':lobby,'actionable':not lobby,'waiting':False,
            'total_seconds':None if lobby else 27})
        stop.wait(.02)


def failing_once_agent(config,hwnd,session,skip_menu,once,output,active=None):
    _emit(output,kind='event',event='worker_ready',fields={})
    _emit(output,kind='event',event='agent_game_started',fields={})
    (config.artifacts_root/'agent-started').write_text(str(time.monotonic()))
    time.sleep(.3)
    _emit(output,kind='event',event='verification_failed',fields={'reason':'simulated wrong transition'})
    time.sleep(30)
    (config.artifacts_root/'late-input').write_text('fenced Agent survived')


def _released_idle(output):
    while True:
        _emit(output,kind='event',event='heartbeat',fields={'stage':'released_idle'})
        time.sleep(.02)


def handback_protector(config,hwnd,directory,output,release_mouse=False,active=None,release=None):
    root=directory.parent
    assert (root/'agent-termination.json').exists(),'replacement input before exit acknowledgment'
    directory.mkdir(parents=True)
    (directory/'input-owner-started').write_text(str(time.monotonic()))
    _emit(output,kind='event',event='worker_ready',fields={})
    if (root/'flapped').exists() or (root.parent/'no-handback').exists():
        (root/'lobby').write_text('latched protection finished the match')
    rejected=False
    started=time.monotonic()
    while True:
        if (root.parent/'reject-first').exists() and release is not None and release.is_set() and not rejected:
            rejected=True
            release.clear()
            (root/'rejected').write_text(str(time.monotonic()))
            _emit(output,kind='event',event='handback_rejected',fields={'error':'simulated toggle still ON'})
        elif release is not None and release.is_set():
            (root/'released').write_text(str(time.monotonic()))
            _emit(output,kind='event',event='handback_released',fields={'missing':[]})
            _released_idle(output)
        if (root.parent/'own-turn').exists() and time.monotonic()-started>2.5:
            (root/'lobby').write_text('no safe handback point appeared; protected to lobby')
        _emit(output,kind='event',event='heartbeat',fields={'stage':'protect_capture'})
        time.sleep(.02)


def ready_rejoin(config,hwnd,session,output,active,granted,activation,ready,stable_frames):
    root=config.artifacts_root
    _emit(output,kind='event',event='worker_ready',fields={'warm':True})
    ready.set();activation.wait()
    _emit(output,kind='event',event='worker_ready',fields={})
    while not granted.value:
        _emit(output,kind='event',event='heartbeat',fields={'stage':'capture_done'})
        _emit(output,kind='event',event='agent_ready_for_handback',fields={'frames':stable_frames.value})
        time.sleep(.02)
    released=root/'released'
    (root/f'granted-{session}').write_text(str(time.monotonic())+' after release '+
        (released.read_text() if released.exists() else 'MISSING'))
    _emit(output,kind='event',event='input_granted',fields={})
    if (root.parent/'flap').exists() and not (root/'flapped').exists():
        (root/'flapped').write_text(session)
        time.sleep(.2)
        _emit(output,kind='event',event='verification_failed',fields={'reason':'simulated repeat'})
        time.sleep(30)
    _emit(output,kind='event',event='verified',fields={})
    (root/'lobby').write_text('rejoined Agent finished the match')
    while True:
        _emit(output,kind='event',event='heartbeat',fields={'stage':'capture_done'})
        time.sleep(.02)
