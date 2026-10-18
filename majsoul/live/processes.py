"""Process plumbing shared by the live controllers; it never plays the game.

Spawned Agent children, bounded IPC, the game-window lease and fenced process
shutdown. Moved out of ``protection`` (the retired takeover supervisor) so the
watchdog workflow does not depend on it.
"""
import json
from pathlib import Path
from queue import Empty, Full
import time

# Agent log events forwarded to the parent process.
FORWARDED_EVENTS = frozenset((
    'verified', 'window_resolved', 'emergency_verified', 'emergency_resolved',
    'verification_failed', 'fatal_error', 'emergency_failed', 'agent_stopped', 'worker_exception',
    'verification_unknown', 'preflight_abort', 'decision_expired', 'decision_failed', 'safe_pause',
    'emergency_unavailable', 'agent_ready_for_handback', 'input_granted',
    'heartbeat', 'worker_ready', 'worker_exit', 'table_wait_started', 'agent_game_started',
    'match_complete', 'emergency_clicked', 'modal_observed', 'modal_confirm_clicked'))


def _emit(queue, **message):
    if len(json.dumps(message,ensure_ascii=False,default=str))>1800:
        fields=message.get('fields') or {}
        message['fields']={k:fields[k] for k in ('reason','error','stage','exit_code','match_complete') if k in fields}
        message['fields']['details_truncated']=True
        for key in ('reason','error'):
            if key in message['fields']:message['fields'][key]=str(message['fields'][key])[:900]
    try:
        queue.put_nowait(dict(message, emitted_at=time.monotonic()))
    except Full:
        pass  # A full IPC queue must not block capture or input.


def _drain(queue, maximum):
    """Bound IPC work so a busy producer cannot starve deadline checks."""
    for _ in range(maximum):
        try:
            yield queue.get_nowait()
        except (Empty, EOFError, OSError):
            return


def _input_window(hwnd,output,active,*,ignore_stop_key=False,on_heartbeat=None,granted=None):
    from .capture import LiveWindow
    def heartbeat(**fields):
        if on_heartbeat is not None:on_heartbeat(fields)
        _emit(output,kind='event',event='heartbeat',fields=fields)
    class Window(LiveWindow):
        protected_owner = True
        def capture(self):
            heartbeat(stage='capture_begin')
            frame = super().capture()
            heartbeat(stage='capture_done')
            return frame
        def click(self,*args,**kwargs):
            from .modals import input_desktop_status
            from .capture import FocusUnavailable
            if granted is not None and not granted.value:
                # An observe-only rejoin Agent never owns input before the
                # supervisor has fenced the protector and granted it.
                raise FocusUnavailable('Input not granted by the supervisor; no input sent')
            desktop=input_desktop_status()
            if desktop['status']=='unknown':
                raise FocusUnavailable('Desktop observation unavailable; no input sent: '+json.dumps(desktop))
            if desktop['status']!='available':
                raise RuntimeError('input_desktop_blocked: '+json.dumps(desktop))
            if active is not None:active.value=True
            heartbeat(stage='input_begin')
            try:
                return super().click(*args,**kwargs)
            finally:
                if active is not None:active.value=False
                heartbeat(stage='input_end',
                    input_completed=bool(getattr(self,'last_click_timing',{}).get('input_completed')))
    window=Window(hwnd);window.ignore_stop_key=ignore_stop_key
    return window


CHECKPOINT = 'agent-checkpoint.json'


def read_checkpoint(root):
    """The match-level seat layout from this supervised run, or None.

    It is only a hint: the rejoin Agent still requires the layout among the
    fresh metadata candidates and a validated decision snapshot."""
    try:
        data = json.loads((root/CHECKPOINT).read_text(encoding='utf-8'))
        slots = data.get('active_slots')
        if isinstance(slots, list) and slots and all(isinstance(s, str) for s in slots):
            return tuple(slots)
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _run_agent(config, hwnd, session, output, active, *, granted=None, activation=None,
               ready=None, **options):
    import faulthandler
    from .loop import LiveAgentLoop
    directory=config.artifacts_root/session
    directory.mkdir(parents=True,exist_ok=True)
    trace=(directory/'stall-stacks.log').open('w',encoding='utf-8')
    tracing=False
    def arm_trace(_fields=None):
        if tracing:
            faulthandler.dump_traceback_later(config.watchdog.heartbeat_seconds / 2,
                                             file=trace)
    def pulse(event, **fields):
        nonlocal tracing
        if event=='worker_ready':tracing=True
        if tracing and event in {'heartbeat','worker_ready'}:
            # Capture a real stack before the supervisor fences a stalled
            # process. Re-arm only on actual work, never on a timer heartbeat.
            arm_trace()
        if event in FORWARDED_EVENTS:
            _emit(output,kind='event',event=event,fields=fields)
    class Agent(LiveAgentLoop):
        def log(self,event,**fields):
            pulse(event,**fields)
            return super().log(event,**fields)
    try:
        loop = Agent(_input_window(hwnd,output,active,on_heartbeat=arm_trace,granted=granted),config,
                     execute=True,session=session,supervised=True,
                     checkpoint=config.artifacts_root/CHECKPOINT,**options)
        if activation is not None:
            # A warm standby has loaded every model; it neither captures nor
            # emits heartbeats until the supervisor activates it.
            _emit(output,kind='event',event='worker_ready',fields={'warm':True})
            if ready is not None:ready.set()
            activation.wait()
            loop.set_layout_hint(read_checkpoint(config.artifacts_root))
        pulse('worker_ready')
        code = loop.run()
        pulse('worker_exit',exit_code=code)
    except BaseException as exc:
        pulse('worker_exception',error=f'{type(exc).__name__}: {exc}')
    finally:
        faulthandler.cancel_dump_traceback_later()
        trace.close()


class WindowLease:
    """Nonblocking OS lock; supervisor holds it through verified lobby exit."""
    def __init__(self,directory,hwnd):
        self.path=directory/f'window-{hwnd}.lease'
        self.file=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=self.path.open('a+b')
        self.file.seek(0,2)
        if self.file.tell()==0:
            self.file.write(b'0');self.file.flush()
        self.file.seek(0)
        try:
            import os
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.file.close();self.file=None
            raise RuntimeError('Another controller owns this game window')
        return self
    def __exit__(self,*args):
        if self.file:
            self.file.close();self.file=None


def stop_input_owner(process, grace):
    """An exit acknowledgment is mandatory before granting another input owner."""
    if process is None:
        return
    if process.is_alive():
        process.terminate()
    process.join(grace)
    if process.is_alive():
        process.kill()
        process.join(1.0)
    if process.is_alive():
        raise RuntimeError('Old input owner did not exit; replacement input forbidden')
