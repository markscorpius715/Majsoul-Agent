"""Watchdog for the live Agent; it never plays the game.

The Agent owns all game input for the whole match and carries its own fallback
chain (model -> local efficiency -> tsumogiri/PASS -> the client's timeout).
This parent process only keeps an Agent alive: when the Agent crashes, stalls
or cannot start, the watchdog fences it and promotes a pre-loaded standby (or
starts a cold Agent that joins the running match). It sends no game input,
never enables client auto-play and never starts another match.
"""
from dataclasses import dataclass, replace
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

from .processes import (WindowLease, _drain, _run_agent, read_checkpoint,
                        stop_input_owner)


@dataclass(frozen=True)
class WatchdogConfig:
    enabled: bool = True
    heartbeat_seconds: float = 2.0       # Agent capture heartbeat cadence (menu polling, stall stacks)
    stall_seconds: float = 8.0           # restart a ready Agent that is silent this long
    startup_seconds: float = 120.0       # model loading limit before worker_ready
    recognition_seconds: float = 30.0    # Agent's table-start recognition window; it then restarts
    restart_limit: int = 10              # restarts per match before the watchdog gives up
    restart_backoff_seconds: float = 1.0
    restart_backoff_max_seconds: float = 10.0
    warm_standby: bool = True            # keep a loaded Agent copy ready to take over at once
    standby_delay_seconds: float = 15.0  # let an Agent settle before loading another copy
    poll_seconds: float = 0.1
    stop_grace_seconds: float = 0.5
    max_matches: int = 0                 # matches to play back to back; 0 = until STOP/Ctrl+C

    BOOLEANS = ('enabled', 'warm_standby')
    COUNTS = ('restart_limit', 'max_matches')

    @classmethod
    def from_dict(cls, data):
        values = dict(data or {})
        if set(values)-set(cls.__dataclass_fields__):
            raise ValueError('watchdog contains unsupported configuration')
        obj = cls(**values)
        for key in cls.BOOLEANS:
            if type(getattr(obj, key)) is not bool:
                raise ValueError(f'watchdog.{key} must be boolean')
        if type(obj.restart_limit) is not int or not 0 <= obj.restart_limit <= 100:
            raise ValueError('Invalid watchdog restart_limit')
        if type(obj.max_matches) is not int or not 0 <= obj.max_matches <= 100000:
            raise ValueError('Invalid watchdog max_matches')
        for key, value in vars(obj).items():
            if key not in cls.BOOLEANS+cls.COUNTS and (
                    type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 600):
                raise ValueError('Invalid watchdog '+key)
        if obj.stall_seconds <= obj.heartbeat_seconds:
            raise ValueError('watchdog.stall_seconds must exceed heartbeat_seconds')
        return obj


TERMINAL_WINDOW_ERRORS = ('Window closed', 'Window handle does not exist')
# Evidence that a restarted Agent is playing again.
RESUME_EVENTS = ('agent_game_started', 'verified', 'emergency_verified', 'window_resolved')


def _agent_child(config, hwnd, session, output, active, granted, activation, ready, join, skip_menu, once):
    """One Agent process. A standby gets ``granted``/``activation``: it loads
    every model, then waits; promotion grants input before it starts capturing."""
    options = dict(skip_menu=True, join=True) if join else dict(skip_menu=skip_menu, once=once)
    if granted is not None:
        options['input_gate'] = lambda: bool(granted.value)
    if join and activation is None:
        options['layout_hint'] = read_checkpoint(config.artifacts_root)
    _run_agent(config, hwnd, session, output, active, granted=granted, activation=activation,
               ready=ready, **options)


def _release_left_button():
    """Safety only, not a game action: a fenced Agent may die mid-click."""
    from ..window import _user32
    from .modals import input_desktop_available
    if input_desktop_available():
        _user32().mouse_event(0x0004, 0, 0, 0, 0)


def run_watchdog(config, hwnd, *, skip_menu=False, once=False, session=None, notifier=None,
                 _targets=None, _clock=time.monotonic, _sleep=time.sleep):
    """Play matches back to back with the Agent in control (``max_matches``,
    0 = until STOP/Ctrl+C). Each match gets a fresh Agent that enters it from
    the lobby. Returns 0 when every finished match was an Agent acceptance."""
    from .notify import FeishuNotifier
    from .notifications import NotificationQueue
    name = session or ('live-'+time.strftime('%Y%m%d-%H%M%S')+f'-{time.time_ns() % 1000000:06d}')
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('session must be a single directory name')
    root = config.artifacts_root/name
    if root.exists() and any(root.iterdir()):
        raise ValueError('Session directory is not empty')
    root.mkdir(parents=True, exist_ok=True)
    wc = config.watchdog
    targets = _targets or {}
    agent_config = replace(config, artifacts_root=root)
    ctx = mp.get_context('spawn')
    state = {'agents': 0, 'match_seen': False, 'matching': False, 'complete': False,
             'restarts': [], 'emergency_clicks': 0, 'stop': None, 'interrupt': False,
             'matches': [], 'match_index': 1}

    def log(event, **fields):
        row = dict(event=event, monotonic=round(_clock(), 3), wall_time=time.time(), **fields)
        line = json.dumps(row, ensure_ascii=False, default=str)
        with (root/'watchdog.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
        print(line, flush=True)

    queue = NotificationQueue(notifier or FeishuNotifier(config.notify), root/'notifications',
                              capacity=config.notify.queue_capacity, logger=log,
                              max_attempts=config.notify.max_attempts, retry_seconds=config.notify.retry_seconds)

    def notify(text):
        queue.submit('notify_text', '[majsoul-watchdog] '+text)

    def spawn(*, join, standby=False):
        state['agents'] += 1
        index = state['agents']
        session_name = 'agent' if index == 1 else f'agent-{index:03d}'
        bundle = dict(queue=ctx.Queue(maxsize=128), active=ctx.Value('b', False, lock=False),
                      granted=ctx.Value('b', False, lock=False) if standby else None,
                      activation=ctx.Event() if standby else None,
                      ready_event=ctx.Event() if standby else None,
                      session=session_name, standby=standby, started_at=_clock(), progress=_clock(),
                      ready=False, failure=None, exit_code=None, resume=None)
        # Only the first match may skip the menus; later ones enter from the lobby.
        menu_skip = skip_menu and not state['matches']
        bundle['process'] = ctx.Process(
            target=targets.get('agent', _agent_child),
            args=(agent_config, hwnd, session_name, bundle['queue'], bundle['active'], bundle['granted'],
                  bundle['activation'], bundle['ready_event'], join, menu_skip, once),
            name='majsoul-agent', daemon=True)
        bundle['process'].start()
        log('agent_spawned', session=session_name, pid=bundle['process'].pid, join=join, standby=standby,
            match=state['match_index'])
        return bundle

    def drain(bundle, now):
        for item in _drain(bundle['queue'], 128):
            event, fields = item.get('event'), item.get('fields', {})
            bundle['progress'] = max(bundle['progress'], min(now, item.get('emitted_at', now)))
            if event == 'worker_ready':
                bundle['ready'] = True
            elif event in ('worker_exception', 'fatal_error', 'agent_stopped'):
                bundle['failure'] = {'kind': event, 'error': fields.get('error') or fields.get('reason')}
            elif event == 'worker_exit':
                bundle['exit_code'] = fields.get('exit_code')
            if bundle['standby'] and not bundle['granted'].value:
                continue  # A loading standby only reports readiness or death.
            if bundle.get('resume') is not None and event in RESUME_EVENTS:
                # User request 2026-10-18: report successful recoveries too.
                restart = bundle['resume']
                bundle['resume'] = None
                seconds = round(now-restart['at'], 1)
                log('agent_resumed', session=bundle['session'], after_seconds=seconds,
                    evidence=event, restart=restart['number'])
                notify(f'Agent resumed control ({bundle["session"]}, restart {restart["number"]}) '
                       f'{seconds} s after the failure; evidence: {event}.')
            if event == 'table_wait_started':
                state['matching'] = True
            elif event == 'agent_game_started':
                state['match_seen'] = True
            elif event == 'match_complete':
                state['complete'] = True
            elif event == 'emergency_clicked':
                state['emergency_clicks'] += 1
            if event != 'heartbeat':
                log('agent_event', session=bundle['session'], source_event=event, details=fields)

    def problem(bundle, now):
        if not bundle['process'].is_alive():
            return {'kind': 'agent_exited', 'exit_code': bundle['process'].exitcode,
                    'failure': bundle['failure']}
        if bundle['failure']:
            return dict(bundle['failure'])
        if bundle['ready'] and now-bundle['progress'] > wc.stall_seconds:
            return {'kind': 'agent_stalled', 'silent_seconds': round(now-bundle['progress'], 3)}
        if not bundle['ready'] and now-bundle['started_at'] > wc.startup_seconds:
            return {'kind': 'agent_startup_timeout'}
        return None

    def fence(bundle, reason):
        held = bool(bundle['active'].value)
        stop_input_owner(bundle['process'], wc.stop_grace_seconds)
        if held:
            _release_left_button()
        log('agent_fenced', session=bundle['session'], pid=bundle['process'].pid,
            exit_code=bundle['process'].exitcode, reason=reason, mouse_released=held)

    def error_text(reason):
        failure = reason.get('failure') or {}
        return str(reason.get('error') or failure.get('error') or '')

    owner = standby = None
    respawn_at = None
    standby_at = _clock()+wc.standby_delay_seconds
    result_reason = None
    import signal

    def interrupt(*_):
        state['interrupt'] = True

    with WindowLease(config.artifacts_root/'control-leases', hwnd):
        previous = signal.signal(signal.SIGINT, interrupt)
        try:
            owner = spawn(join=False)
            while True:
                now = _clock()
                for bundle in (owner, standby):
                    if bundle is not None:
                        drain(bundle, now)
                if (root/'STOP').exists() or state['interrupt']:
                    result_reason = {'kind': 'operator_interrupt' if state['interrupt'] else 'operator_stop'}
                    log('operator_stop', reason=result_reason)
                    break
                if owner is not None and state['complete'] and not owner['process'].is_alive():
                    finished = {'match': state['match_index'], 'returned_lobby': True,
                                'agent_acceptance': not state['restarts'],
                                'agent_restarts': state['restarts'],
                                'emergency_rung_clicks': state['emergency_clicks']}
                    state['matches'].append(finished)
                    log('match_complete', session=owner['session'], **finished)
                    state.update(match_seen=False, matching=False, complete=False, restarts=[],
                                 emergency_clicks=0)
                    if wc.max_matches and len(state['matches']) >= wc.max_matches:
                        break
                    # Back to back: a fresh Agent enters the next match from the lobby.
                    if standby is not None:
                        stop_input_owner(standby['process'], wc.stop_grace_seconds)
                        standby = None
                    (root/'agent-checkpoint.json').unlink(missing_ok=True)
                    state['match_index'] += 1
                    notify(f'Match {finished["match"]} finished (restarts {len(finished["agent_restarts"])}); '
                           f'starting match {state["match_index"]}.')
                    owner = spawn(join=False)
                    standby_at = _clock()+wc.standby_delay_seconds
                    continue
                if owner is not None and once and owner['exit_code'] == 0 and not owner['process'].is_alive():
                    log('once_complete', session=owner['session'])
                    break
                reason = problem(owner, now) if owner is not None else None
                if reason is not None and state['complete']:
                    # The lobby was reached (e.g. the match ended during a lock
                    # screen and the replacement found the lobby): the match is
                    # over, not crashed. Finish it and start the next one.
                    if owner['process'].is_alive():
                        stop_input_owner(owner['process'], wc.stop_grace_seconds)
                    reason = None
                if reason is not None:
                    fence(owner, reason)
                    owner = None
                    text = error_text(reason)
                    if any(marker in text for marker in TERMINAL_WINDOW_ERRORS):
                        result_reason = {'kind': 'window_gone', 'error': text}
                        notify('The game window is gone; the watchdog stops. No input was sent.')
                        log('window_gone', reason=reason)
                        break
                    state['restarts'].append({'at': round(now, 3), 'reason': reason})
                    if len(state['restarts']) > wc.restart_limit:
                        result_reason = {'kind': 'restart_budget_exhausted', 'last': reason}
                        notify(f'Agent failed {len(state["restarts"])} times; the watchdog stops. '
                               'The match continues only on client timeouts. Last reason: '
                               +json.dumps(reason, ensure_ascii=False))
                        log('restart_budget_exhausted', restarts=len(state['restarts']))
                        break
                    if 'input_desktop_blocked' in text:
                        notify('The desktop is locked; waiting for it to unlock before restarting the Agent.')
                    if standby is not None and standby['ready'] and standby['process'].is_alive():
                        standby['granted'].value = True
                        standby['activation'].set()
                        owner, standby = standby, None
                        owner['progress'] = now
                        owner['resume'] = {'at': now, 'number': len(state['restarts'])}
                        log('standby_promoted', session=owner['session'], reason=reason)
                        notify(f'Agent restarted from the loaded standby (restart {len(state["restarts"])}); '
                               'reason: '+json.dumps(reason, ensure_ascii=False))
                    else:
                        delay = min(wc.restart_backoff_seconds*2**max(0, len(state['restarts'])-1),
                                    wc.restart_backoff_max_seconds)
                        respawn_at = now+delay
                        log('restart_scheduled', after_seconds=delay, reason=reason)
                        notify(f'Agent restarting (restart {len(state["restarts"])}); the client timeout covers '
                               'turns meanwhile. Reason: '+json.dumps(reason, ensure_ascii=False))
                    standby_at = now+wc.standby_delay_seconds
                if owner is None and respawn_at is not None and now >= respawn_at:
                    from .modals import input_desktop_available
                    if targets.get('agent') is not None or input_desktop_available():
                        # Joining is required once matchmaking began: a menu run
                        # could queue a second match.
                        owner = spawn(join=state['matching'] or state['match_seen'])
                        owner['resume'] = {'at': state['restarts'][-1]['at'] if state['restarts'] else now,
                                           'number': len(state['restarts'])}
                        respawn_at = None
                if standby is not None and not standby['process'].is_alive():
                    log('standby_exited', session=standby['session'], exit_code=standby['process'].exitcode)
                    standby = None
                    standby_at = now+wc.standby_delay_seconds
                if (wc.warm_standby and standby is None and owner is not None and owner['ready']
                        and state['match_seen'] and now >= standby_at):
                    standby = spawn(join=True, standby=True)
                _sleep(wc.poll_seconds)
        finally:
            signal.signal(signal.SIGINT, previous)
            for bundle in (owner, standby):
                if bundle is not None:
                    stop_input_owner(bundle['process'], wc.stop_grace_seconds)
            matches = state['matches']
            result = {
                'workflow': 'watchdog', 'matches': matches, 'matches_completed': len(matches),
                'returned_lobby': bool(matches),
                'agent_acceptance': bool(matches) and all(m['agent_acceptance'] for m in matches),
                'agent_in_control_at_end': bool(matches),
                'mixed_control': False,
                'agent_restarts': [r for m in matches for r in m['agent_restarts']]+state['restarts'],
                'emergency_rung_clicks': sum(m['emergency_rung_clicks'] for m in matches)+state['emergency_clicks'],
                'unfinished_match': None if not (state['match_seen'] or state['matching'])
                else {'match': state['match_index'], 'agent_restarts': state['restarts']},
                'reason': result_reason}
            text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
            (root/'watchdog-result.json').write_text(text, encoding='utf-8')
            queue.close(config.notify.shutdown_seconds)
    return 0 if result['agent_acceptance'] else 1
