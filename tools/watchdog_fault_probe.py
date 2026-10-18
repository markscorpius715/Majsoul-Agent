"""Fake Agent children for offline watchdog tests; they never touch a window.

Each child has the watchdog's child signature and records how it was started
in ``<artifacts_root>/<session>.json`` so tests can check join and promotion.
"""
import json
import time

from majsoul.live.processes import _emit


def _record(config, session, **fields):
    (config.artifacts_root/f'{session}.json').write_text(json.dumps(fields), encoding='utf-8')


def _wait_promotion(granted, activation, ready, output):
    """A standby loads, reports readiness, then waits to be granted input."""
    _emit(output, kind='event', event='worker_ready', fields={'warm': True})
    ready.set()
    activation.wait()
    if not granted.value:
        raise AssertionError('standby activated without input')


def _play(output, seconds, *, game=True, complete=True):
    _emit(output, kind='event', event='worker_ready', fields={})
    if game:
        _emit(output, kind='event', event='agent_game_started', fields={})
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        _emit(output, kind='event', event='heartbeat', fields={'stage': 'capture_done'})
        time.sleep(.02)
    if complete:
        _emit(output, kind='event', event='match_complete', fields={})
        _emit(output, kind='event', event='worker_exit', fields={'exit_code': 0})


def completing_agent(config, hwnd, session, output, active, granted, activation, ready, join, skip_menu, once):
    _record(config, session, join=join, standby=granted is not None)
    if granted is not None:
        _wait_promotion(granted, activation, ready, output)
    _play(output, .3)


def crash_then_complete(config, hwnd, session, output, active, granted, activation, ready, join, skip_menu, once):
    """The first Agent plays a little and then crashes; any later one completes."""
    _record(config, session, join=join, standby=granted is not None)
    if granted is not None:
        _wait_promotion(granted, activation, ready, output)
        _play(output, .2)
        return
    if session != 'agent':
        _play(output, .2)
        return
    _play(output, 1.5, complete=False)
    _emit(output, kind='event', event='worker_exception', fields={'error': 'RuntimeError: boom'})


def hang_then_complete(config, hwnd, session, output, active, granted, activation, ready, join, skip_menu, once):
    _record(config, session, join=join, standby=granted is not None)
    if session == 'agent':
        _play(output, .2, complete=False)
        time.sleep(60)  # silent: no heartbeats
    _play(output, .2)


def window_closed(config, hwnd, session, output, active, granted, activation, ready, join, skip_menu, once):
    _record(config, session, join=join, standby=granted is not None)
    _play(output, .1, complete=False)
    _emit(output, kind='event', event='worker_exception', fields={'error': 'WindowError: Window closed'})


def always_crash(config, hwnd, session, output, active, granted, activation, ready, join, skip_menu, once):
    _record(config, session, join=join, standby=granted is not None)
    _emit(output, kind='event', event='worker_ready', fields={})
    _emit(output, kind='event', event='worker_exception', fields={'error': 'RuntimeError: boom'})


def crash_while_matching(config, hwnd, session, output, active, granted, activation, ready, join, skip_menu, once):
    """Fails after matchmaking began but before the table: the next one must join."""
    _record(config, session, join=join, standby=granted is not None)
    if session == 'agent':
        _emit(output, kind='event', event='worker_ready', fields={})
        _emit(output, kind='event', event='table_wait_started', fields={})
        _emit(output, kind='event', event='worker_exception', fields={'error': 'LiveError: matching'})
        return
    _play(output, .2)


def endless_agent(config, hwnd, session, output, active, granted, activation, ready, join, skip_menu, once):
    _record(config, session, join=join, standby=granted is not None)
    _play(output, 60, complete=False)


def lobby_after_outage(config, hwnd, session, output, active, granted, activation, ready, join, skip_menu, once):
    """The match ended while no Agent could act; the next one finds the lobby."""
    _record(config, session, join=join, standby=granted is not None)
    if session == 'agent':
        _play(output, .3, complete=False)
        _emit(output, kind='event', event='worker_exception', fields={'error': 'RuntimeError: input_desktop_blocked'})
        return
    if join:
        _emit(output, kind='event', event='worker_ready', fields={})
        _emit(output, kind='event', event='match_complete', fields={'round_index': 0})
        _emit(output, kind='event', event='fatal_error', fields={'error': 'Returned to lobby before Agent reached a table'})
        time.sleep(.3)
        return
    _play(output, .2)
