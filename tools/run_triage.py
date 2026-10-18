"""Triage one or more protected live sessions into comparable metrics.

    python -m tools.run_triage artifacts/live/protected-20261018-151224
    python -m tools.run_triage --all --ledger artifacts/live/workflow-audit/stability/ledger.csv

For each run it writes ``triage.json`` and ``triage.md`` (into the run, or into
``--output``): the first takeover with its trigger frame and suspected class,
Agent decision timing, model health, observer gaps, handback progress, and
supervisor restart storms. Read-only with respect to the run's own logs.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]

# (a) the monitor misread a normal state, (b) the Agent was not self-sufficient,
# (c) external. Class (a) is only suspected until the trigger frame is checked.
TRIP_CLASSES = {
    'operation_clock_unrecognized': 'a', 'critical_local_ui_uncertain': 'a',
    'observer_stalled': 'a', 'stale_or_invalid_observer_frame': 'a',
    'unverified_operation_ended': 'a',
    'operation_deadline': 'b', 'agent_stalled': 'b', 'agent_failure': 'b',
    'repeated_soft_failures': 'b', 'agent_recognition_timeout': 'b',
    'pre_match_timeout': 'b', 'agent_process_exited': 'b',
    'modal': 'c', 'observation_error': 'c',
    'operator_stop': 'c', 'operator_interrupt': 'c', 'operator_stop_before_start': 'c',
}
RESTART_STORM_PER_MINUTE = 10
LEDGER_FIELDS = ('run', 'agent_acceptance', 'result_file_written', 'first_trip_kind', 'first_trip_class',
                 'first_trip_at_s', 'decisions', 'clicked', 'verified', 'latency_p50_s', 'latency_p95_s',
                 'jev_valid_rate', 'base_expired_turns', 'observer_max_gap_s', 'startup_recognition_s',
                 'takeovers', 'handbacks_complete', 'handback_rejected', 'protector_restarts',
                 'restart_storm', 'agent_restarts', 'local_rung_decisions', 'emergency_rung_clicks')


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered)-1, round(q*(len(ordered)-1)))]


def agent_metrics(run):
    decisions = clicked = verified = 0
    latencies, base_expired = [], 0
    jev = [0, 0]
    local_rung = 0
    for directory in sorted(run.glob('agent*')):
        if not directory.is_dir():
            continue
        events = read_jsonl(directory/'events.jsonl')
        decisions += sum(e.get('event') == 'decision_submitted' for e in events)
        verified += sum(e.get('event') == 'verified' for e in events)
        clicks = [e for e in events if e.get('event') == 'clicked']
        clicked += len(clicks)
        local_rung += sum(e.get('event') == 'model_timing' and e.get('rung') == 'local' for e in events)
        for e in events:
            timing = e if e.get('event') == 'model_timing' else {}
            for attempt in timing.get('attempts') or (timing.get('model_timing') or {}).get('attempts') or ():
                if attempt.get('backend') == 'jev' and attempt.get('status') != 'cancelled':
                    jev[1] += 1
                    jev[0] += attempt.get('status') == 'ok'
        # A finished decision record carries the snapshot time and clock; the
        # matching click carries the input time.
        records = [r for r in read_jsonl(directory/'decisions.jsonl') if r.get('mode') not in ('cancelled',)]
        for record, click in zip(records, clicks):
            started = (click.get('input_timing') or {}).get('started_at')
            if started is None or record.get('timestamp') is None:
                continue
            latency = started-record['timestamp']
            latencies.append(latency)
            base = (record.get('countdown') or {}).get('base_seconds')
            if type(base) in (int, float) and latency > base:
                base_expired += 1
    return {'decisions': decisions, 'clicked': clicked, 'verified': verified,
            'latency_p50_s': round(median(latencies), 3) if latencies else None,
            'latency_p95_s': round(percentile(latencies, .95), 3) if latencies else None,
            'jev_valid_rate': round(jev[0]/jev[1], 3) if jev[1] else None,
            'jev_attempts': jev[1], 'base_expired_turns': base_expired, 'local_rung_decisions': local_rung}


def observer_metrics(run, start=None, end=None):
    samples = [s for s in read_jsonl(run/'observer'/'samples.jsonl') if 'captured_at' in s]
    table = [s for s in samples if s.get('table') or s.get('actionable') or s.get('waiting')]
    if start is not None:
        table = [s for s in table if s['captured_at'] >= start]
    if end is not None:
        table = [s for s in table if s['captured_at'] <= end]
    gaps = [(b['captured_at']-a['captured_at'], a['captured_at']) for a, b in zip(table, table[1:])]
    worst = max(gaps, default=(None, None))
    unknown_in_operation = sum(1 for s in table if s.get('actionable') and s.get('total_seconds') is None)
    return {'observer_samples': len(samples),
            'observer_max_gap_s': round(worst[0], 3) if worst[0] is not None else None,
            'observer_max_gap_at': worst[1],
            'unknown_clock_samples_in_operation': unknown_in_operation}


def trigger_frame(run, at):
    samples = [s for s in read_jsonl(run/'observer'/'samples.jsonl') if 'captured_at' in s]
    before = [s for s in samples if s['captured_at'] <= at+1e-6]
    if not before:
        return None
    sample = before[-1]
    evidence = sample.get('evidence')
    return {'captured_at': sample['captured_at'], 'evidence': evidence,
            'clock_status': sample.get('clock_status'), 'clock_reason': sample.get('clock_reason'),
            'actionable': sample.get('actionable'), 'hand_count': sample.get('hand_count')}


def watchdog_triage(run):
    """A watchdog run: the Agent owned all input; count restarts and rungs."""
    log = read_jsonl(run/'watchdog.jsonl')
    result_path = run/'watchdog-result.json'
    result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
    start = log[0]['monotonic'] if log else None
    restarts = [e for e in log if e.get('event') == 'agent_fenced']
    first = restarts[0] if restarts else None
    first_trip = None
    if first:
        reason = first.get('reason') or {}
        kind = reason.get('kind')
        first_trip = {'kind': kind, 'class': 'b', 'at': first['monotonic'],
                      'after_start_s': round(first['monotonic']-start, 3) if start is not None else None,
                      'reason': reason, 'trigger': None}
    agent_events = [e for e in log if e.get('event') == 'agent_event']
    report = {
        'run': run.name, 'workflow': 'watchdog', 'result_file_written': result is not None,
        'agent_acceptance': (result or {}).get('agent_acceptance'),
        'returned_lobby': (result or {}).get('returned_lobby'),
        'first_trip': first_trip, 'startup_recognition_s': None,
        'takeovers': 0, 'agent_restarts': len(restarts),
        'standby_promotions': sum(e.get('event') == 'standby_promoted' for e in log),
        'emergency_rung_clicks': sum(e.get('source_event') == 'emergency_clicked' for e in agent_events),
        'modal_clicks': sum(e.get('source_event') == 'modal_confirm_clicked' for e in agent_events),
        'handbacks_requested': 0, 'handbacks_complete': 0, 'handback_rejected': [],
        'time_to_retake_s': [], 'handback_requested_unresolved': False,
        'recovery_latched': [], 'protector_restarts': 0, 'restart_storm': False,
        'end_reason': (result or {}).get('reason'),
        **agent_metrics(run), **observer_metrics(run),
    }
    wait = [e for e in agent_events if e.get('source_event') == 'table_wait_started']
    started = [e for e in agent_events if e.get('source_event') == 'agent_game_started']
    if wait and started and started[0]['monotonic'] >= wait[0]['monotonic']:
        report['startup_recognition_s'] = round(started[0]['monotonic']-wait[0]['monotonic'], 3)
    flags = []
    if result is None:
        flags.append('watchdog-result.json missing (watchdog did not finish cleanly)')
    if report['jev_valid_rate'] == 0:
        flags.append('Jev never returned a valid response')
    report['flags'] = flags
    return report


def triage(run):
    run = Path(run)
    if (run/'watchdog.jsonl').exists():
        return watchdog_triage(run)
    protection = read_jsonl(run/'protection.jsonl')
    start = protection[0]['monotonic'] if protection else None
    def worker(name):
        return [e for e in protection if e.get('event') in ('worker_event', 'rejoin_event')
                and e.get('source_event') == name]
    trips = [e for e in protection if e.get('event') == 'protection_requested']
    first = trips[0] if trips else None
    first_trip = None
    if first:
        reason = first.get('reason') or {}
        kind = reason.get('kind')
        first_trip = {'kind': kind, 'class': TRIP_CLASSES.get(kind, '?'), 'at': first['monotonic'],
                      'after_start_s': round(first['monotonic']-start, 3) if start is not None else None,
                      'reason': reason, 'trigger': trigger_frame(run, first['monotonic'])}
    wait = worker('table_wait_started')
    started = [e for e in worker('agent_game_started') if e.get('event') == 'worker_event']
    startup = (round(started[0]['monotonic']-wait[0]['monotonic'], 3)
               if wait and started and started[0]['monotonic'] >= wait[0]['monotonic'] else None)
    restarts = [e['monotonic'] for e in protection if e.get('event') == 'protector_restarting']
    storm = any(sum(1 for t in restarts if r <= t < r+60) > RESTART_STORM_PER_MINUTE for r in restarts)
    result_path = run/'protection-result.json'
    result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
    requested = [e for e in protection if e.get('event') == 'handback_requested']
    complete = [e for e in protection if e.get('event') == 'handback_complete']
    rejected = [e.get('details', {}).get('error') for e in worker('handback_rejected')]
    takeovers = len(trips)
    report = {
        'run': run.name, 'result_file_written': result is not None,
        'agent_acceptance': (result or {}).get('agent_acceptance'),
        'returned_lobby': (result or {}).get('returned_lobby'),
        'first_trip': first_trip, 'startup_recognition_s': startup,
        'takeovers': takeovers, 'handbacks_requested': len(requested),
        'handbacks_complete': len(complete), 'handback_rejected': rejected,
        'time_to_retake_s': [e.get('recovery_seconds') for e in complete],
        'handback_requested_unresolved': len(requested) > len(complete)+len(rejected)+sum(
            e.get('event') == 'handback_timeout' for e in protection),
        'recovery_latched': [e.get('reason') for e in protection if e.get('event') == 'recovery_latched'],
        'protector_restarts': len(restarts), 'restart_storm': storm,
        **agent_metrics(run),
        **observer_metrics(run, end=first['monotonic'] if first else None),
    }
    flags = []
    if not report['result_file_written']:
        flags.append('protection-result.json missing (supervisor did not finish cleanly)')
    if storm:
        flags.append(f'restart storm: more than {RESTART_STORM_PER_MINUTE} protector restarts in a minute')
    if report['handback_requested_unresolved']:
        flags.append('handback requested but never completed, rejected or timed out')
    if report['jev_valid_rate'] == 0:
        flags.append('Jev never returned a valid response')
    if (report['observer_max_gap_s'] or 0) > 1:
        flags.append(f"observer gap {report['observer_max_gap_s']} s before the first trip")
    report['flags'] = flags
    return report


def markdown(report):
    trip = report['first_trip']
    lines = [f"# Triage: {report['run']}", '',
             f"- Agent acceptance: {report['agent_acceptance']} (result file: {report['result_file_written']})"]
    if report.get('workflow') == 'watchdog':
        lines.append(f"- Agent restarts {report['agent_restarts']} (standby promotions "
                     f"{report['standby_promotions']}); first: {(trip or {}).get('kind')}; "
                     f"end: {report['end_reason']}")
        lines.append(f"- Fallback rungs: local efficiency {report['local_rung_decisions']}, "
                     f"tsumogiri/PASS {report['emergency_rung_clicks']}; popups dismissed {report['modal_clicks']}")
    elif trip:
        trigger = trip.get('trigger') or {}
        lines.append(f"- First takeover: `{trip['kind']}` (class {trip['class']}) "
                     f"{trip['after_start_s']} s after start; trigger frame `{trigger.get('evidence')}` "
                     f"(clock {trigger.get('clock_status')}/{trigger.get('clock_reason')})")
    else:
        lines.append('- No takeover')
    lines += [f"- Decisions {report['decisions']}, clicked {report['clicked']}, verified {report['verified']}; "
              f"latency p50/p95 {report['latency_p50_s']}/{report['latency_p95_s']} s; "
              f"base expired on {report['base_expired_turns']} turns",
              f"- Jev valid rate {report['jev_valid_rate']} over {report['jev_attempts']} attempts",
              f"- Startup recognition {report['startup_recognition_s']} s; observer max gap "
              f"{report['observer_max_gap_s']} s at {report['observer_max_gap_at']}",
              ]
    if report.get('workflow') != 'watchdog':
        lines += [f"- Takeovers {report['takeovers']}, handbacks requested {report['handbacks_requested']}, "
                  f"complete {report['handbacks_complete']}, rejected {report['handback_rejected']}",
                  f"- Protector restarts {report['protector_restarts']} (storm: {report['restart_storm']})"]
    if report['flags']:
        lines += ['', '## Flags', *[f'- {flag}' for flag in report['flags']]]
    return '\n'.join(lines)+'\n'


def ledger_row(report):
    trip = report['first_trip'] or {}
    row = {key: report.get(key) for key in LEDGER_FIELDS}
    row.update(first_trip_kind=trip.get('kind'), first_trip_class=trip.get('class'),
               first_trip_at_s=trip.get('after_start_s'), handback_rejected=len(report['handback_rejected']))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='*', type=Path)
    parser.add_argument('--all', action='store_true', help='every artifacts/live/protected-* and live-* run')
    parser.add_argument('--output', type=Path, help='write reports here instead of into each run')
    parser.add_argument('--ledger', type=Path, help='CSV ledger to rewrite with one row per run')
    args = parser.parse_args(argv)
    runs = list(args.runs)
    if args.all:
        runs += sorted(p for pattern in ('protected-*', 'live-*')
                       for p in (ROOT/'artifacts/live').glob(pattern) if p.is_dir())
    if not runs:
        parser.error('give a run directory or --all')
    rows = []
    for run in runs:
        report = triage(run)
        target = args.output/run.name if args.output else run
        target.mkdir(parents=True, exist_ok=True)
        (target/'triage.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                                          encoding='utf-8')
        (target/'triage.md').write_text(markdown(report), encoding='utf-8')
        rows.append(ledger_row(report))
        print(markdown(report))
    if args.ledger:
        args.ledger.parent.mkdir(parents=True, exist_ok=True)
        with args.ledger.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
