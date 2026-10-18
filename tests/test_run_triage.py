"""Triage of real protected runs reproduces the hand-made 2026-10-18 analysis."""
from pathlib import Path

from tools.run_triage import ledger_row, markdown, triage

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT/'artifacts/live'


def test_run_151224_first_trip_trigger_and_flags():
    report = triage(LIVE/'protected-20261018-151224')
    trip = report['first_trip']
    assert trip['kind'] == 'operation_clock_unrecognized' and trip['class'] == 'a'
    assert trip['trigger']['evidence'].endswith('000016.jpg')
    assert report['decisions'] == 2 and report['clicked'] == 1 and report['verified'] == 1
    assert report['jev_valid_rate'] == 0
    assert 24 < report['startup_recognition_s'] < 25
    assert report['observer_max_gap_s'] > 4
    assert report['handbacks_requested'] == 1 and report['handbacks_complete'] == 0
    assert report['handback_rejected'] == ['toggle drawer is not visibly open']
    assert report['restart_storm'] and not report['result_file_written']
    assert 'operation_clock_unrecognized' in markdown(report)
    assert ledger_row(report)['first_trip_class'] == 'a'


def test_run_151154_startup_observer_trip_is_suspected_false_positive():
    report = triage(LIVE/'protected-20261018-151154')
    assert report['first_trip']['kind'] == 'observer_stalled'
    assert report['first_trip']['class'] == 'a'
    assert report['first_trip']['after_start_s'] < 3
    assert report['decisions'] == 0


def test_watchdog_run_is_triaged_from_its_own_logs(tmp_path):
    import json
    run = tmp_path/'live-x'
    (run/'agent').mkdir(parents=True)
    rows = [{'event': 'agent_spawned', 'monotonic': 10.0},
            {'event': 'agent_event', 'monotonic': 12.0, 'source_event': 'emergency_clicked'},
            {'event': 'agent_fenced', 'monotonic': 20.0, 'reason': {'kind': 'agent_stalled'}},
            {'event': 'standby_promoted', 'monotonic': 20.1}]
    (run/'watchdog.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    (run/'watchdog-result.json').write_text(json.dumps({'agent_acceptance': False, 'returned_lobby': True,
                                                       'reason': None}), encoding='utf-8')
    (run/'agent'/'events.jsonl').write_text(json.dumps({'event': 'model_timing', 'rung': 'local'}), encoding='utf-8')
    report = triage(run)
    assert report['workflow'] == 'watchdog' and report['agent_restarts'] == 1
    assert report['standby_promotions'] == 1 and report['emergency_rung_clicks'] == 1
    assert report['local_rung_decisions'] == 1 and report['first_trip']['kind'] == 'agent_stalled'
    assert 'Agent restarts 1' in markdown(report)
    assert ledger_row(report)['agent_restarts'] == 1
