"""Local execution proof tolerates unrelated occlusion, animation and timing."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from majsoul.domain import Action
from majsoul.live.execution_verification import assess_execution_action, apply_execution_context
from majsoul.state import ObservedMeld, ObservedTile
from test_live_verification import (FILLER, snapshot, advance, tiles, meld, pon_case,
                                    discard_case, ui_hidden)
from test_live_single_frame import decode_observation
from test_live_decisions import make_loop, close, process, SlowPolicy
from test_live_loop import FakePolicy, visible_reading
from tools.live_test_supervisor import AcceptanceGuard

ROOT = Path(__file__).resolve().parents[1]


def real_pon_case():
    path = ROOT/'artifacts/live/acceptance-20261010/agent-recheck-230811/frames/00015-verification.json'
    record = json.loads(path.read_text())
    return decode_observation(record['before']), decode_observation(record['after'])


def check(action, before, after):
    return assess_execution_action(action, before, after, 'bottom')


def test_latest_real_pon_failure_replays_as_verified_on_local_evidence():
    before, after = real_pon_case()
    result = check(Action('button:pon', 'pon', button='pon'), before, after)
    assert result.status == 'verified'
    assert result.evidence['consumed_tiles'] == ['1s', '1s']
    assert result.evidence['source_river_evidence_required'] is False


@pytest.mark.parametrize('interference', ['own_ui', 'opponent_animation', 'missing_wall', 'source_unchanged', 'metadata'])
def test_pon_uses_only_new_meld_and_exact_physical_hand_removal(interference):
    action, before, after = pon_case()
    if interference == 'own_ui':
        before = ui_hidden(before, ('river_bottom', 'kita_bottom', 'melds_bottom'))
    elif interference == 'opponent_animation':
        after = replace(after, issues=('river_left: animation', 'melds_right: animation'))
    elif interference == 'missing_wall':
        before, after = replace(before, remaining_tiles=None), replace(after, remaining_tiles=None)
    elif interference == 'source_unchanged':
        after = replace(after, players=(after.players[0], before.players[1], before.players[2]))
    else:
        after = replace(after, round_wind=None, remaining_tiles=47)
    result = check(action, before, after)
    assert result.status == 'verified'
    assert result.evidence['consumed_tiles'] == ['0p', '5p']


@pytest.mark.parametrize('change', ['unchanged_hand', 'wrong_removed', 'extra_change', 'wrong_meld',
                                   'unknown_hand', 'missing_meld', 'hidden_positive_baseline', 'round', 'unordered'])
def test_pon_still_rejects_missing_or_contradictory_local_evidence(change):
    action, before, after = pon_case()
    if change == 'unchanged_hand': after = replace(after, hand=before.hand)
    elif change == 'wrong_removed': after = replace(after, hand=tiles(FILLER+('0p',)))
    elif change == 'extra_change': after = replace(after, hand=after.hand[:-1])
    elif change == 'wrong_meld': after = replace(after, players=(replace(after.players[0], melds=(meld(('6s',)*3),)), *after.players[1:]))
    elif change == 'unknown_hand': after = replace(after, hand=(ObservedTile(None), *after.hand[1:]))
    elif change == 'missing_meld': after = ui_hidden(after, ('melds_bottom',))
    elif change == 'hidden_positive_baseline': before = ui_hidden(replace(before, hand=before.hand[3:]), ('melds_bottom',))
    elif change == 'round': after = replace(after, hand_number=2)
    else: after = replace(after, timestamp=before.timestamp)
    assert check(action, before, after).status == 'unknown'


def kan_case(kind, replacement='6s'):
    action = Action('button:kan', 'kan', button='kan')
    group = meld(('5p', '0p', '5p', '5p'), kind)
    if kind == 'ankan': before = snapshot(FILLER+('5p', '0p', '5p', '5p'))
    elif kind == 'daiminkan': before = snapshot(FILLER+('5p', '0p', '5p'), source_river=('5p',))
    else: before = snapshot(FILLER+('5p',), own_melds=(meld(('5p', '0p', '5p')),))
    after = advance(before, hand=FILLER+(replacement,), own_melds=(group,),
                    source_river=(), wall_draw=1, drawn_index=10)
    return action, before, after


@pytest.mark.parametrize('kind', ['ankan', 'daiminkan', 'kakan'])
@pytest.mark.parametrize('interference', ['own_ui', 'public_animation', 'missing_wall', 'wall_lag'])
def test_all_kan_types_use_local_structure_and_supplement(kind, interference):
    action, before, after = kan_case(kind)
    if interference == 'own_ui': before = ui_hidden(before, ('river_bottom', 'kita_bottom'))
    elif interference == 'public_animation': after = replace(after, issues=('river_right: animation',))
    elif interference == 'missing_wall': after = replace(after, remaining_tiles=None)
    else: after = replace(after, remaining_tiles=before.remaining_tiles)
    result = check(action, before, after)
    assert result.status == 'verified'
    assert result.evidence['meld_kind'] == kind and result.evidence['replacement_tile'] == '6s'


def test_hidden_ankan_preserves_red_identity_using_only_own_hand():
    action, before, after = kan_case('ankan')
    hidden = ObservedMeld((ObservedTile(None, face_down=True), ObservedTile('5p'),
                           ObservedTile('5p'), ObservedTile(None, face_down=True)), 'ankan')
    after = replace(after, players=(replace(after.players[0], melds=(hidden,)), *after.players[1:]))
    result = check(action, before, after)
    assert result.status == 'verified' and result.evidence['consumed_tiles'] == ['0p', '5p', '5p', '5p']
    duplicated_red = replace(after, hand=tiles(FILLER+('0p',)))
    assert check(action, before, duplicated_red).reason == 'own_physical_tile_count_conflict'
    ambiguous = replace(hidden, kind='unknown')
    assert check(action, before, replace(after, players=(replace(after.players[0], melds=(ambiguous,)), *after.players[1:]))).status == 'unknown'


@pytest.mark.parametrize('kind', ['ankan', 'daiminkan', 'kakan'])
@pytest.mark.parametrize('change', ['no_supplement', 'unrelated_hand_change', 'wrong_target', 'unknown_meld'])
def test_kan_does_not_accept_missing_supplement_or_unrelated_changes(kind, change):
    action, before, after = kan_case(kind)
    if change == 'no_supplement': after = replace(after, hand=after.hand[:-1], drawn_index=None)
    elif change == 'unrelated_hand_change': after = replace(after, hand=tiles(('9s', *FILLER[1:], '6s')))
    elif change == 'wrong_target': action = replace(action, tile='9s')
    else: after = ui_hidden(after, ('melds_bottom',))
    assert check(action, before, after).status == 'unknown'


@pytest.mark.parametrize('replacement', ['6s', '4z'])
def test_nuki_with_visible_increment_needs_no_public_table_or_wall(replacement):
    before = snapshot(FILLER+('1s', '2s', '3s', '4z'), kita=1, buttons=('nuki',))
    after = replace(advance(before, hand=FILLER+('1s', '2s', '3s', replacement), kita=2,
                            issues=('river_left: animation',)), remaining_tiles=None)
    result = check(Action('button:nuki', 'nuki', button='nuki'), before, after)
    assert result.status == 'verified' and result.evidence['replacement_tile'] == replacement


def test_nuki_hidden_counter_can_use_north_removal_and_different_supplement():
    before = ui_hidden(snapshot(FILLER+('1s', '2s', '3s', '4z'), buttons=('nuki',)), ('kita_bottom',))
    after = advance(before, hand=FILLER+('1s', '2s', '3s', '6s'))
    action = Action('button:nuki', 'nuki', button='nuki')
    assert check(action, before, after).status == 'verified'
    assert check(action, before, advance(before)).status == 'unknown'
    assert check(action, before, advance(before, hand=FILLER+('1s', '2s', '3s'))).status == 'unknown'


def test_nuki_cannot_claim_a_fifth_north_as_supplement():
    before = snapshot(FILLER+('4z',)*4, buttons=('nuki',))
    after = advance(before, kita=1)
    assert check(Action('button:nuki', 'nuki', button='nuki'), before, after).reason == 'own_physical_tile_count_conflict'


def test_riichi_hand_removal_closes_selection_without_river_declaration_proof():
    _, before, after = discard_case()
    action = Action('riichi_discard:13:9s', 'riichi', '9s', 13)
    after = ui_hidden(after, ('river_bottom',))
    result = check(action, before, after)
    assert result.status == 'verified' and result.evidence['riichi_declared'] is None
    assert check(action, before, advance(before)).status == 'unknown'
    assert check(replace(action, hand_index=0), before, after).status == 'unknown'


def test_pass_closed_window_continues_without_claiming_click_acceptance():
    action = Action('button:pass', 'pass', button='pass')
    before = ui_hidden(snapshot(FILLER+('1s', '2s', '3s'), buttons=('pass', 'pon')))
    after = replace(advance(before, action_buttons=(), issues=('river_left: animation',)), remaining_tiles=None)
    result = check(action, before, after)
    assert result.status == 'resolved' and result.evidence['click_acceptance'] == 'unattributed'
    guard = AcceptanceGuard()
    guard.observe('decision_gate', {'countdown': {'status': 'complete'}}, 10)
    guard.observe('window_resolved', {'reason': result.reason}, 11)
    guard.check(30)
    assert guard.failure is None
    assert check(action, before, advance(before)).status == 'unknown'
    assert check(action, before, advance(before, action_buttons=None)).status == 'unknown'


def test_runtime_pon_verifier_accepts_first_local_proof_despite_global_instability(tmp_path):
    before, after = real_pon_case()
    loop, window, _ = make_loop(tmp_path, FakePolicy())
    loop.session_ui = None
    loop._track = lambda *args: (after, None, SimpleNamespace(stable=False, validation_issues=('river_right: animation',)))
    try:
        assert loop._verify(Action('button:pon', 'pon', button='pon'), before, 15) == 'verified'
        record = json.loads((loop.artifacts/'frames/00015-verification.json').read_text())
        assert record['assessed_frame_index'] == 319
        assert window.clicks == []
    finally:
        close(loop)


def test_real_post_pon_snapshot_is_discard_ready_without_draw_marker(tmp_path):
    from majsoul.live.bridge import infer_phase
    _, after = real_pon_case()
    loop, _, _ = make_loop(tmp_path, FakePolicy())
    try:
        state = loop.tracker.decision_snapshot(after)
        assert after.drawn_index is None and after.action_buttons == ()
        assert state.decision_ready and state.self_meld_count == 1
        assert infer_phase(after, state, pending_riichi=False) == 'discard'
        assert loop.tracker.state is None  # No invented stable event or DRAW.
    finally:
        close(loop)


def test_real_post_pon_state_submits_discard_on_first_frame(tmp_path):
    _, after = real_pon_case()
    policy = SlowPolicy()
    loop, _, clock = make_loop(tmp_path, policy)
    loop.session_ui = None
    loop._active_slots = after.active_slots
    def read(image, roi, slots, *, timestamp, frame_index):
        return (replace(after, timestamp=timestamp, frame_index=frame_index),
                visible_reading(tuple(t.tile for t in after.hand)))
    loop.reader.read_with_visible = read
    # The live loop retains this proof immediately on successful verification.
    anchor = replace(after, timestamp=clock(), frame_index=0)
    result = check(Action('button:pon', 'pon', button='pon'), *real_pon_case())
    loop._verified_operation = {'kind': 'pon', 'after': anchor, 'evidence': result.evidence}
    try:
        process(loop)
        assert policy.entered.wait(1)
        pending = loop._pending_decision
        assert pending['phase'] == 'discard'
        assert pending['snapshot'].self_meld_count == 1 and loop.tracker.state is None
        assert any(a.kind == 'discard' for a in pending['actions'])
        assert pending['domain'].kuikae_tile == '1s'
        assert pending['domain'].history_complete is False
    finally:
        close(loop)


def test_verified_pon_context_preserves_kuikae_and_does_not_invent_history(tmp_path):
    from majsoul.live.bridge import build_domain_state
    action, before, after = pon_case()
    # One fourth same-family tile remains concealed and must be forbidden.
    before = replace(before, hand=(ObservedTile('5p'), *before.hand[1:]))
    after = replace(after, hand=(ObservedTile('5p'), *after.hand[1:]))
    result = check(action, before, after)
    assert result.status == 'verified'
    loop, _, _ = make_loop(tmp_path, FakePolicy())
    try:
        state = loop.tracker.decision_snapshot(after)
        domain = build_domain_state(state, after, visible_reading(tuple(t.tile for t in after.hand)),
                                    {}, pending_riichi=False, window=(0, 0, 1000, 800))
        assert loop.engine.legal_actions(domain) == []  # Unknown call history remains guarded.
        proof = {'kind': 'pon', 'after': after, 'evidence': result.evidence}
        bound = apply_execution_context(domain, after, proof, 'bottom')
        actions = loop.engine.legal_actions(bound)
        assert actions and all(a.tile not in ('5p', '0p') for a in actions)
        assert bound.history_complete is False and bound.kuikae_tile == '5p'
        changed = replace(after, hand=after.hand[:-1])
        assert apply_execution_context(domain, changed, proof, 'bottom') is domain
        assert apply_execution_context(domain, replace(after, hand_number=2), proof, 'bottom') is domain
    finally:
        close(loop)


def test_pon_cannot_verify_an_impossible_fifth_own_tile():
    action, before, after = pon_case()
    before = replace(before, hand=(ObservedTile('5p'), ObservedTile('5p'), *before.hand[2:]))
    after = replace(after, hand=(ObservedTile('5p'), ObservedTile('5p'), *after.hand[2:]))
    assert check(action, before, after).reason == 'own_physical_tile_count_conflict'


def test_pass_unknown_button_candidates_cannot_be_mistaken_for_closed_window(tmp_path):
    loop, _, _ = make_loop(tmp_path, FakePolicy())
    loop.session_ui = None
    before = snapshot(FILLER+('1s', '2s', '3s'), buttons=('pass',))
    after = advance(before, action_buttons=())
    def track(*args):
        loop._frame_unknowns = [('unknown', None, None)]
        return after, None, SimpleNamespace(stable=False, validation_issues=())
    loop._track = track
    try:
        assert loop._verify(Action('button:pass', 'pass', button='pass'), before, 1) == 'verification_unknown'
        record = json.loads((loop.artifacts/'frames/00001-verification.json').read_text())
        assert record['reason'] == 'response_buttons_unrecognized'
    finally:
        close(loop)


@pytest.mark.parametrize('kind', ['ron', 'tsumo', 'abort', 'discard'])
def test_result_overlay_completes_window_without_false_failure_or_winner_claim(tmp_path, kind):
    loop, window, _ = make_loop(tmp_path, FakePolicy())
    loop.session_ui = SimpleNamespace(ui=SimpleNamespace(confirm=lambda _: 'blue'))
    _, before, _ = discard_case()
    try:
        assert loop._verify(Action(f'button:{kind}', kind, button=kind), before, 1) == 'window_resolved'
        record = json.loads((loop.artifacts/'frames/00001-verification.json').read_text())
        assert record['status'] == 'resolved' and record['evidence']['winner_attribution'] == 'unknown'
        assert 'verification_unknown' not in (loop.artifacts/'events.jsonl').read_text()
        assert window.clicks == []
    finally:
        close(loop)
