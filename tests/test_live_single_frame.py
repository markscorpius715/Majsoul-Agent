"""A frozen valid decision is independent of subsequent public animations."""
from dataclasses import fields, replace
import json
from pathlib import Path

import pytest

from majsoul.domain import Action, Box
from majsoul.live.verification import assess_discard_hand_change
from majsoul.state import FrameObservation, ObservedMeld, ObservedTile, PlayerObservation
from test_live_decisions import SlowPolicy, close, make_loop, process
from test_live_loop import HAND14, visible_reading
from test_live_verification import advance, discard_case, snapshot, FILLER

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/live/acceptance-20261010/agent-recheck-223629/observations.jsonl'


def decode_observation(raw):
    def tile(value):
        return ObservedTile(**value)
    players = tuple(PlayerObservation(
        **{key: raw_player[key] for key in ('slot', 'seat', 'dealer', 'score', 'score_available', 'kita_count')},
        riichi=raw_player.get('riichi'), riichi_evidence=raw_player.get('riichi_evidence', {}),
        river=tuple(tile(t) for t in raw_player['river']),
        melds=tuple(ObservedMeld(tuple(tile(t) for t in m['tiles']), m['kind'], m['opened'])
                    for m in raw_player['melds'])) for raw_player in raw['players'])
    values = {f.name: raw[f.name] for f in fields(FrameObservation) if f.name in raw}
    values.update(players=players, hand=tuple(tile(t) for t in raw['hand']),
                  dora_indicators=tuple(tile(t) for t in raw['dora_indicators']))
    for name in ('active_slots', 'action_buttons', 'issues', 'unavailable_regions'):
        if values.get(name) is not None:
            values[name] = tuple(values[name])
    return FrameObservation(**values)


def saved_observations(source=SOURCE):
    observations = [decode_observation(json.loads(line)['observation'])
                    for line in source.read_text().splitlines()]
    return {observation.frame_index: observation for observation in observations}


def test_real_failed_response_uses_first_good_frame_and_survives_next_bad_frame(tmp_path):
    samples = saved_observations()
    policy = SlowPolicy()
    loop, window, clock = make_loop(tmp_path, policy)
    loop.session_ui = None
    loop._active_slots = samples[157].active_slots
    source = [157]
    def read(image, roi, slots, *, timestamp, frame_index):
        obs = replace(samples[source[0]], timestamp=timestamp, frame_index=frame_index)
        return obs, visible_reading(tuple(t.tile for t in obs.hand))
    loop.reader.read_with_visible = read
    loop.button_reader._buttons = {'pon': Box(200, 110, 250, 130), 'pass': Box(260, 110, 320, 130)}
    loop.button_reader.last_buttons = dict(loop.button_reader._buttons)
    try:
        process(loop)
        assert policy.entered.wait(1)
        pending = loop._pending_decision
        assert loop.tracker.state is None  # No two-frame table commitment needed.
        assert pending['snapshot'].decision_ready
        assert {a.kind for a in pending['actions']} == {'pon', 'pass'}
        original_input = policy.calls[0][0].for_model()
        source[0] = 158  # Same buttons/hand, river_top now fails segmentation.
        process(loop)
        assert loop._pending_decision is pending and not pending['future'].cancelled()
        assert policy.calls[0][0].for_model() == original_input
        policy.release.set()
        pending['future'].result(timeout=1)
        process(loop)
        assert len(policy.calls) == 1 and loop._decisions_made == 1
        assert window.clicks == []
        records = [json.loads(line) for line in (loop.artifacts/'events.jsonl').read_text().splitlines()]
        gates = [r for r in records if r['event'] == 'decision_gate']
        assert gates[0]['stable'] is False and gates[0]['ready_for_model'] is True
        assert gates[0]['buttons']['ready'] is True
        assert gates[1]['validation_issues'] and gates[1]['blocking_reasons'] == []
    finally:
        close(loop)


@pytest.mark.parametrize('change', ['unknown_hand', 'changed_hand', 'closed_buttons', 'round'])
def test_frozen_decision_still_cancels_when_local_opportunity_changes(tmp_path, change):
    policy = SlowPolicy()
    loop, _, _ = make_loop(tmp_path, policy)
    try:
        process(loop)
        assert policy.entered.wait(1)
        pending = loop._pending_decision
        read = loop.reader.read_with_visible
        def changed(*args, **kwargs):
            obs, visible = read(*args, **kwargs)
            if change == 'unknown_hand':
                obs = replace(obs, issues=('hand: unknown',))
            elif change == 'changed_hand':
                obs = replace(obs, hand=obs.hand[:-1], drawn_index=None)
            elif change == 'closed_buttons':
                obs = replace(obs, action_buttons=('pass',))
            else:
                obs = replace(obs, hand_number=2)
            return obs, visible
        loop.reader.read_with_visible = changed
        process(loop)
        assert pending['future'].cancelled() and policy.cancelled.wait(1)
        assert loop._pending_decision is None
    finally:
        close(loop)


def test_initial_public_bad_frame_submits_with_explicit_blind_spot(tmp_path):
    policy = SlowPolicy()
    loop, _, _ = make_loop(tmp_path, policy)
    read = loop.reader.read_with_visible
    def bad(*args, **kwargs):
        obs, visible = read(*args, **kwargs)
        return replace(obs, issues=('river_top: Oversized white component may be animation or occlusion',)), visible
    loop.reader.read_with_visible = bad
    try:
        process(loop)
        assert policy.entered.wait(1) and loop._pending_decision is not None
        record = next(json.loads(line) for line in (loop.artifacts/'events.jsonl').read_text().splitlines()
                      if json.loads(line)['event'] == 'decision_gate')
        assert record['blocking_reasons'] == []
        assert not record['decision_validation_issues']
        assert record['public_blind_spots'] == ['river_top']
        assert 'rivers_1' in loop._pending_decision['domain'].observation.unavailable_fields
        assert record['table_stability']['consecutive_frames'] == 0
    finally:
        close(loop)


def test_first_clean_frame_resumes_paused_agent_without_stability_wait(tmp_path):
    policy = SlowPolicy()
    loop, _, _ = make_loop(tmp_path, policy)
    loop._paused = True
    loop._issue_since = 0
    try:
        process(loop)
        assert not loop._paused
        assert policy.entered.wait(1)
        assert loop._pending_decision['snapshot'].decision_ready
        records = [json.loads(line) for line in (loop.artifacts/'events.jsonl').read_text().splitlines()]
        assert next(r for r in records if r['event'] == 'resumed')['clean_frames'] == 1
    finally:
        close(loop)


@pytest.mark.parametrize('interference', ['river_missing', 'called', 'wall_advanced', 'public_unknown', 'sorted'])
def test_exact_hand_difference_ignores_public_interference(interference):
    action, before, after = discard_case()
    after = replace(after, players=before.players)  # No river append whatsoever.
    if interference == 'called':
        other = replace(after.players[1], melds=(ObservedMeld(
            tuple(ObservedTile('9s') for _ in range(3)), 'pon'),))
        after = replace(after, players=(after.players[0], other, after.players[2]))
    elif interference == 'wall_advanced':
        after = replace(after, remaining_tiles=before.remaining_tiles-3)
    elif interference == 'public_unknown':
        after = replace(after, remaining_tiles=None, issues=('river_left: animation',))
    elif interference == 'sorted':
        after = replace(after, hand=tuple(reversed(after.hand)))
    result = assess_discard_hand_change(action, before, after)
    assert result.status == 'verified'
    assert result.evidence['river_evidence_required'] is False


@pytest.mark.parametrize('change,expected', [
    ('unchanged', 'unknown'), ('wrong_tile', 'failed'), ('two_removed', 'unknown'),
    ('unknown_tile', 'unknown'), ('hand_issue', 'unknown'), ('different_round', 'unknown'),
    ('red_identity', 'failed')])
def test_hand_difference_rejects_wrong_or_unrecognized_execution(change, expected):
    action, before, after = discard_case()
    if change == 'unchanged': after = advance(before)
    elif change == 'wrong_tile': after = advance(before, hand=[t.tile for t in before.hand][1:])
    elif change == 'two_removed': after = replace(after, hand=after.hand[:-1])
    elif change == 'unknown_tile': after = replace(after, hand=(ObservedTile(None), *after.hand[1:]))
    elif change == 'hand_issue': after = replace(after, issues=('hand: animation',))
    elif change == 'different_round': after = replace(after, hand_number=2)
    else:
        before = snapshot(FILLER + ('0p', '5p', '3s', '9s'))
        action = Action('discard:10:0p', 'discard', '0p', 10)
        after = advance(before, hand=FILLER + ('0p', '3s', '9s'))
    assert assess_discard_hand_change(action, before, after).status == expected
