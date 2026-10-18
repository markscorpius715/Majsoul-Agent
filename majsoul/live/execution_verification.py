"""Action-local execution evidence, independent of unrelated table animation."""
from collections import Counter
from dataclasses import replace

from ..tiles import tile_index
from ..state.meld_transition import prove_pon_transition
from .verification import (VerificationResult, assess_discard_hand_change,
                           _labels, _melds, _new_meld, _meld_signature)


def _unknown(reason):
    return VerificationResult('unknown', reason)


def _verified(reason, **evidence):
    return VerificationResult('verified', reason, evidence)


def _same_window(before, after):
    return (after.timestamp > before.timestamp and after.frame_index > before.frame_index
            and before.active_slots == after.active_slots and all(
                getattr(before, name) is None or getattr(after, name) is None or
                getattr(before, name) == getattr(after, name)
                for name in ('round_wind', 'hand_number', 'honba')))


def _known_hand(observation):
    if any(issue.startswith('hand') for issue in observation.issues):
        return False
    try:
        return all(t.tile is not None and not t.face_down and tile_index(t.tile) >= 0
                   for t in observation.hand)
    except (ValueError, KeyError, IndexError):
        return False


def _own_melds(observation, self_slot):
    """A complete 13/14-tile concealed hand proves zero existing melds."""
    own = observation.player(self_slot)
    name = f'melds_{self_slot}'
    if observation.field_available(name) and not any(
            issue.startswith(name + ':') for issue in observation.issues):
        return own
    if (name in observation.unavailable_regions and not own.melds and
            len(observation.hand) in (13, 14)):
        return own
    return None


def _family(labels):
    try:
        indices = {tile_index(label) for label in labels if label is not None}
        return next(iter(indices)) if len(indices) == 1 else None
    except (ValueError, KeyError, IndexError):
        return None


def _physical_labels_possible(labels):
    try:
        return (max(Counter(tile_index(t) for t in labels).values(), default=0) <= 4 and
                all(n <= 1 for t, n in Counter(labels).items() if t.startswith('0')))
    except (ValueError, KeyError, IndexError):
        return False


def _local_tiles_possible(observation, own):
    labels = [t.tile for t in observation.hand] + [t.tile for m in own.melds for t in m.tiles
                                                if t.tile is not None]
    name = f'kita_{own.slot}'
    if (own.kita_count is not None and observation.field_available(name) and
            not any(issue.startswith(name + ':') for issue in observation.issues)):
        labels.extend(['4z'] * own.kita_count)
    return _physical_labels_possible(labels)


def _replacement(before, after, consumed):
    old, new = Counter(_labels(before.hand)), Counter(_labels(after.hand))
    if consumed - old:
        return None
    retained = old - consumed
    added = new - retained
    if retained - new or sum(added.values()) != 1:
        return None
    tile = next(iter(added))
    if after.drawn_index is not None and (
            not 0 <= after.drawn_index < len(after.hand) or
            after.hand[after.drawn_index].tile != tile):
        return None
    return tile


def apply_execution_context(domain, observation, proof, self_slot):
    """Carry only a proved local operation into its still-identical own hand.

    This supplies kuikae/replacement context without claiming full event history.
    Hand, meld, round or temporal changes make the evidence unusable.
    """
    if proof is None:
        return domain
    anchor = proof['after']
    if (observation.timestamp < anchor.timestamp or
            observation.active_slots != anchor.active_slots or
            Counter(_labels(observation.hand)) != Counter(_labels(anchor.hand)) or
            any(getattr(anchor, name) is not None and getattr(observation, name) is not None and
                getattr(anchor, name) != getattr(observation, name)
                for name in ('round_wind', 'hand_number', 'honba'))):
        return domain
    own, previous = _own_melds(observation, self_slot), _own_melds(anchor, self_slot)
    if own is None or previous is None or _melds(own) != _melds(previous):
        return domain
    kind, evidence = proof['kind'], proof['evidence']
    sources = dict(domain.context_sources)
    sources['local_operation_frame'] = anchor.frame_index
    if kind == 'pon':
        sources.update(kuikae_tile='verified_local_pon', replacement_draw='verified_local_pon')
        return replace(domain, observation=replace(domain.observation,
                       riichi=(False, *domain.observation.riichi[1:])),
                       kuikae_tile=evidence['called_tile'], replacement_draw=False,
                       context_sources=sources)
    if kind in ('kan', 'nuki'):
        sources['replacement_draw'] = 'verified_local_' + kind
        return replace(domain, replacement_draw=True, context_sources=sources)
    return domain


def assess_execution_action(action, before, after, self_slot):
    """Verify the requested local change; never require a full-table fingerprint.

    ``resolved`` closes an opportunity without claiming active PASS acceptance
    or attributing a winner. Structural actions still require exact hand proof.
    """
    if not _same_window(before, after):
        return _unknown('different_or_unordered_window')
    if action.kind == 'pass':
        if ('pass' in (before.action_buttons or ()) and after.action_buttons == () and
                not any(issue.startswith('action_buttons') for issue in after.issues)):
            return VerificationResult('resolved', 'response_window_closed', {
                'click_acceptance': 'unattributed', 'client_timeout_possible': True})
        if (len(before.hand) % 3 == 1 and before.drawn_index is None and
                len(after.hand) == len(before.hand)+1 and type(after.drawn_index) is int):
            # Our own draw replaced the call window; its own buttons (e.g. a
            # nuki/pass prompt) are a new turn, not the call still open.
            return VerificationResult('resolved', 'response_window_closed_own_draw', {
                'click_acceptance': 'unattributed', 'client_timeout_possible': True})
        return _unknown('response_still_open_or_unavailable')
    if action.kind in ('ron', 'tsumo', 'abort'):
        return _unknown('awaiting_result_overlay')
    if not _known_hand(before) or not _known_hand(after):
        return _unknown('hand_observation_uncertain')
    if action.kind == 'discard':
        return assess_discard_hand_change(action, before, after)
    if action.kind == 'riichi':
        if action.hand_index is None:
            return _unknown('riichi_selection_not_observed')
        result = assess_discard_hand_change(replace(action, kind='discard'), before, after)
        if result.status != 'verified':
            return result
        # The selected discard completing is sufficient to close this input
        # interaction. Declaration attribution remains explicit and separate.
        try:
            own = after.player(self_slot)
        except KeyError:
            return _unknown('self_slot_unavailable')
        name = f'river_{self_slot}'
        visible = all(obs.field_available(name) and not any(issue.startswith(name + ':')
                      for issue in obs.issues) for obs in (before, after))
        old_horizontal = Counter(t.tile for t in before.player(self_slot).river
                                 if t.orientation.startswith('horizontal')) if visible else Counter()
        new_horizontal = Counter(t.tile for t in own.river
                                 if t.orientation.startswith('horizontal')) if visible else Counter()
        declared = visible and (new_horizontal - old_horizontal)[action.tile] > 0
        return _verified('riichi_selected_tile_removed', **result.evidence,
                         riichi_declared=True if declared else None)
    if action.kind not in ('pon', 'kan', 'nuki'):
        return _unknown('unsupported_action')
    try:
        old, new = _own_melds(before, self_slot), _own_melds(after, self_slot)
    except KeyError:
        return _unknown('self_slot_unavailable')
    if old is None or new is None:
        return _unknown('own_meld_structure_unavailable')
    if not _local_tiles_possible(before, old) or not _local_tiles_possible(after, new):
        return _unknown('own_physical_tile_count_conflict')
    if action.kind == 'nuki':
        if _melds(old) != _melds(new):
            return _unknown('own_meld_changed_during_nuki')
        count_known = (old.kita_count is not None and new.kita_count is not None and all(
            obs.field_available(f'kita_{self_slot}') and not any(
                issue.startswith(f'kita_{self_slot}:') for issue in obs.issues)
            for obs in (before, after)))
        increment = count_known and new.kita_count == old.kita_count + 1
        if count_known and new.kita_count not in (old.kita_count, old.kita_count + 1):
            return _unknown('kita_count_conflict')
        if increment:
            # The North area proves the extraction. Returning to the original
            # concealed count proves one supplement, including North -> North.
            # No wall OCR or exact replacement identity is a dependency.
            if len(after.hand) != len(before.hand):
                return _unknown('kita_increment_observed_replacement_count_pending')
            replacement = _replacement(before, after, Counter({'4z': 1}))
            return _verified('kita_increment_and_hand_count_restored', consumed_tiles=['4z'],
                             replacement_tile=replacement, kita_increment_observed=True,
                             before_kita_count=old.kita_count, after_kita_count=new.kita_count,
                             before_hand_count=len(before.hand), after_hand_count=len(after.hand),
                             replacement_proof='concealed_count_restored_after_kita_increment',
                             public_table_evidence_required=False, wall_count_required=False)
        if count_known:
            return _unknown('kita_count_not_incremented')
        replacement = _replacement(before, after, Counter({'4z': 1}))
        if replacement is None:
            return _unknown('north_removal_or_replacement_unproved')
        # With a hidden counter, a different supplement plus the bound nuki
        # button proves the north removal; an unchanged hand alone does not.
        hand_proof = ('nuki' in (before.action_buttons or ()) and replacement != '4z')
        if not hand_proof:
            return _unknown('north_removal_evidence_unavailable')
        return _verified('north_removed_and_replacement_observed', consumed_tiles=['4z'],
                         replacement_tile=replacement, kita_increment_observed=bool(increment),
                         public_table_evidence_required=False)
    group = _new_meld(old, new)
    old_hand, new_hand = Counter(_labels(before.hand)), Counter(_labels(after.hand))
    if action.kind == 'pon':
        proof = prove_pon_transition(before, after, self_slot)
        if proof is None:
            return _unknown('new_pon_or_exact_two_tile_consumption_unproved')
        if (group is None or group.kind != 'pon' or group.opened is not True or
                any(t.tile is None or t.face_down for t in group.tiles)):
            return _unknown('new_open_pon_not_observed')
        labels = _labels(group.tiles)
        if action.tile is not None and _family(labels) != _family((action.tile,)):
            return _unknown('meld_target_mismatch')
        return _verified('new_pon_and_two_matching_hand_tiles_removed', meld_kind='pon',
                         consumed_tiles=sorted(proof['consumed_tiles']), called_tile=proof['called_tile'],
                         source_slot=proof['source_slot'], source_river_evidence_required=False)
    consumed, kind = None, None
    if group is not None and len(group.tiles) == 4:
        labels = _labels(group.tiles)
        family = _family(labels)
        if family is None:
            return _unknown('kan_faces_incompatible')
        if group.kind == 'ankan' and group.opened is False:
            if all(t.tile is not None and not t.face_down for t in group.tiles):
                consumed = Counter(labels)
            elif (group.tiles[0].face_down and group.tiles[-1].face_down and
                  all(t.tile is not None and not t.face_down for t in group.tiles[1:-1])):
                # Only own known hand may supply the hidden red/ordinary faces.
                consumed = Counter(t.tile for t in before.hand if tile_index(t.tile) == family)
                if sum(consumed.values()) != 4 or Counter(t for t in labels if t is not None) - consumed:
                    consumed = None
            kind = 'ankan'
        elif (group.opened is True and group.kind in ('daiminkan', 'unknown') and
              all(t.tile is not None and not t.face_down for t in group.tiles)):
            # A call consumes three matching hand tiles and adds a supplement.
            # Enumerate the single external face; exactly one physical removal
            # must agree with the observed hand (red identity is preserved).
            candidates = {tuple(sorted((Counter(labels) - Counter({tile: 1})).elements()))
                          for tile in labels}
            matching = [Counter(c) for c in candidates
                        if _replacement(before, after, Counter(c)) is not None]
            if len(matching) == 1:
                consumed = matching[0]
            kind = 'daiminkan'
    elif group is None:
        removed, added = _melds(old) - _melds(new), _melds(new) - _melds(old)
        if sum(removed.values()) == sum(added.values()) == 1:
            previous = next(m for m in old.melds if _meld_signature(m) in removed)
            extended = next(m for m in new.melds if _meld_signature(m) in added)
            labels = _labels(extended.tiles)
            if (previous.kind == 'pon' and extended.kind in ('kakan', 'unknown') and
                    extended.opened is True and len(labels) == 4 and
                    all(t.tile is not None and not t.face_down for t in extended.tiles) and
                    all(t.tile is not None and not t.face_down for t in previous.tiles) and
                    _family(labels) is not None and
                    not Counter(_labels(previous.tiles)) - Counter(labels)):
                consumed = Counter(labels) - Counter(_labels(previous.tiles))
                kind = 'kakan'
    if consumed is None or sum(consumed.values()) != {'ankan': 4, 'daiminkan': 3, 'kakan': 1}.get(kind):
        return _unknown('corresponding_kan_not_proved')
    if action.tile is not None and _family(consumed) != _family((action.tile,)):
        return _unknown('meld_target_mismatch')
    replacement = _replacement(before, after, consumed)
    if replacement is None:
        return _unknown('kan_hand_or_replacement_unproved')
    if kind == 'ankan' and not _physical_labels_possible(
            [*_labels(after.hand), *consumed.elements(),
             *(t.tile for m in old.melds for t in m.tiles if t.tile is not None)]):
        return _unknown('own_physical_tile_count_conflict')
    return _verified('kan_structure_and_replacement_observed', meld_kind=kind,
                     consumed_tiles=sorted(consumed.elements()), replacement_tile=replacement,
                     source_river_evidence_required=False, wall_evidence_required=False)
