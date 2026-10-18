"""Evidence for an executed action, separate from client window dismissal.

These checks prove a local table transition, not that a network request was
acknowledged. In particular, closing a response window cannot distinguish PASS
from the client's timeout. Tile conservation retains physical red-five labels.
"""

from collections import Counter
from dataclasses import dataclass, field
import math

from ..domain import Action
from ..state.schemas import FrameObservation
from ..tiles import tile_index
from ..tile_inventory import TileInventory, observed_meld_knowledge


@dataclass(frozen=True)
class VerificationResult:
    status: str
    reason: str
    evidence: dict = field(default_factory=dict)


def assess_discard_hand_change(action, before, after):
    """User acceptance contract: exactly the requested physical tile disappeared.

    River visibility, calls by opponents and wall progress are not dependencies.
    Red and ordinary fives remain different physical labels; sorting is allowed.
    """
    unknown = lambda reason: VerificationResult("unknown", reason)
    if action.kind != "discard" or action.tile is None or action.hand_index is None:
        return unknown("discard_target_unbound")
    if (after.timestamp <= before.timestamp or after.frame_index <= before.frame_index or
            before.active_slots != after.active_slots or
            any(getattr(before, name) is not None and getattr(after, name) is not None and
                getattr(before, name) != getattr(after, name)
                for name in ('round_wind', 'hand_number', 'honba'))):
        return unknown("different_or_unordered_window")
    if any(issue.startswith("hand") for obs in (before, after) for issue in obs.issues):
        return unknown("hand_observation_uncertain")
    old, new = tuple(t.tile for t in before.hand), tuple(t.tile for t in after.hand)
    if (any(t.tile is None or t.face_down for obs in (before, after) for t in obs.hand) or
            not 0 <= action.hand_index < len(old) or old[action.hand_index] != action.tile):
        return unknown("hand_or_target_unrecognized")
    removed, added = Counter(old) - Counter(new), Counter(new) - Counter(old)
    if len(new) == len(old) - 1 and not added and sum(removed.values()) == 1:
        tile = next(iter(removed))
        if tile != action.tile:
            return VerificationResult("failed", "wrong_tile_removed_from_hand",
                                      {"expected_tile": action.tile, "removed_tile": tile})
        return VerificationResult("verified", "requested_tile_removed_from_hand", {
            "discarded_tile": tile, "before_hand_count": len(old), "after_hand_count": len(new),
            "proof": "exact_requested_physical_tile_removed", "river_evidence_required": False})
    return unknown("hand_unchanged" if Counter(old) == Counter(new) else "hand_change_not_single_discard")


def _labels(tiles):
    return tuple(tile.tile for tile in tiles)


def _meld_signature(meld):
    # Geometry and confidence can change without changing the physical meld.
    return (meld.kind, meld.opened,
            tuple(sorted((tile.tile or "?", tile.face_down) for tile in meld.tiles)))


def _melds(player):
    return Counter(_meld_signature(meld) for meld in player.melds)


def _physical_tiles_possible(observation):
    labels = list(_labels(observation.hand)) + list(_labels(observation.dora_indicators))
    for player in observation.players:
        labels.extend(_labels(player.river))
        if player.kita_count is not None:
            labels.extend(["4z"] * player.kita_count)
    known = [label for label in labels if label is not None]
    try:
        inventory = TileInventory.build(known, (observed_meld_knowledge(meld)
                                               for player in observation.players for meld in player.melds))
        inventory.validate_family_counts()
        known = inventory.known
    except ValueError:
        return False
    return all(count <= 1 for label, count in Counter(known).items() if label.startswith("0"))


def _new_meld(old, new):
    removed = _melds(old) - _melds(new)
    added = _melds(new) - _melds(old)
    if removed or sum(added.values()) != 1:
        return None
    signature = next(iter(added))
    return next(meld for meld in new.melds if _meld_signature(meld) == signature)


def _source(before, after, self_slot):
    """A call removes exactly the last discard from one other river."""
    sources = []
    for slot in before.active_slots:
        if slot == self_slot:
            continue
        old, new = before.player(slot), after.player(slot)
        if _melds(old) != _melds(new) or old.kita_count != new.kita_count:
            return None
        if old.river == new.river:
            continue
        # Ignore confidence changes, preserve tile identity and riichi orientation.
        old_river = tuple(tile.semantic() for tile in old.river)
        new_river = tuple(tile.semantic() for tile in new.river)
        if old_river == new_river:
            continue
        if not old_river or new_river != old_river[:-1] or old.river[-1].tile is None:
            return None
        sources.append((slot, old.river[-1].tile))
    return sources[0] if len(sources) == 1 else None


def _others_unchanged(before, after, self_slot):
    return all(before.player(slot).semantic() == after.player(slot).semantic()
               for slot in before.active_slots if slot != self_slot)


def _hand_transition(before, after, consumed, *, replacement=False, other_draw=False):
    old, new = Counter(_labels(before.hand)), Counter(_labels(after.hand))
    if any(label is None for label in (*old, *new)) or consumed - old:
        return None
    retained = old - consumed
    wall_delta = before.remaining_tiles - after.remaining_tiles
    if not replacement:
        return {} if new == retained and wall_delta in ((0, 1) if other_draw else (0,)) else None
    # A complete retained hand plus exactly one wall tile proves a replacement,
    # even when auto-sort has already removed the physical drawn-tile gap.
    added = new - retained
    if retained - new or sum(added.values()) != 1 or wall_delta != 1:
        return None
    if after.drawn_index is not None:
        if not 0 <= after.drawn_index < len(after.hand):
            return None
        if after.hand[after.drawn_index].tile != next(iter(added)):
            return None
    return {"replacement_tile": next(iter(added)),
            "replacement_source": "own_hand_conservation_and_wall_draw"}


def confirm_anchored_discard(action, before, anchor, after, self_slot):
    """Confirm an earlier conserved discard when later opponent draws interleave.

    The caller must still require a stable final observation. A single earlier
    clean frame is provisional only; neither it nor a late wall delta alone
    verifies execution. Calls and declarations keep their existing contracts.
    """
    rejected = VerificationResult('unknown', 'earlier_discard_not_confirmed')
    if action.kind != 'discard' or anchor is None:
        return rejected
    early = assess_action(action, before, anchor, self_slot)
    late = assess_action(action, before, after, self_slot)
    if (early.status != 'verified' or early.evidence.get('called_by') is not None or
            late.status != 'unknown' or late.reason != 'discard_hand_conservation_unproved' or
            after.timestamp <= anchor.timestamp or after.frame_index <= anchor.frame_index or
            after.remaining_tiles > anchor.remaining_tiles):
        return rejected
    first, final = anchor.player(self_slot), after.player(self_slot)
    if (anchor.round_id != after.round_id or anchor.active_slots != after.active_slots or
            tuple(tile.semantic() for tile in anchor.hand) != tuple(tile.semantic() for tile in after.hand) or
            anchor.drawn_index != after.drawn_index or
            (first.seat, first.dealer, first.kita_count) != (final.seat, final.dealer, final.kita_count) or
            tuple(tile.semantic() for tile in first.river) != tuple(tile.semantic() for tile in final.river) or
            _melds(first) != _melds(final) or
            tuple(tile.semantic() for tile in anchor.dora_indicators) !=
            tuple(tile.semantic() for tile in after.dora_indicators)):
        return rejected
    return VerificationResult('verified', 'corresponding_table_transition', {
        **early.evidence, 'anchor_frame_index': anchor.frame_index,
        'confirmation_frame_index': after.frame_index,
        'anchor_wall_delta': before.remaining_tiles-anchor.remaining_tiles,
        'confirmation_wall_delta': before.remaining_tiles-after.remaining_tiles,
        'proof': 'conserved_early_discard_confirmed_by_stable_local_state'})


def _valid_box(box):
    return (box is not None and len(box) == 4 and
            all(type(value) in (int, float) and math.isfinite(value) for value in box) and
            box[0] < box[2] and box[1] < box[3])


def _outside_ui(box, observation):
    return (_valid_box(box) and bool(observation.ui_occlusions) and
            all(_valid_box(ui) and (box[2] < ui[0] or ui[2] < box[0] or
                                   box[3] < ui[1] or ui[3] < box[1])
                for ui in observation.ui_occlusions))


def _partial_discard_evidence(before, after, self_slot, old, new):
    """A revealed old river tile is not evidence for a newly executed discard."""
    meld_field = f"melds_{self_slot}"
    if any(not obs.field_available(meld_field) for obs in (before, after)):
        # 14 -> 13 concealed tiles prove zero declared melds. Positive meld
        # counts cannot establish hidden meld type, faces or unchanged state.
        if len(before.hand) != 14 or len(after.hand) != 13 or old.melds or new.melds:
            return "self_meld_structure_unavailable"
    river_field = f"river_{self_slot}"
    if not before.field_available(river_field):
        if not new.river or not _outside_ui(new.river[-1].box, before):
            return "discard_may_have_been_hidden_before_action"
        # Match surviving prefix positions too: duplicate tile identities must
        # not make a moved/revealed physical tile look like a new append.
        if any(not _valid_box(a.box) or a.box != b.box
               for a, b in zip(old.river, new.river[:-1])):
            return "river_positions_unproved_across_occlusion"
    if not after.field_available(river_field):
        if not new.river or not _outside_ui(new.river[-1].box, after):
            return "discard_still_occluded"
    return None


def assess_action(action: Action, before: FrameObservation,
                  after: FrameObservation, self_slot: str) -> VerificationResult:
    """Legacy whole-table proof for historical/offline evidence analysis.

    The live loop uses execution_verification.assess_execution_action, whose
    dependencies are local to each action. Keep this strict contract for old
    archived proofs and compatibility callers.
    """
    def unknown(reason, **evidence):
        return VerificationResult("unknown", reason, evidence)

    def verified(**evidence):
        return VerificationResult("verified", "corresponding_table_transition", evidence)

    if before.actionable_issues or after.actionable_issues:
        return unknown("observation_uncertain")
    partial = bool(before.unavailable_regions or after.unavailable_regions)
    if partial:
        if any(not obs.ui_occlusions for obs in (before, after) if obs.unavailable_regions):
            return unknown("occlusion_evidence_unavailable")
        # Calls, replacement draws and declarations need their own complete
        # source/structure proof. A partial table alone cannot establish it.
        if action.kind != "discard" and not (action.kind == "riichi" and action.hand_index is not None):
            return unknown("action_dependencies_unavailable")
    if not _physical_tiles_possible(before) or not _physical_tiles_possible(after):
        return unknown("physical_tile_count_conflict")
    if (before.round_id is None or before.round_id != after.round_id or
            before.active_slots != after.active_slots or
            after.timestamp <= before.timestamp or after.frame_index <= before.frame_index):
        return unknown("different_or_unordered_window")
    try:
        old, new = before.player(self_slot), after.player(self_slot)
    except KeyError:
        return unknown("self_slot_unavailable")
    if before.remaining_tiles is None or after.remaining_tiles is None:
        return unknown("wall_count_unavailable")
    if action.kind == "pass":
        if before.action_buttons and after.action_buttons == ():
            return unknown("response_closed_without_acceptance_evidence",
                           client_timeout_possible=True,
                           self_meld_changed=_melds(old) != _melds(new))
        return unknown("response_still_open_or_unavailable")
    if action.kind in ("ron", "tsumo", "abort"):
        return unknown("result_attribution_required")
    if action.kind == 'riichi' and action.hand_index is None:
        return unknown('riichi_discard_target_unbound')

    old_river = tuple(tile.semantic() for tile in old.river)
    new_river = tuple(tile.semantic() for tile in new.river)
    if action.kind in ("discard", "riichi"):
        # Within one hand, nuki counts cannot decrease. Revealing a proved zero
        # after a UI blind spot therefore also proves the earlier count was zero.
        # Positive revealed counts remain ambiguous; never assume they were old.
        zero_revealed = (old.kita_count is None and new.kita_count == 0 and
                         not before.field_available(f'kita_{self_slot}') and
                         after.field_available(f'kita_{self_slot}'))
        if (_melds(old) != _melds(new) or
                old.kita_count != new.kita_count and not zero_revealed):
            return unknown("self_structure_changed_during_discard")
        appended = len(new_river) == len(old_river) + 1 and new_river[:-1] == old_river
        called_by = None
        if not appended:
            if partial:
                return unknown("corresponding_visible_discard_not_observed")
            # A just-discarded tile may already be in another player's meld.
            # Prove the sole call and own-hand removal together; do not require
            # the transient river frame to have been captured.
            if action.kind != "discard" or new_river != old_river or action.tile is None:
                return unknown("corresponding_discard_not_observed")
            for slot in before.active_slots:
                if slot == self_slot:
                    continue
                previous, current = before.player(slot), after.player(slot)
                group = _new_meld(previous, current)
                if group is None:
                    if previous.semantic() != current.semantic():
                        return unknown("interleaved_changes_during_called_discard")
                    continue
                labels = _labels(group.tiles)
                if (called_by is not None or group.kind not in ("pon", "daiminkan") or
                        group.opened is not True or action.tile not in labels or
                        not all(label is not None for label in labels) or
                        len({tile_index(label) for label in labels}) != 1 or
                        tuple(t.semantic() for t in previous.river) != tuple(t.semantic() for t in current.river) or
                        previous.kita_count != current.kita_count):
                    return unknown("called_discard_source_unproved")
                called_by = slot
            if called_by is None:
                return unknown("corresponding_discard_not_observed")
        if partial:
            missing = _partial_discard_evidence(before, after, self_slot, old, new)
            if missing:
                return unknown(missing)
        discarded = new.river[-1].tile if appended else action.tile
        if discarded is None or (action.tile is not None and discarded != action.tile):
            if discarded is not None and _hand_transition(before, after, Counter({discarded: 1}), other_draw=True) is not None:
                return VerificationResult("failed", "different_discard_observed",
                                          {"requested_tile": action.tile, "discarded_tile": discarded})
            return unknown("discard_target_mismatch")
        if action.kind == "discard" or (action.kind == "riichi" and action.hand_index is not None):
            index = action.hand_index
            if index is None or not 0 <= index < len(before.hand) or before.hand[index].tile != action.tile:
                return unknown("original_hand_target_unavailable")
        if action.kind == "riichi" and not new.river[-1].orientation.startswith("horizontal"):
            return unknown("riichi_declaration_not_observed")
        if action.kind == "riichi" and any(tile.orientation.startswith("horizontal") for tile in old.river):
            return unknown("riichi_already_declared")
        consumed = Counter({discarded: 1})
        if _hand_transition(before, after, consumed, other_draw=called_by is None) is None:
            return unknown("discard_hand_conservation_unproved")
        return verified(discarded_tile=discarded,
                        riichi_declared=action.kind == "riichi", called_by=called_by)

    if old_river != new_river:
        return unknown("self_discard_interleaved")
    if action.kind == "nuki":
        if (old.kita_count is None or new.kita_count != old.kita_count + 1 or
                _melds(old) != _melds(new) or not _others_unchanged(before, after, self_slot)):
            return unknown("corresponding_kita_transition_not_observed")
        consumed = Counter({"4z": 1})
        replacement = _hand_transition(before, after, consumed, replacement=True)
        if replacement is None:
            return unknown("kita_hand_or_replacement_unproved")
        return verified(consumed_tiles=["4z"], **replacement)

    if action.kind not in ("pon", "kan") or old.kita_count != new.kita_count:
        return unknown("unsupported_or_unrelated_transition")
    group = _new_meld(old, new)
    source = _source(before, after, self_slot)
    consumed = None
    kind = None
    if group is not None:
        labels = _labels(group.tiles)
        if action.kind == "pon":
            if group.kind != "pon" or group.opened is not True or source is None:
                return unknown("corresponding_pon_source_not_observed")
            consumed = Counter(labels)
            if consumed[source[1]] < 1:
                return unknown("called_tile_not_in_meld")
            consumed.subtract([source[1]])
            consumed = +consumed
            kind = "pon"
        elif len(labels) == 4:
            if len({tile_index(label) for label in labels if label is not None}) != 1:
                return unknown("kan_faces_incompatible")
            if source is not None and group.opened is True and group.kind in ("daiminkan", "unknown"):
                consumed = Counter(labels)
                if consumed[source[1]] < 1:
                    return unknown("called_tile_not_in_meld")
                consumed.subtract([source[1]])
                consumed = +consumed
                kind = "daiminkan"
            elif source is None and group.kind == "ankan" and group.opened is False and _others_unchanged(before, after, self_slot):
                # Back identities come from the *own* hand, with the observed
                # replacement removed first when a draw marker is available.
                retained = list(_labels(after.hand))
                if after.drawn_index is not None and 0 <= after.drawn_index < len(retained):
                    retained.pop(after.drawn_index)
                removed = Counter(_labels(before.hand)) - Counter(retained)
                known = Counter(label for label in labels if label is not None)
                if all(label is not None for label in labels):
                    consumed = Counter(labels)
                    kind = "ankan"
                elif (sum(removed.values()) == 4 and not known - removed and
                        None not in removed and len({tile_index(label) for label in removed}) == 1):
                    consumed = removed
                    kind = "ankan"
            else:
                return unknown("kan_type_or_source_unproved")
    elif action.kind == "kan" and len(old.melds) == len(new.melds) and source is None and _others_unchanged(before, after, self_slot):
        # Kakan is one old pon replaced by a superset of exactly four faces.
        removed = _melds(old) - _melds(new)
        added = _melds(new) - _melds(old)
        if sum(removed.values()) == sum(added.values()) == 1:
            previous = next(meld for meld in old.melds if _meld_signature(meld) in removed)
            extended = next(meld for meld in new.melds if _meld_signature(meld) in added)
            labels = _labels(extended.tiles)
            if (previous.kind == "pon" and extended.kind in ("kakan", "unknown") and
                    extended.opened is not False and len(labels) == 4 and
                    all(label is not None for label in labels) and
                    len({tile_index(label) for label in labels}) == 1 and
                    not Counter(_labels(previous.tiles)) - Counter(labels)):
                consumed = Counter(labels) - Counter(_labels(previous.tiles))
                kind = "kakan"
    if consumed is None or not consumed or None in consumed:
        return unknown("corresponding_meld_not_proved")
    if action.tile is not None and any(tile_index(label) != tile_index(action.tile) for label in consumed):
        return unknown("meld_target_mismatch")
    replacement = _hand_transition(before, after, consumed, replacement=action.kind == "kan")
    if replacement is None:
        return unknown("meld_hand_or_replacement_unproved")
    return verified(meld_kind=kind, consumed_tiles=sorted(consumed.elements()),
                    source_slot=source[0] if source else None,
                    called_tile=source[1] if source else None, **replacement)


def verify_action(action, before, after, self_slot) -> bool:
    """Compatibility predicate; unknown evidence is never counted as success."""
    return assess_action(action, before, after, self_slot).status == "verified"
