from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from ..tiles import tile_index, tile_name
from ..tile_inventory import TileInventory, observed_meld_knowledge
from .schemas import FrameObservation, ObservedMeld, ObservedTile
from .meld_transition import unmatched_melds, prove_pon_transition
from .decision import OPTIONAL_DECISION_FIELDS


def _labels(tiles: tuple[ObservedTile, ...]) -> tuple[str, ...]:
    return tuple(tile.tile for tile in tiles if tile.tile is not None)


def _meld_labels(
    melds: tuple[ObservedMeld, ...],
) -> tuple[tuple[str | None, ...], ...]:
    return tuple(tuple(tile.tile for tile in meld.tiles) for meld in melds)


def _normalized_tile(tile: str) -> str:
    return "5" + tile[1] if len(tile) == 2 and tile[0] == "0" else tile


def _meld_can_be_legal(labels: tuple[str | None, ...], *, allow_chi: bool) -> bool:
    """Return whether known labels admit at least one legal completion."""
    if len(labels) not in (3, 4):
        return False
    known = tuple(_normalized_tile(tile) for tile in labels if tile is not None)
    if not known:
        return True
    same_tile = len(set(known)) == 1
    if len(labels) == 4 or same_tile:
        return same_tile
    if not allow_chi or len(labels) != 3:
        return False
    if any(len(tile) != 2 or tile[1] not in "mps" or not tile[0].isdigit()
           for tile in known):
        return False
    if len({tile[1] for tile in known}) != 1:
        return False
    ranks = {int(tile[0]) for tile in known}
    if len(ranks) != len(known):
        return False
    return any(ranks.issubset({start, start + 1, start + 2})
               for start in range(1, 8))


def _meld_can_include_called_tile(
    labels: tuple[str | None, ...], called_tile: str, *, allow_chi: bool,
) -> bool:
    normalized_called = _normalized_tile(called_tile)
    if any(tile is not None and _normalized_tile(tile) == normalized_called
           for tile in labels):
        return _meld_can_be_legal(labels, allow_chi=allow_chi)
    for index, tile in enumerate(labels):
        if tile is not None:
            continue
        candidate = labels[:index] + (called_tile,) + labels[index + 1:]
        if _meld_can_be_legal(candidate, allow_chi=allow_chi):
            return True
    return False


def _single_removal(before: tuple[str, ...], after: tuple[str, ...]) -> bool:
    if len(before) != len(after) + 1:
        return False
    return any(before[:index] + before[index + 1:] == after
               for index in range(len(before)))


def _river_delta(before: tuple[str, ...], after: tuple[str, ...]):
    """Return (added, removed) for one local, order-preserving river change."""
    if before == after:
        return (), ()
    if len(after) == len(before) + 1 and after[:-1] == before:
        return (after[-1],), ()
    if _single_removal(before, after):
        removed = next(tile for index, tile in enumerate(before)
                       if before[:index] + before[index + 1:] == after)
        return (), (removed,)
    # One called tile can disappear while the caller adds one discard.
    if len(after) == len(before):
        for index, removed in enumerate(before):
            shortened = before[:index] + before[index + 1:]
            if after[:-1] == shortened:
                return (after[-1],), (removed,)
    return None


class StateValidator:
    """Deterministic validation for fixed-ROI snapshots and local transitions."""

    def __init__(self, *, player_count: int = 3, starting_points: int = 35000,
                 removed_tiles=(), red_tiles=("0p", "0s"), self_slot: str = "bottom",
                 allow_chi: bool = False):
        if player_count not in (3, 4) or starting_points <= 0:
            raise ValueError("Invalid game validation configuration")
        self.player_count = player_count
        self.starting_points = starting_points
        self.removed_tiles = frozenset(removed_tiles)
        self.red_tiles = frozenset(red_tiles)
        self.self_slot = self_slot
        self.allow_chi = allow_chi

    @classmethod
    def load(cls, path: str | Path) -> "StateValidator":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cls(player_count=int(data["players"]),
                   starting_points=int(data["starting_points"]),
                   removed_tiles=data.get("removed_tiles", ()),
                   red_tiles=data.get("red_tiles", ()),
                   self_slot="bottom",
                   allow_chi=bool(data.get("allow_chi", True)))

    def validate_observation(self, observation: FrameObservation, *, allow_ui_occlusion=False,
                             allow_public_uncertainty=False) -> tuple[str, ...]:
        partial = (allow_ui_occlusion or allow_public_uncertainty) and bool(observation.unavailable_regions)
        issues = list(observation.actionable_issues if partial else observation.issues)
        fields = {f"{prefix}_{slot}" for prefix in ("river", "melds", "kita")
                  for slot in observation.active_slots}
        if allow_public_uncertainty:
            fields |= OPTIONAL_DECISION_FIELDS
        supported = (set(observation.unavailable_regions).issubset(fields) and
                     len(set(observation.unavailable_regions)) == len(observation.unavailable_regions))
        geometry = bool(observation.ui_occlusions) and all(
            len(box) == 4 and all(type(value) in (int, float) and math.isfinite(value) for value in box) and
            box[0] < box[2] and box[1] < box[3] for box in observation.ui_occlusions)
        vision_fields = set(observation.perception_unavailable_regions)
        vision_supported = (allow_public_uncertainty and vision_fields.issubset(fields) and
                            f"melds_{self.self_slot}" not in vision_fields and
                            vision_fields.issubset(observation.unavailable_regions))
        evidence = all(
            (name in vision_fields and vision_supported and any(
                issue.startswith(name + ":") for issue in observation.issues)) or
            (geometry and f"{name}: UI occlusion; region unavailable" in observation.issues)
            for name in observation.unavailable_regions)
        if vision_fields and not vision_supported:
            return ("Public vision uncertainty is allowed only in decision snapshots",)
        if partial:
            # Validate every current known field. No guessed zero/past snapshot
            # is inserted to make the full-table completeness predicate pass.
            required = (supported and evidence and
                        not issues and observation.round_wind is not None and observation.hand_number is not None and
                        all(getattr(observation, name) is not None or
                            not observation.field_available(name)
                            for name in ('remaining_tiles', 'honba', 'riichi_sticks')) and
                        len(observation.players) == len(observation.active_slots) and
                        all(p.seat is not None and p.dealer is not None and
                            (p.score is not None or not p.score_available) and
                            (p.kita_count is not None or not observation.field_available(f"kita_{p.slot}"))
                            for p in observation.players) and
                        all(t.tile is not None for t in (*observation.hand, *observation.dora_indicators)) and
                        (observation.drawn_index is None or 0 <= observation.drawn_index < len(observation.hand)))
        else:
            required = observation.complete
        if not required:
            if not issues:
                issues.append("Observation is incomplete")
            return tuple(dict.fromkeys(issues))
        if (len(observation.active_slots) != self.player_count or
                len(set(observation.active_slots)) != self.player_count or
                self.self_slot not in observation.active_slots):
            issues.append("Active slots do not match the configured player count")
        if tuple(player.slot for player in observation.players) != observation.active_slots:
            issues.append("Player observations are not aligned with active slots")
            return tuple(dict.fromkeys(issues))
        if any(p.kita_count is not None and
               (type(p.kita_count) is not int or not 0 <= p.kita_count <= 4)
               for p in observation.players):
            issues.append("Kita counts must be unknown or integers from zero to four")
            return tuple(dict.fromkeys(issues))
        if partial and any(not observation.field_available(f"kita_{p.slot}") and p.kita_count is not None
                           for p in observation.players):
            issues.append("UI-hidden kita count must remain unknown")
        seats = [player.seat for player in observation.players]
        expected_seats = set(("1z", "2z", "3z", "4z")[:self.player_count])
        if set(seats) != expected_seats:
            issues.append("Seat winds do not match the configured player count")
        elif len(set(seats)) != len(seats):
            issues.append("Seat winds must be unique")
        if sum(player.dealer is True for player in observation.players) != 1:
            issues.append("Exactly one active player must be dealer")
        if observation.round_wind not in ("1z", "2z", "3z", "4z"):
            issues.append("Round wind is invalid")
        if (observation.hand_number is None or
                not 1 <= observation.hand_number <= self.player_count):
            issues.append("Hand number is invalid for the configured player count")
        if any(not (partial and name in vision_fields and value is None) and
               (type(value) is not int or not 0 <= value <= bound)
               for name, value, bound in (('remaining_tiles', observation.remaining_tiles, 200),
                                         ('honba', observation.honba, 99),
                                         ('riichi_sticks', observation.riichi_sticks, 99))):
            issues.append("Round counters are outside supported ranges")
        scores = [player.score for player in observation.players]
        if any(not player.score_available and player.score is not None
               for player in observation.players):
            issues.append("Scores must be bounded by their availability contract")
        if all(score is None for score in scores):
            pass  # Score OCR unavailable: never fabricate or conserve guesses.
        elif any(score is None or not -999900 <= score <= 999900 or score % 100
                 for score in scores):
            issues.append("Scores must be bounded multiples of 100")
        elif observation.riichi_sticks is not None and sum(scores) + observation.riichi_sticks * 1000 != (
                self.player_count * self.starting_points):
            issues.append("Scores plus riichi deposits do not conserve match points")

        self_player = observation.player(self.self_slot)
        meld_count = len(self_player.melds)
        if partial and not observation.field_available(f"melds_{self.self_slot}"):
            possible = [m for m in range(5) if len(observation.hand) in (13-3*m, 14-3*m)]
            if len(possible) != 1:
                return ("Concealed hand cannot establish a legal meld count",)
            meld_count = possible[0]
        expected_hand = 13 - 3 * meld_count
        if len(observation.hand) not in (expected_hand, expected_hand + 1):
            issues.append(
                f"Concealed hand has {len(observation.hand)} tiles; expected "
                f"{expected_hand} or {expected_hand + 1}")
        if observation.drawn_index is not None and len(observation.hand) != expected_hand + 1:
            issues.append("Drawn-tile marker requires the extra concealed tile")
        for player in observation.players:
            if any(len(meld.tiles) not in (3, 4) for meld in player.melds):
                issues.append(f"{player.slot} has a meld outside 3 or 4 tiles")
            for meld in player.melds:
                labels = tuple(tile.tile for tile in meld.tiles)
                if not _meld_can_be_legal(labels, allow_chi=self.allow_chi):
                    issues.append(f"{player.slot} has an invalid meld: {labels}")

        required_visible = [tile.tile for tile in observation.hand]
        required_visible.extend(tile.tile for tile in observation.dora_indicators)
        for player in observation.players:
            required_visible.extend(tile.tile for tile in player.river)
        if any(tile is None for tile in required_visible):
            issues.append("Visible tile set contains unknown labels")
        else:
            visible = list(required_visible)
            for player in observation.players:
                visible.extend(tile.tile for meld in player.melds
                               for tile in meld.tiles if tile.tile is not None)
                visible.extend("4z" for _ in range(player.kita_count or 0))
            invalid = []
            for tile in visible:
                try:
                    tile_index(tile)
                except ValueError:
                    invalid.append(tile)
            if invalid:
                issues.append(f"Invalid tile labels: {sorted(set(invalid))}")
            forbidden = sorted(tile for tile in visible if tile in self.removed_tiles)
            if forbidden:
                issues.append(f"Removed sanma tiles are visible: {forbidden}")
            counts = Counter(("5" + tile[1]) if tile[0] == "0" else tile
                             for tile in visible)
            if not invalid:
                inventory = TileInventory.build(
                    [*required_visible, *["4z" for player in observation.players
                                         for _ in range(player.kita_count or 0)]],
                    (observed_meld_knowledge(meld) for player in observation.players
                     for meld in player.melds))
                counts = Counter({tile_name(i): count for i, count in enumerate(inventory.family_counts)})
            excessive = sorted(tile for tile, count in counts.items() if count > 4)
            if excessive:
                issues.append(f"More than four visible copies: {excessive}")
            red_counts = Counter(tile for tile in visible if tile[0] == "0")
            unsupported_red = sorted(tile for tile in red_counts if tile not in self.red_tiles)
            excessive_red = sorted(tile for tile, count in red_counts.items() if count > 1)
            if unsupported_red:
                issues.append(f"Unsupported red tiles: {unsupported_red}")
            if excessive_red:
                issues.append(f"More than one red copy: {excessive_red}")
        return tuple(dict.fromkeys(issues))

    def validate_available_transition(self, before: FrameObservation,
                                      after: FrameObservation) -> tuple[str, ...]:
        """Check invariants that UI-hidden public fields cannot excuse.

        Missing forward history is recorded by RESYNC, without guessed local
        events. Reverse counters, seat changes and dora replacement are errors.
        """
        if before.active_slots != after.active_slots:
            return ("Active slot layout changed within the match",)
        if before.round_id != after.round_id:
            return () if self._valid_round_transition(before, after) else ("Round transition is not locally valid",)
        issues = []
        if any((before.player(slot).seat, before.player(slot).dealer) !=
               (after.player(slot).seat, after.player(slot).dealer) for slot in before.active_slots):
            issues.append("Seat winds or dealer changed inside one round")
        if after.remaining_tiles > before.remaining_tiles:
            issues.append("Remaining tiles increased inside one round")
        if after.riichi_sticks < before.riichi_sticks:
            issues.append("Riichi deposits decreased inside one round")
        for slot in before.active_slots:
            old_player, new_player = before.player(slot), after.player(slot)
            if (before.field_available(f"kita_{slot}") and after.field_available(f"kita_{slot}") and
                    new_player.kita_count < old_player.kita_count):
                issues.append(f"{slot} kita count decreased inside one round")
            if (before.field_available(f"melds_{slot}") and after.field_available(f"melds_{slot}") and
                    len(new_player.melds) < len(old_player.melds)):
                issues.append(f"{slot} meld count decreased inside one round")
        old, new = _labels(before.dora_indicators), _labels(after.dora_indicators)
        if not (new == old or len(new) == len(old)+1 and new[:-1] == old):
            issues.append("Dora indicators changed non-locally")
        return tuple(issues)

    def validate_transition(self, before: FrameObservation,
                            after: FrameObservation) -> tuple[str, ...]:
        issues = list(self.validate_observation(after))
        if issues:
            return tuple(issues)
        if before.active_slots != after.active_slots:
            issues.append("Active slot layout changed within the match")
            return tuple(issues)
        if after.timestamp <= before.timestamp or after.frame_index <= before.frame_index:
            issues.append("Committed observations must be strictly ordered")
        before_base = (before.round_wind, before.hand_number)
        after_base = (after.round_wind, after.hand_number)
        if before_base != after_base or before.honba != after.honba:
            if not self._valid_round_transition(before, after):
                issues.append("Round transition is not locally valid")
            return tuple(issues)

        if after.remaining_tiles > before.remaining_tiles:
            issues.append("Remaining tiles increased inside one round")
        elif before.remaining_tiles - after.remaining_tiles > 1:
            issues.append("More than one wall draw occurred between stable snapshots")
        if after.riichi_sticks < before.riichi_sticks:
            issues.append("Riichi deposits decreased inside one round")
        elif after.riichi_sticks - before.riichi_sticks > 1:
            issues.append("Multiple riichi declarations occurred between stable snapshots")

        before_scores = {player.slot: player.score for player in before.players}
        after_scores = {player.slot: player.score for player in after.players}
        stick_delta = after.riichi_sticks - before.riichi_sticks
        if any(score is None for score in (*before_scores.values(), *after_scores.values())):
            changes = {}  # Score OCR unavailable: no score-based transition checks.
        else:
            changes = {slot: after_scores[slot] - before_scores[slot]
                       for slot in before.active_slots if after_scores[slot] != before_scores[slot]}
        if changes and not (stick_delta == 1 and list(changes.values()) == [-1000]):
            issues.append("Score change is not a single riichi deposit")
        if (stick_delta == 1 and list(changes.values()) != [-1000] and
                all(score is not None for score in (*before_scores.values(), *after_scores.values()))):
            issues.append("Riichi deposit lacks exactly one 1000-point payment")

        meld_changes = self._meld_changes(before, after)
        if meld_changes is None:
            issues.append("Melds changed non-locally")
            meld_change_count = 0
        else:
            meld_change_count = len(meld_changes)
            if meld_change_count > 1:
                issues.append("Multiple meld changes occurred between stable snapshots")
        total_added = 0
        total_removed = 0
        removed_tiles = []
        river_deltas = {}
        for slot in before.active_slots:
            river_before = _labels(before.player(slot).river)
            river_after = _labels(after.player(slot).river)
            delta = _river_delta(river_before, river_after)
            if delta is None:
                issues.append(f"{slot} river changed non-locally")
                continue
            river_deltas[slot] = delta
            total_added += len(delta[0])
            total_removed += len(delta[1])
            removed_tiles.extend(delta[1])
        if total_added > 1:
            issues.append("Multiple discards occurred between stable snapshots")
        if total_removed > 1 or (total_removed and not meld_change_count):
            issues.append("River tile removal is not explained by one meld")
        called_melds = [change for change in (meld_changes or ())
                        if change[1] == "add"]
        three_tile_calls = [change for change in called_melds
                            if len(change[2]) == 3]
        self_discards = river_deltas.get(self.self_slot, ((), ()))[0]
        pon_proof = prove_pon_transition(before, after, self.self_slot, self_discards)
        if three_tile_calls and total_removed != 1 and not (total_removed == 0 and pon_proof):
            issues.append("A three-tile meld lacks exactly one called river tile")
        if total_removed == 1:
            if len(called_melds) != 1 or meld_change_count != 1:
                issues.append("River tile removal is not paired with one new meld")
            elif not _meld_can_include_called_tile(
                    called_melds[0][2], removed_tiles[0],
                    allow_chi=self.allow_chi):
                issues.append("Called river tile is incompatible with the new meld")

        kita_delta = 0
        kita_by_slot = {}
        for slot in before.active_slots:
            old = before.player(slot).kita_count
            new = after.player(slot).kita_count
            if new < old:
                issues.append(f"{slot} kita count decreased")
            change = max(0, new - old)
            kita_by_slot[slot] = change
            kita_delta += change
        if kita_delta > 1:
            issues.append("Multiple kita events occurred between stable snapshots")
        if meld_change_count + kita_delta > 1:
            issues.append("Multiple meld or kita events occurred between stable snapshots")

        hand_before = Counter(_labels(before.hand))
        hand_after = Counter(_labels(after.hand))
        wall_delta = before.remaining_tiles - after.remaining_tiles
        self_discards = river_deltas.get(self.self_slot, ((), ()))[0]
        self_meld_changes = [change for change in (meld_changes or ())
                             if change[0] == self.self_slot]
        self_kita = kita_by_slot.get(self.self_slot, 0)
        before_self_melds = before.player(self.self_slot).melds
        after_self_melds = after.player(self.self_slot).melds
        exact_meld_labels = all(
            meld.known_complete for meld in (*before_self_melds, *after_self_melds))
        if exact_meld_labels:
            before_meld_tiles = Counter(
                tile.tile for meld in before_self_melds for tile in meld.tiles)
            after_meld_tiles = Counter(
                tile.tile for meld in after_self_melds for tile in meld.tiles)
            consumed = after_meld_tiles - before_meld_tiles
            if self_meld_changes and total_removed == 1:
                called_tile = removed_tiles[0]
                if consumed[called_tile] < 1:
                    issues.append("Self meld does not contain its called river tile")
                else:
                    consumed[called_tile] -= 1
                    if consumed[called_tile] == 0:
                        del consumed[called_tile]
            elif self_meld_changes and total_removed == 0 and pon_proof:
                consumed.subtract([pon_proof["called_tile"]])
                consumed = +consumed
            if self_kita:
                consumed["4z"] += self_kita
            explained = hand_after + consumed + Counter(self_discards)
            missing = hand_before - explained
            inferred_draws = explained - hand_before
            if missing:
                issues.append("Self hand transition leaves prior tiles unexplained")
            if sum(inferred_draws.values()) > max(0, wall_delta):
                issues.append("Self hand transition requires extra wall draws")
        else:
            hand_removed = hand_before - hand_after
            hand_added = hand_after - hand_before
            removed_count = sum(hand_removed.values())
            added_count = sum(hand_added.values())
            concealed_for_meld = 0
            for _, change, tiles in self_meld_changes:
                if change == "extend":
                    concealed_for_meld += 1
                else:
                    called = int(total_removed == 1)
                    concealed_for_meld += len(tiles) - called
            concealed_for_meld += self_kita
            if added_count > wall_delta:
                issues.append("Partial self meld transition gains unexplained hand tiles")
            if removed_count > concealed_for_meld + len(self_discards):
                issues.append("Partial self meld transition loses unexplained hand tiles")
        dora_before = _labels(before.dora_indicators)
        dora_after = _labels(after.dora_indicators)
        if not (dora_after == dora_before or
                len(dora_after) == len(dora_before) + 1 and
                dora_after[:len(dora_before)] == dora_before):
            issues.append("Dora indicators changed non-locally")
        return tuple(dict.fromkeys(issues))

    def _valid_round_transition(self, before: FrameObservation,
                                after: FrameObservation) -> bool:
        if after.honba is None or before.honba is None:
            return False
        before_key = (before.round_wind, before.hand_number)
        after_key = (after.round_wind, after.hand_number)
        if after_key == before_key:
            return after.honba == before.honba + 1
        winds = ("1z", "2z", "3z", "4z")
        try:
            before_ordinal = winds.index(before.round_wind) * self.player_count + before.hand_number
            after_ordinal = winds.index(after.round_wind) * self.player_count + after.hand_number
        except (ValueError, TypeError):
            return False
        return after_ordinal == before_ordinal + 1 and after.honba == 0

    @staticmethod
    def _meld_changes(before: FrameObservation, after: FrameObservation):
        changes = []
        for slot in before.active_slots:
            old = _meld_labels(before.player(slot).melds)
            new = _meld_labels(after.player(slot).melds)
            if old == new:
                continue
            missing, added = unmatched_melds(before.player(slot).melds, after.player(slot).melds)
            if not missing and not added:
                continue  # UI ordering/orientation does not change a meld.
            if not missing and len(added) == 1:
                changes.append((slot, "add", tuple(t.tile for t in added[0].tiles)))
                continue
            if len(missing) == len(added) == 1:
                previous = tuple(t.tile for t in missing[0].tiles)
                current = tuple(t.tile for t in added[0].tiles)
                if (len(previous) == 3 and len(current) == 4 and
                        all(t is not None for t in (*previous, *current)) and
                        not Counter(previous) - Counter(current) and
                        len({_normalized_tile(t) for t in current}) == 1):
                    changes.append((slot, "extend", current))
                    continue
            if len(new) == len(old) + 1 and new[:-1] == old:
                changes.append((slot, "add", new[-1]))
                continue
            if len(new) == len(old):
                differing = [index for index, (left, right) in enumerate(zip(old, new))
                             if left != right]
                refinements = all(
                    len(old[index]) == len(new[index]) and
                    all(left == right or left is None and right is not None
                        for left, right in zip(old[index], new[index]))
                    for index in differing)
                if differing and refinements:
                    continue
                if len(differing) == 1:
                    old_group = old[differing[0]]
                    new_group = new[differing[0]]
                    if len(old_group) == 3 and len(new_group) == 4:
                        old_known = {_normalized_tile(tile) for tile in old_group
                                     if tile is not None}
                        new_known = {_normalized_tile(tile) for tile in new_group
                                     if tile is not None}
                        if len(old_known | new_known) <= 1:
                            changes.append((slot, "extend", new_group))
                            continue
            return None
        return tuple(changes)
