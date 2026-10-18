from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any

from ..tile_inventory import MeldKnowledge


class GameEventType(StrEnum):
    """Observable game events; PASS belongs to AgentActionType, not here."""

    ROUND_START = "ROUND_START"
    DRAW = "DRAW"
    DISCARD = "DISCARD"
    RIICHI = "RIICHI"
    MELD = "MELD"
    CHI = "CHI"
    PON = "PON"
    KAN = "KAN"
    KITA = "KITA"
    TSUMO = "TSUMO"
    RON = "RON"
    SCORE_UPDATE = "SCORE_UPDATE"
    ROUND_END = "ROUND_END"
    GAME_END = "GAME_END"
    RESYNC = "RESYNC"


class TrackingPhase(StrEnum):
    WAIT_GAME = "WAIT_GAME"
    TRACKING = "TRACKING"
    ROUND_END = "ROUND_END"
    GAME_END = "GAME_END"
    RESYNC = "RESYNC"


@dataclass(frozen=True)
class ObservedTile:
    tile: str | None
    orientation: str = "upright"
    confidence: float = 0.0
    face_down: bool = False
    # Window-local pixels are evidence for UI occlusion, not tile identity.
    box: tuple[float, float, float, float] | None = None

    def semantic(self):
        return {"tile": self.tile, "orientation": self.orientation, "face_down": self.face_down}

    def to_dict(self):
        return {**self.semantic(), "confidence": self.confidence, "box": self.box}


@dataclass(frozen=True)
class ObservedMeld:
    tiles: tuple[ObservedTile, ...]
    kind: str = "unknown"
    opened: bool | None = None

    def __post_init__(self):
        kinds = {"unknown", "pon", "chi", "ankan", "daiminkan", "kakan"}
        if self.kind not in kinds:
            raise ValueError(f"Unknown meld kind: {self.kind}")
        expected = {"pon": True, "chi": True, "ankan": False,
                    "daiminkan": True, "kakan": True}.get(self.kind)
        if self.opened is not None and type(self.opened) is not bool:
            raise ValueError("Meld opened must be bool or None")
        if expected is not None:
            size = 3 if self.kind in {"pon", "chi"} else 4
            if len(self.tiles) != size:
                raise ValueError("Meld kind conflicts with tile count")
            if self.opened is not None and self.opened != expected:
                raise ValueError("Meld kind conflicts with opened state")
            object.__setattr__(self, "opened", expected)
        elif len(self.tiles) == 3:
            if self.opened is False:
                raise ValueError("A declared three-tile meld cannot be closed")
            # Every declared three-tile meld is open; four tiles alone do not
            # distinguish ankan, daiminkan and kakan.
            object.__setattr__(self, "opened", True)
            labels = [tile.tile for tile in self.tiles]
            if all(label is not None for label in labels):
                normalized = {"5" + label[1] if label[0] == "0" else label
                              for label in labels}
                if len(normalized) == 1:
                    object.__setattr__(self, "kind", "pon")

    @property
    def known_complete(self) -> bool:
        return all(tile.tile is not None for tile in self.tiles)

    def semantic(self):
        return {"tiles": [tile.semantic() for tile in self.tiles],
                "kind": self.kind, "opened": self.opened}

    def to_dict(self):
        return {"tiles": [tile.to_dict() for tile in self.tiles],
                "kind": self.kind, "opened": self.opened}


@dataclass(frozen=True)
class PlayerObservation:
    slot: str
    seat: str | None
    dealer: bool | None
    score: int | None
    river: tuple[ObservedTile, ...] = ()
    melds: tuple[ObservedMeld, ...] = ()
    kita_count: int | None = None
    # False when the pipeline disables score OCR: score stays None and is not
    # required for completeness.
    score_available: bool = True
    riichi: bool | None = None
    riichi_evidence: dict = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return (self.seat is not None and self.dealer is not None and
                (self.score is not None or not self.score_available) and
                self.kita_count is not None and
                all(tile.tile is not None for tile in self.river) and
                all(len(meld.tiles) in (3, 4) for meld in self.melds))

    def semantic(self):
        return {
            "slot": self.slot,
            "seat": self.seat,
            "dealer": self.dealer,
            "score": self.score,
            "score_available": self.score_available,
            "river": [tile.semantic() for tile in self.river],
            "melds": [meld.semantic() for meld in self.melds],
            "kita_count": self.kita_count,
            "riichi": self.riichi,
        }

    def to_dict(self):
        return {
            "slot": self.slot,
            "seat": self.seat,
            "dealer": self.dealer,
            "score": self.score,
            "score_available": self.score_available,
            "river": [tile.to_dict() for tile in self.river],
            "melds": [meld.to_dict() for meld in self.melds],
            "kita_count": self.kita_count,
            "complete": self.complete,
            "riichi": self.riichi,
            "riichi_evidence": self.riichi_evidence,
        }


@dataclass(frozen=True)
class FrameObservation:
    timestamp: float
    frame_index: int
    active_slots: tuple[str, ...]
    round_wind: str | None
    hand_number: int | None
    honba: int | None
    riichi_sticks: int | None
    remaining_tiles: int | None
    players: tuple[PlayerObservation, ...]
    hand: tuple[ObservedTile, ...]
    drawn_index: int | None
    dora_indicators: tuple[ObservedTile, ...]
    action_buttons: tuple[str, ...] | None = None
    turn_indicator: str | None = None
    issues: tuple[str, ...] = ()
    confidence: dict[str, float] = field(default_factory=dict)
    # UI clock is diagnostic/window data. Excluded from semantic fingerprint so
    # each tick does not invalidate table stability or repeat a model request.
    countdown: dict | None = None
    ui_occlusions: tuple[tuple[float, float, float, float], ...] = ()
    unavailable_regions: tuple[str, ...] = ()
    riichi_discard_indices: tuple[int, ...] | None = None
    selection_evidence: dict | None = None
    # Decision-only projection: failed public vision has no UI occlusion box.
    perception_unavailable_regions: tuple[str, ...] = ()

    @property
    def actionable_issues(self) -> tuple[str, ...]:
        # Only explicit UI blind spots in public tile fields can be relaxed.
        # Missing hand, dora, metadata, unknown labels or other errors still stop.
        allowed = {f"{name}: UI occlusion; region unavailable"
                   for name in self.unavailable_regions
                   if name.startswith(("river_", "melds_", "kita_")) or name == 'dora'}
        return tuple(issue for issue in self.issues if issue not in allowed and
                     issue.split(":", 1)[0] not in self.perception_unavailable_regions)

    def field_available(self, name: str) -> bool:
        return name not in self.unavailable_regions

    @property
    def round_id(self) -> str | None:
        if self.round_wind is None or self.hand_number is None or self.honba is None:
            return None
        return f"{self.round_wind}:{self.hand_number}:{self.honba}"

    @property
    def complete(self) -> bool:
        return (not self.issues and not self.unavailable_regions and self.round_id is not None and
                self.riichi_sticks is not None and self.remaining_tiles is not None and
                len(self.players) == len(self.active_slots) and
                all(player.complete for player in self.players) and
                all(tile.tile is not None for tile in self.hand) and
                all(tile.tile is not None for tile in self.dora_indicators) and
                (self.drawn_index is None or 0 <= self.drawn_index < len(self.hand)))

    @property
    def uncertain_fields(self) -> tuple[str, ...]:
        return self.unavailable_regions + tuple(
            f"melds_{player.slot}"
            for player in self.players
            if any(not meld.known_complete for meld in player.melds)
        )

    def player(self, slot: str) -> PlayerObservation:
        try:
            return next(player for player in self.players if player.slot == slot)
        except StopIteration as exc:
            raise KeyError(f"Unknown active slot: {slot}") from exc

    def semantic(self) -> dict[str, Any]:
        return {
            "active_slots": self.active_slots,
            "round_wind": self.round_wind,
            "hand_number": self.hand_number,
            "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
            "remaining_tiles": self.remaining_tiles,
            "players": [player.semantic() for player in self.players],
            "hand": [tile.semantic() for tile in self.hand],
            "drawn_index": self.drawn_index,
            "dora_indicators": [tile.semantic() for tile in self.dora_indicators],
            "action_buttons": self.action_buttons,
            "turn_indicator": self.turn_indicator,
            "unavailable_regions": self.unavailable_regions,
            "riichi_discard_indices": self.riichi_discard_indices,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.semantic(), sort_keys=True, ensure_ascii=False)
        return sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "active_slots": self.active_slots,
            "round_wind": self.round_wind,
            "hand_number": self.hand_number,
            "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
            "remaining_tiles": self.remaining_tiles,
            "players": [player.to_dict() for player in self.players],
            "hand": [tile.to_dict() for tile in self.hand],
            "drawn_index": self.drawn_index,
            "dora_indicators": [tile.to_dict() for tile in self.dora_indicators],
            "action_buttons": self.action_buttons,
            "turn_indicator": self.turn_indicator,
            "round_id": self.round_id,
            "issues": list(self.issues),
            "confidence": self.confidence,
            "complete": self.complete,
            "uncertain_fields": list(self.uncertain_fields),
            "countdown": self.countdown,
            "ui_occlusions": self.ui_occlusions,
            "unavailable_regions": self.unavailable_regions,
            "riichi_discard_indices": self.riichi_discard_indices,
            "selection_evidence": self.selection_evidence,
            "perception_unavailable_regions": self.perception_unavailable_regions,
        }


@dataclass(frozen=True)
class GameEvent:
    type: GameEventType
    timestamp: float
    frame_index: int
    player: str | None = None
    tile: str | None = None
    certainty: str = "observed"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "player": self.player,
            "tile": self.tile,
            "certainty": self.certainty,
            "details": self.details,
        }


@dataclass(frozen=True)
class TrackedMeld:
    tiles: tuple[str | None, ...]
    kind: str = "unknown"
    opened: bool | None = None
    face_down: tuple[bool, ...] = ()
    evidence_frames: tuple[int, ...] = ()

    def __post_init__(self):
        if self.face_down and (len(self.face_down) != len(self.tiles) or
                               any(type(value) is not bool for value in self.face_down)):
            raise ValueError('Tracked meld back flags must match tile count')
        observation = ObservedMeld(tuple(ObservedTile(t) for t in self.tiles),
                                   self.kind, self.opened)
        object.__setattr__(self, "kind", observation.kind)
        object.__setattr__(self, "opened", observation.opened)

    def to_dict(self):
        return {"tiles": self.tiles, "kind": self.kind, "opened": self.opened,
                "face_down": self.face_down, "evidence_frames": self.evidence_frames}


@dataclass(frozen=True)
class PlayerState:
    slot: str
    seat: str
    dealer: bool
    score: int | None
    river: tuple[str, ...]
    discard_history: tuple[str, ...]
    melds: tuple[TrackedMeld, ...]
    riichi: bool | None
    kita_count: int | None
    score_available: bool = True

    def __post_init__(self):
        # Normalize callers using the earlier public tuple schema at the edge.
        object.__setattr__(self, "melds", tuple(
            meld if isinstance(meld, TrackedMeld) else TrackedMeld(tuple(meld))
            for meld in self.melds))

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "seat": self.seat,
            "dealer": self.dealer,
            "score": self.score,
            "score_available": self.score_available,
            "river": self.river,
            "discard_history": self.discard_history,
            "melds": [meld.to_dict() for meld in self.melds],
            "riichi": self.riichi,
            "kita_count": self.kita_count,
        }


@dataclass(frozen=True)
class MahjongState:
    mode: str
    player_count: int
    active_slots: tuple[str, ...]
    self_slot: str
    round_wind: str
    hand_number: int
    honba: int | None
    riichi_sticks: int | None
    remaining_tiles: int | None
    dealer: str
    self_seat: str
    players: tuple[PlayerState, ...]
    hand: tuple[str, ...]
    drawn_tile: str | None
    dora_indicators: tuple[str, ...]
    current_actor: str | None
    last_discard: tuple[str, str] | None
    available_actions: tuple[str, ...]
    uncertain_fields: tuple[str, ...]
    phase: TrackingPhase
    history: tuple[GameEvent, ...]
    history_complete: bool
    source_frame: int
    source_timestamp: float
    unavailable_regions: tuple[str, ...] = ()
    current_snapshot_validated: bool = False
    observation_fingerprint: str | None = None

    @property
    def self_meld_count(self) -> int | None:
        if f"melds_{self.self_slot}" not in self.unavailable_regions:
            return len(self.player(self.self_slot).melds)
        candidates = [count for count in range(5)
                      if len(self.hand) in (13 - 3 * count, 14 - 3 * count)]
        return candidates[0] if len(candidates) == 1 else None

    @property
    def decision_ready(self) -> bool:
        return not self.decision_blockers

    @property
    def decision_blockers(self) -> tuple[str, ...]:
        count = self.self_meld_count
        window_proven = (self.drawn_tile is not None or bool(self.available_actions) or
                         self.current_actor == self.self_slot and count is not None and
                         len(self.hand) == 14 - 3 * count)
        reasons = []
        if self.phase != TrackingPhase.TRACKING:
            reasons.append("tracking_phase:" + self.phase.value)
        if not self.history_complete:
            if not self.current_snapshot_validated:
                reasons.append("snapshot_not_validated")
            if not window_proven:
                reasons.append("action_window_not_proven")
        if self.self_meld_count is None:
            reasons.append("self_meld_count_unknown")
        if self.blocking_uncertain_fields:
            reasons.append("blocking_uncertain_fields")
        return tuple(reasons)

    @property
    def blind_spots(self) -> tuple[str, ...]:
        permitted = [name for name in self.unavailable_regions
                     if name.startswith(("river_", "melds_", "kita_")) or
                     name in ('remaining_tiles', 'honba', 'riichi_sticks', 'dora')]
        for player in self.players:
            unknown = [MeldKnowledge(m.tiles, m.kind, m.opened, m.face_down)
                       for m in player.melds if any(t is None for t in m.tiles)]
            if (player.slot != self.self_slot and unknown and
                    all(m.bounded_concealed_quad for m in unknown)):
                permitted.append(f"melds_{player.slot}")
        return tuple(permitted)

    @property
    def blocking_uncertain_fields(self) -> tuple[str, ...]:
        # Derive unknown melds too: older/manual callers may omit diagnostics.
        unknown = tuple(f"melds_{p.slot}" for p in self.players
                        if any(any(t is None for t in m.tiles) for m in p.melds))
        permitted = set(self.blind_spots)
        return tuple(dict.fromkeys(field for field in (*self.uncertain_fields, *unknown)
                                   if field not in permitted))

    @property
    def round_id(self) -> str | None:
        if self.round_wind is None or self.hand_number is None or self.honba is None:
            return None
        return f"{self.round_wind}:{self.hand_number}:{self.honba}"

    def player(self, slot: str) -> PlayerState:
        try:
            return next(player for player in self.players if player.slot == slot)
        except StopIteration as exc:
            raise KeyError(f"Unknown active slot: {slot}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "player_count": self.player_count,
            "active_slots": self.active_slots,
            "self_slot": self.self_slot,
            "round": self.round_wind,
            "hand_number": self.hand_number,
            "round_id": self.round_id,
            "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
            "remaining_tiles": self.remaining_tiles,
            "dealer": self.dealer,
            "self_seat": self.self_seat,
            "players": {player.slot: player.to_dict() for player in self.players},
            "hand": self.hand,
            "drawn_tile": self.drawn_tile,
            "dora_indicators": self.dora_indicators,
            "current_actor": self.current_actor,
            "last_discard": self.last_discard,
            "available_actions": self.available_actions,
            "uncertain_fields": self.uncertain_fields,
            "blocking_uncertain_fields": self.blocking_uncertain_fields,
            "blind_spots": self.blind_spots,
            "decision_ready": self.decision_ready,
            "phase": self.phase.value,
            "history": [event.to_dict() for event in self.history],
            "history_complete": self.history_complete,
            "source_frame": self.source_frame,
            "source_timestamp": self.source_timestamp,
            "unavailable_regions": self.unavailable_regions,
            "current_snapshot_validated": self.current_snapshot_validated,
            "observation_fingerprint": self.observation_fingerprint,
            "self_meld_count": self.self_meld_count,
        }


@dataclass(frozen=True)
class TrackingUpdate:
    observation: FrameObservation
    stable: bool
    committed: bool
    state: MahjongState | None
    events: tuple[GameEvent, ...] = ()
    validation_issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.observation.frame_index,
            "timestamp": self.observation.timestamp,
            "stable": self.stable,
            "committed": self.committed,
            "events": [event.to_dict() for event in self.events],
            "validation_issues": self.validation_issues,
            "state": self.state.to_dict() if self.state is not None else None,
        }
