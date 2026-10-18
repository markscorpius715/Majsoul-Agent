from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any

from .tile_inventory import MeldKnowledge


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class TileView:
    tile: str
    box: Box


@dataclass(frozen=True)
class MeldView:
    tiles: tuple[str | None, ...]
    opened: bool | None = True
    kind: str = "unknown"
    face_down: tuple[bool, ...] = ()


@dataclass
class Observation:
    captured_at: float
    window: tuple[int, int, int, int]
    hand: tuple[TileView, ...] = ()  # Draw tile is included once, at draw_index.
    draw_index: int | None = None
    rivers: tuple[tuple[str, ...], ...] = ((), (), ())
    melds: tuple[tuple[MeldView, ...], ...] = ((), (), ())
    nuki: tuple[int | None, ...] = (0, 0, 0)
    dora_indicators: tuple[str, ...] = ()
    riichi: tuple[bool | None, ...] = (False, False, False)
    buttons: dict[str, Box] = field(default_factory=dict)
    phase: str = "unknown"
    round_id: str | None = None
    seat_wind: str | None = None
    round_wind: str | None = None
    issues: list[str] = field(default_factory=list)
    # Logical player indices after bridge ordering. Missing fields contain only
    # currently visible evidence, never zero/empty assertions about hidden UI.
    unavailable_fields: tuple[str, ...] = ()
    own_meld_count: int | None = None
    riichi_discard_indices: tuple[int, ...] | None = None
    remaining_tiles: int | None = None
    honba: int | None = None
    riichi_sticks: int | None = None
    # In a call window: the opponent discard that ron/pon/kan would take.
    call_tile: str | None = None
    call_from: int | None = None

    @property
    def meld_count(self) -> int:
        return len(self.melds[0]) if self.own_meld_count is None else self.own_meld_count

    def semantic(self) -> dict[str, Any]:
        constraints = []
        for player, groups in enumerate(self.melds):
            for index, meld in enumerate(groups):
                constraint = MeldKnowledge(meld.tiles, meld.kind, meld.opened, meld.face_down).constraint()
                if constraint is not None:
                    constraints.append({"player_index": player, "meld_index": index, **constraint})
        return {
            "hand": [t.tile for t in self.hand], "draw_index": self.draw_index,
            "rivers": self.rivers, "melds": [[asdict(m) for m in ms] for ms in self.melds],
            "nuki": self.nuki, "dora_indicators": self.dora_indicators,
            "riichi": self.riichi, "buttons": sorted(self.buttons), "phase": self.phase,
            "round_id": self.round_id, "seat_wind": self.seat_wind, "round_wind": self.round_wind,
            "visible_meld_constraints": constraints,
            "unavailable_fields": self.unavailable_fields,
            "own_meld_count": self.meld_count,
            "riichi_discard_indices": self.riichi_discard_indices,
            "call_tile": self.call_tile, "call_from": self.call_from,
            "remaining_tiles": self.remaining_tiles, "honba": self.honba,
            "riichi_sticks": self.riichi_sticks,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.semantic(), sort_keys=True)
        return sha256(payload.encode()).hexdigest()


@dataclass
class State:
    observation: Observation
    stable: bool
    discard_history: tuple[tuple[str, ...], ...]
    temporary_furiten: bool | None = None
    riichi_furiten: bool | None = None
    history_complete: bool = False
    kuikae_tile: str | None = None
    replacement_draw: bool | None = None
    context_sources: dict[str, Any] = field(default_factory=dict)
    rules_context: dict[str, Any] = field(default_factory=dict)

    def for_model(self) -> dict[str, Any]:
        return {
            **self.observation.semantic(),
            "discard_history": self.discard_history,
            "temporary_furiten": self.temporary_furiten,
            "riichi_furiten": self.riichi_furiten,
            "history_complete": self.history_complete,
            "turn_lower_bound": len(self.discard_history[0]),
            "replacement_draw": self.replacement_draw,
            "kuikae_tile": self.kuikae_tile,
            "context_sources": self.context_sources,
            "rules_context": self.rules_context,
            "hidden_opponent_hands": "unknown",
            "visible_tile_inventory_complete": not bool(self.observation.unavailable_fields),
        }


@dataclass(frozen=True)
class Action:
    id: str
    kind: str
    tile: str | None = None
    hand_index: int | None = None
    button: str | None = None
    features: dict[str, Any] = field(default_factory=dict)

    def for_model(self) -> dict[str, Any]:
        return asdict(self)
