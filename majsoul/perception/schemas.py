from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..domain import Box


@dataclass(eq=False)
class SegmentedTile:
    box: Box
    image: np.ndarray = field(repr=False)
    region: str = ""
    row: int = 0
    column: int = 0
    orientation: str = "upright"
    face_down: bool = False


@dataclass(frozen=True)
class SegmentationResult:
    tiles: tuple[SegmentedTile, ...]
    drawn_index: int | None = None
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class TilePrediction:
    tile: str | None
    confidence: float
    distance: float | None
    margin: float | None
    orientation: str
    candidates: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RecognizedTile:
    detection: SegmentedTile
    prediction: TilePrediction


@dataclass(frozen=True)
class TileSequenceReading:
    region: str
    tiles: tuple[RecognizedTile, ...]
    drawn_index: int | None = None
    issues: tuple[str, ...] = ()

    @property
    def labels(self) -> tuple[str | None, ...]:
        return tuple(item.prediction.tile for item in self.tiles)

    @property
    def complete(self) -> bool:
        return not self.issues and all(label is not None for label in self.labels)


@dataclass(frozen=True)
class MeldReading:
    region: str
    groups: tuple[tuple[RecognizedTile, ...], ...]
    issues: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()
    opened: tuple[bool | None, ...] = ()

    @property
    def labels(self) -> tuple[tuple[str | None, ...], ...]:
        return tuple(tuple(tile.prediction.tile for tile in group) for group in self.groups)

    @property
    def complete(self) -> bool:
        return (not self.issues and
                all(tile.prediction.tile is not None or
                    (tile.detection.face_down and self.kinds and self.kinds[index] == 'ankan')
                    for index, group in enumerate(self.groups) for tile in group))


@dataclass(frozen=True)
class VisibleTilesReading:
    active_slots: tuple[str, ...]
    hand: TileSequenceReading
    rivers: dict[str, TileSequenceReading]
    melds: dict[str, MeldReading]
    dora: TileSequenceReading
    kita: dict[str, TileSequenceReading]
    # Screen-space UI evidence, retained separately from tile predictions.
    occlusion_boxes: tuple[Box, ...] = ()

    @property
    def issues(self) -> tuple[str, ...]:
        entries = [("hand", self.hand), ("dora", self.dora)]
        entries.extend((f"river_{slot}", reading) for slot, reading in self.rivers.items())
        entries.extend((f"melds_{slot}", reading) for slot, reading in self.melds.items())
        entries.extend((f"kita_{slot}", reading) for slot, reading in self.kita.items())
        return tuple(f"{name}: {issue}" for name, reading in entries
                     for issue in reading.issues)

    @property
    def complete(self) -> bool:
        readings = [self.hand, self.dora, *self.rivers.values(),
                    *self.melds.values(), *self.kita.values()]
        return all(reading.complete for reading in readings)


@dataclass(frozen=True)
class GlyphPrediction:
    label: str | None
    confidence: float
    distance: float | None
    margin: float | None
    box: Box


@dataclass(eq=False)
class SegmentedGlyph:
    box: Box
    mask: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class NumericReading:
    region: str
    value: int | None
    glyphs: tuple[GlyphPrediction, ...]
    issues: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.value is not None and not self.issues


@dataclass(frozen=True)
class RoundReading:
    round_wind: str | None
    hand_number: int | None
    glyphs: tuple[GlyphPrediction, ...]
    issues: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.round_wind is not None and self.hand_number is not None and not self.issues


@dataclass(frozen=True)
class SeatReading:
    slot: str
    wind: str | None
    dealer: bool | None
    glyph: GlyphPrediction | None
    red_fill_ratio: float
    issues: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.wind is not None and self.dealer is not None and not self.issues


@dataclass(frozen=True)
class GameMetadataReading:
    active_slots: tuple[str, ...]
    round: RoundReading
    remaining_tiles: NumericReading
    honba: NumericReading
    riichi_sticks: NumericReading
    scores: dict[str, NumericReading]
    seats: dict[str, SeatReading]
    issues: tuple[str, ...] = ()
    # False when score reading is disabled for this pipeline: score values are
    # then None (unavailable), never fabricated, and excluded from `complete`.
    scores_available: bool = True

    @property
    def round_id(self) -> str | None:
        if not self.round.complete or not self.honba.complete:
            return None
        return f"{self.round.round_wind}:{self.round.hand_number}:{self.honba.value}"

    @property
    def complete(self) -> bool:
        readings = [self.round, self.remaining_tiles, self.honba, self.riichi_sticks,
                    *self.seats.values()]
        if self.scores_available:
            readings.extend(self.scores.values())
        return not self.issues and all(reading.complete for reading in readings)
