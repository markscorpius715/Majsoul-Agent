"""Visible physical labels and structural constraints are different evidence.

A validated concealed quad with two identical exposed faces constrains four
copies of one tile family. It does not reveal which hidden position is red.
Never fill unknown physical labels merely to count the family correctly.
"""

from collections import Counter
from dataclasses import dataclass

from .tiles import tile_index, tile_name


@dataclass(frozen=True)
class MeldKnowledge:
    tiles: tuple[str | None, ...]
    kind: str
    opened: bool | None
    face_down: tuple[bool, ...] = ()

    @property
    def complete(self):
        return all(tile is not None for tile in self.tiles)

    @property
    def bounded_concealed_quad(self):
        """The supported visible ankan grammar, not a generic unknown exemption."""
        if (self.kind != "ankan" or self.opened is not False or len(self.tiles) != 4 or
                self.face_down != (True, False, False, True) or
                any(type(flag) is not bool for flag in self.face_down) or
                self.tiles[0] is not None or self.tiles[-1] is not None or
                self.tiles[1] is None or self.tiles[2] is None):
            return False
        try:
            return tile_index(self.tiles[1]) == tile_index(self.tiles[2])
        except ValueError:
            return False

    def constraint(self):
        if not self.bounded_concealed_quad:
            return None
        return {"kind": "ankan", "tile_family": tile_name(tile_index(self.tiles[1])),
                "family_count": len(self.tiles), "unknown_physical_positions": [0, 3],
                "hidden_red_identity": "unknown",
                "evidence": "typed_closed_quad_with_two_matching_exposed_faces"}


@dataclass(frozen=True)
class TileInventory:
    known: tuple[str, ...]
    family_counts: tuple[int, ...]
    unresolved: tuple[int, ...]
    constraints: tuple[dict, ...]

    @classmethod
    def build(cls, known, melds):
        physical = list(known)
        extra = Counter()
        unresolved = []
        constraints = []
        for index, meld in enumerate(melds):
            physical.extend(tile for tile in meld.tiles if tile is not None)
            if meld.complete:
                continue
            constraint = meld.constraint()
            if constraint is None:
                unresolved.append(index)
            else:
                missing = sum(tile is None for tile in meld.tiles)
                extra[tile_index(constraint["tile_family"])] += missing
                constraints.append({"meld_index": index, **constraint})
        counts = [0] * 34
        for tile in physical:
            counts[tile_index(tile)] += 1
        for index, value in extra.items():
            counts[index] += value
        return cls(tuple(physical), tuple(counts), tuple(unresolved), tuple(constraints))

    def validate_family_counts(self):
        if max(self.family_counts, default=0) > 4:
            raise ValueError("More than four copies after structural meld constraints")


def observed_meld_knowledge(meld):
    """Explicit adapter for observation tiles, including their back flags."""
    return MeldKnowledge(tuple(tile.tile for tile in meld.tiles), meld.kind, meld.opened,
                         tuple(tile.face_down for tile in meld.tiles))
