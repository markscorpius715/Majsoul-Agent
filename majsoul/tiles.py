from collections import Counter
import re


def tile_index(tile: str) -> int:
    if not isinstance(tile, str) or not re.fullmatch(r"[0-9][mpsz]", tile):
        raise ValueError(f"Invalid tile: {tile!r}")
    rank, suit = int(tile[0]), tile[1]
    if suit == "z" and not 1 <= rank <= 7:
        raise ValueError(f"Invalid honor: {tile}")
    if rank == 0:
        if suit == "z":
            raise ValueError("Honors cannot be red")
        rank = 5
    return "mpsz".index(suit) * 9 + rank - 1


def tile_name(index: int) -> str:
    if not 0 <= index < 34:
        raise ValueError("Tile index must be 0..33")
    return f"{index % 9 + 1}{'mpsz'[index // 9]}"


def counts34(tiles: list[str] | tuple[str, ...]) -> list[int]:
    result = [0] * 34
    for tile in tiles:
        result[tile_index(tile)] += 1
    if max(result) > 4:
        raise ValueError("More than four copies of a tile")
    return result


def parse_hand(text: str) -> list[str]:
    groups = re.findall(r"([0-9]+)([mpsz])", text)
    if "".join(digits + suit for digits, suit in groups) != text:
        raise ValueError("Use compact notation, e.g. 19m123456p123s11z")
    tiles = [digit + suit for digits, suit in groups for digit in digits]
    counts34(tiles)
    return tiles


def physical_ids(groups: list[list[str]]) -> list[list[int]]:
    """Allocate distinct 136-format IDs across concealed hand and melds."""
    used: set[int] = set()
    result = []
    for group in groups:
        ids = []
        for tile in group:
            index = tile_index(tile)
            base = 4 * index
            # Reserve copy zero for red fives even if red is encountered later.
            candidates = [base] if tile[0] == "0" else list(range(base + (index in (4, 13, 22)), base + 4))
            available = next((i for i in candidates if i not in used), None)
            if available is None:
                raise ValueError(f"Impossible tile multiplicity/red allocation: {tile}")
            used.add(available)
            ids.append(available)
        result.append(ids)
    return result
