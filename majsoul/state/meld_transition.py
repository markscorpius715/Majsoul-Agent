"""Physical meld/hand conservation shared by tracking and input verification."""
from collections import Counter

from ..tiles import tile_index


def meld_key(meld):
    return tuple(sorted((t.tile for t in meld.tiles), key=lambda t: t or ""))


def unmatched_melds(before, after):
    remaining = list(after)
    missing = []
    for old in before:
        match = next((m for m in remaining if meld_key(m) == meld_key(old)), None)
        if match is None:
            missing.append(old)
        else:
            remaining.remove(match)
    return missing, remaining


def prove_pon_transition(before, after, self_slot, discards=()):
    """Exactly two own tiles enter one new pon; no normal draw is required.

    An observed subsequent own discard can be included by the event tracker.
    The called tile's physical identity is deduced; its source stays unknown.
    """
    old, new = before.player(self_slot), after.player(self_slot)
    missing, added = unmatched_melds(old.melds, new.melds)
    if missing or len(added) != 1:
        return None
    group = added[0]
    if group.kind != "pon" or group.opened is not True:
        return None
    tiles = [t.tile for t in group.tiles]
    if any(t.tile is None or t.face_down for t in (*before.hand, *after.hand, *group.tiles)):
        return None
    prior = Counter(t.tile for t in before.hand)
    retained = Counter(t.tile for t in after.hand) + Counter(discards)
    consumed = prior - retained
    if retained - prior or sum(consumed.values()) != 2:
        return None
    try:
        if len({tile_index(t) for t in [*consumed, *tiles]}) != 1:
            return None
    except ValueError:
        return None
    external = Counter(tiles) - consumed
    if consumed - Counter(tiles) or sum(external.values()) != 1:
        return None
    return {"consumed_tiles": list(consumed.elements()),
            "called_tile": next(external.elements()), "source_slot": None}
