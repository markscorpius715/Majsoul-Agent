"""Current decision inputs with explicitly unavailable public vision fields.

This projection never enters the event tracker. Failed public readings are
unknown, not empty observations and not silently replaced by an old frame.
"""
from dataclasses import replace


OPTIONAL_DECISION_FIELDS = frozenset({'remaining_tiles', 'honba', 'riichi_sticks', 'dora'})


def decision_observation(observation, self_slot="bottom"):
    permitted = {f"{prefix}_{slot}" for slot in observation.active_slots
                 for prefix in ("river", "melds", "kita")
                 if prefix != "melds" or slot != self_slot} | OPTIONAL_DECISION_FIELDS
    failed = {issue.split(":", 1)[0] for issue in observation.issues} & permitted
    for name in ('remaining_tiles', 'honba', 'riichi_sticks'):
        if getattr(observation, name) is None:
            failed.add(name)
    if any(t.tile is None for t in observation.dora_indicators):
        failed.add('dora')
    for player in observation.players:
        if any(t.tile is None for t in player.river):
            failed.add(f"river_{player.slot}")
        if player.kita_count is None and observation.field_available(f"kita_{player.slot}"):
            failed.add(f"kita_{player.slot}")
        # A bounded concealed opponent kan is already a valid partial reading.
        if player.slot != self_slot and any(
                any(t.tile is None for t in m.tiles) and m.kind != "ankan"
                for m in player.melds):
            failed.add(f"melds_{player.slot}")
    failed -= set(observation.unavailable_regions)
    if not failed:
        return observation
    issues = list(observation.issues)
    for name in sorted(failed):
        if not any(issue.startswith(name + ":") for issue in issues):
            issues.append(f"{name}: unrecognized public field")
    players = tuple(replace(
        p, river=() if f"river_{p.slot}" in failed else p.river,
        melds=() if f"melds_{p.slot}" in failed else p.melds,
        kita_count=None if f"kita_{p.slot}" in failed else p.kita_count)
        for p in observation.players)
    counters = {name: None for name in ('remaining_tiles', 'honba', 'riichi_sticks') if name in failed}
    return replace(observation, players=players, issues=tuple(issues), **counters,
                   dora_indicators=() if 'dora' in failed else observation.dora_indicators,
                   unavailable_regions=tuple(sorted(set(observation.unavailable_regions) | failed)),
                   perception_unavailable_regions=tuple(sorted(failed)))
