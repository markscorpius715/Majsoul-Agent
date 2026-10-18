"""Bridge tracked MahjongState plus frame observation into the legacy domain.State."""

from __future__ import annotations

from ..domain import Box, MeldView, Observation, State, TileView
from ..perception.schemas import VisibleTilesReading
from ..state.schemas import FrameObservation, GameEventType, MahjongState, TrackingPhase
from ..tile_inventory import MeldKnowledge

LOGICAL_BUTTONS = frozenset(
    {"ron", "pon", "pass", "tsumo", "nuki", "riichi", "abort", "kan"})

# The "he" (和) template is phase-ambiguous and resolved against the table phase.
_HE_RESOLUTION = {"discard": "tsumo", "response": "ron"}


class BridgeError(RuntimeError):
    pass


def normalize_buttons(raw: dict[str, Box], phase: str) -> dict[str, Box]:
    result = {label: box for label, box in raw.items() if label in LOGICAL_BUTTONS}
    he = raw.get("he")
    resolved = _HE_RESOLUTION.get(phase)
    if he is not None and resolved is not None and resolved not in result:
        result[resolved] = he
    return result


def infer_phase(observation: FrameObservation, state: MahjongState, *,
                pending_riichi: bool) -> str:
    if state.phase in (TrackingPhase.ROUND_END, TrackingPhase.GAME_END):
        return "result"
    if state.phase in (TrackingPhase.WAIT_GAME, TrackingPhase.RESYNC):
        return "unknown"
    if state.self_meld_count is None:
        return "unknown"
    expected = 14 - 3 * state.self_meld_count
    if pending_riichi and len(state.hand) == expected:
        return "riichi_select" if observation.riichi_discard_indices is not None else "unknown"
    if observation.action_buttons:
        return "discard" if len(state.hand) == expected else "response"
    if state.current_actor == state.self_slot and len(state.hand) == expected:
        return "discard"
    return "waiting"


def situational_context(state: MahjongState, buttons: dict[str, Box]) -> dict:
    """Keep unobserved response history unknown; derive only table-proved context.

    A lit ron button is client evidence of eligibility at this exact window.
    Table events alone cannot prove that no earlier ron opportunity was passed.
    """
    ron_available = "ron" in buttons
    riichi = state.player(state.self_slot).riichi
    temporary = False if ron_available else None
    riichi_furiten = False if ron_available or riichi is False else None
    sources = {
        "temporary_furiten": "enabled_client_ron" if ron_available else "unknown_response_history",
        "riichi_furiten": ("enabled_client_ron" if ron_available else
                           "not_in_riichi" if riichi is False else "unknown_response_history"),
        "replacement_draw": "unknown_draw_history",
        "kuikae_tile": "unknown_call_history",
    }
    replacement = None
    kuikae = None
    pending_replacement = None
    for event in state.history:
        if event.type in (GameEventType.ROUND_START, GameEventType.RESYNC):
            pending_replacement = None
            replacement = False if (event.type == GameEventType.ROUND_START and
                                     event.certainty == "observed") else None
            kuikae = None
            sources["replacement_draw"] = event.type.value
            sources["kuikae_tile"] = event.type.value
            continue
        if event.player != state.self_slot:
            continue
        if event.type in (GameEventType.KAN, GameEventType.KITA):
            pending_replacement = event.type.value
            replacement = None  # Replacement is pending, not yet observed.
            kuikae = None
            sources["replacement_draw"] = "pending_" + event.type.value
        elif event.type == GameEventType.DRAW:
            if event.details.get("relative_order") == "ambiguous_with_discard":
                replacement = None
                sources["replacement_draw"] = "ambiguous_draw_order"
            else:
                # Only a complete per-hand history establishes ordinary draws.
                replacement = bool(pending_replacement) if state.history_complete else None
                sources["replacement_draw"] = pending_replacement or "ordinary_draw_history"
            pending_replacement = None
            kuikae = None
        elif event.type == GameEventType.PON:
            replacement = False
            known = event.details.get("tiles", ())
            kuikae = event.tile or (known[0] if known and all(known) else None)
            sources["replacement_draw"] = "PON"
            sources["kuikae_tile"] = "PON"
        elif event.type == GameEventType.DISCARD:
            pending_replacement = None
            replacement = False
            kuikae = None
            sources["replacement_draw"] = "DISCARD"
            sources["kuikae_tile"] = "DISCARD"
    return dict(temporary_furiten=temporary, riichi_furiten=riichi_furiten,
                replacement_draw=replacement, kuikae_tile=kuikae,
                context_sources=sources)


def build_domain_state(state: MahjongState, observation: FrameObservation,
                       visible: VisibleTilesReading, buttons: dict[str, Box], *,
                       pending_riichi: bool,
                       window: tuple[int, int, int, int]) -> State:
    if not state.decision_ready:
        detail = (f"; unknown tiles or metadata: {state.blocking_uncertain_fields}"
                  if state.blocking_uncertain_fields else "")
        raise BridgeError("State is not decision-ready" + detail)
    if state.observation_fingerprint is not None and state.observation_fingerprint != observation.fingerprint:
        raise BridgeError("Current observation differs from the validated state")
    if len(visible.hand.tiles) != len(state.hand):
        raise BridgeError(
            f"Visible hand has {len(visible.hand.tiles)} tiles, "
            f"state has {len(state.hand)}")
    for index, (item, tile) in enumerate(zip(visible.hand.tiles, state.hand)):
        if item.prediction.tile != tile:
            raise BridgeError(
                f"Visible hand mismatch at index {index}: "
                f"{item.prediction.tile} != {tile}")
    for player in state.players:
        for meld in player.melds:
            knowledge = MeldKnowledge(meld.tiles, meld.kind, meld.opened, meld.face_down)
            if not knowledge.complete and not (player.slot != state.self_slot and
                                                knowledge.bounded_concealed_quad):
                raise BridgeError(
                    f"Player {player.slot} has a meld with unknown tiles")

    ordered = (state.player(state.self_slot),) + tuple(
        state.player(slot) for slot in state.active_slots if slot != state.self_slot)
    phase = infer_phase(observation, state, pending_riichi=pending_riichi)
    call_tile = call_from = None
    last = getattr(state, 'last_discard', None)
    if phase == 'response' and last and last[0] != state.self_slot and last[1]:
        slots = [player.slot for player in ordered]
        if last[0] in slots:
            call_tile, call_from = last[1], slots.index(last[0])
    if phase == 'response' and call_tile is None and 'pon' in (observation.action_buttons or ()):
        # History gap: a pon needs two copies in hand, so the newest opponent
        # river tile we hold twice is the target when exactly one qualifies.
        from ..tiles import tile_index
        held = [tile_index(t) for t in state.hand]
        options = [(index, player.river[-1]) for index, player in enumerate(ordered)
                   if index and player.river and held.count(tile_index(player.river[-1])) >= 2]
        if len(options) == 1:
            call_from, call_tile = options[0]
    obs = Observation(
        captured_at=observation.timestamp,
        window=window,
        hand=tuple(TileView(tile, item.detection.box)
                   for tile, item in zip(state.hand, visible.hand.tiles)),
        draw_index=observation.drawn_index,
        rivers=tuple(tuple(player.river) for player in ordered),
        melds=tuple(
            tuple(MeldView(meld.tiles, opened=meld.opened, kind=meld.kind,
                           face_down=meld.face_down)
                  for meld in player.melds)
            for player in ordered),
        nuki=tuple(player.kita_count for player in ordered),
        dora_indicators=state.dora_indicators,
        riichi=tuple(player.riichi for player in ordered),
        buttons=normalize_buttons(buttons, phase),
        phase=phase,
        round_id=state.round_id,
        seat_wind=state.self_seat,
        round_wind=state.round_wind,
        issues=list(observation.actionable_issues),
        unavailable_fields=tuple(
            f"{kind}_{index}" for index, player in enumerate(ordered)
            for kind, prefix in (("rivers", "river"), ("melds", "melds"), ("nuki", "kita"))
            if f"{prefix}_{player.slot}" in state.unavailable_regions) + tuple(
                'dora_indicators' if name == 'dora' else name
                for name in state.unavailable_regions
                if name in ('remaining_tiles', 'honba', 'riichi_sticks', 'dora')),
        own_meld_count=state.self_meld_count,
        riichi_discard_indices=observation.riichi_discard_indices,
        remaining_tiles=state.remaining_tiles, honba=state.honba,
        riichi_sticks=state.riichi_sticks,
        call_tile=call_tile, call_from=call_from,
    )
    return State(
        observation=obs,
        stable=True,
        discard_history=tuple(tuple(player.discard_history) for player in ordered),
        history_complete=state.history_complete,
        **situational_context(state, obs.buttons),
    )
