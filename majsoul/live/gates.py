"""Local input checks after a validated decision snapshot has been captured."""

from dataclasses import replace

from ..domain import TileView
from .bridge import normalize_buttons


def input_issues(observation):
    # Public table fields are needed for the initial decision snapshot, but
    # their animation must not invalidate an already submitted local action.
    return tuple(issue for issue in observation.actionable_issues
                 if issue.startswith(("hand", "action_buttons", "selection")))


def input_signature(observation):
    return (observation.active_slots,
            tuple((t.tile, t.face_down) for t in observation.hand),
            observation.drawn_index, tuple(sorted(observation.action_buttons or ())),
            observation.riichi_discard_indices)


def same_input_window(before, after):
    # Missing public labels during animation do not invalidate a frozen frame;
    # a positively observed different round still closes the opportunity.
    return (input_signature(before) == input_signature(after) and all(
        getattr(before, name) is None or getattr(after, name) is None or
        getattr(before, name) == getattr(after, name)
        for name in ('round_wind', 'hand_number', 'honba')))


def fresh_input_domain(domain, observation, visible, buttons, window):
    """Keep the model's table snapshot; refresh only evidence used for input."""
    if len(visible.hand.tiles) != len(observation.hand) or any(
            item.prediction.tile != tile.tile
            for item, tile in zip(visible.hand.tiles, observation.hand)):
        raise ValueError("Fresh hand geometry does not match recognized hand")
    return replace(domain, observation=replace(
        domain.observation, captured_at=observation.timestamp, window=window,
        hand=tuple(TileView(tile.tile, item.detection.box)
                   for tile, item in zip(observation.hand, visible.hand.tiles)),
        draw_index=observation.drawn_index,
        buttons=normalize_buttons(buttons, domain.observation.phase),
        issues=list(input_issues(observation)),
        riichi_discard_indices=observation.riichi_discard_indices))
