from difflib import SequenceMatcher

from .domain import Action, Observation, State
from .tiles import tile_index


class StateTracker:
    """Commit only repeated observations; keep history separate from visible rivers."""

    def __init__(self, stable_frames: int = 2):
        if stable_frames < 2:
            raise ValueError("At least two stable frames are required")
        self.stable_frames = stable_frames
        self.pending: str | None = None
        self.repeats = 0
        self.last: Observation | None = None
        self.history: list[list[str]] = [[], [], []]
        self.temporary_furiten = False
        self.riichi_furiten = False
        self.history_complete = False
        self.kuikae_tile: str | None = None
        self.replacement_draw = False
        self.last_time = float("-inf")
        self.passed_ron = False
        self.pending_replacement = False

    def record_action(self, action: Action, state: State) -> None:
        if action.kind == "pass" and "ron" in state.observation.buttons:
            self.temporary_furiten = True
            self.riichi_furiten |= state.observation.riichi[0] is not False
            self.passed_ron = True

    def update(self, obs: Observation) -> State:
        if obs.captured_at <= self.last_time:
            raise ValueError("Observations must have strictly increasing capture times")
        self.last_time = obs.captured_at
        if obs.phase == "result" and self.last is not None:
            # The results overlay hides the table's wind/round widgets.
            obs.round_id = obs.round_id or self.last.round_id
            obs.seat_wind = obs.seat_wind or self.last.seat_wind
            obs.round_wind = obs.round_wind or self.last.round_wind
        key = obs.fingerprint
        self.repeats = self.repeats + 1 if key == self.pending and not obs.issues else 1
        self.pending = key if not obs.issues else None
        stable = not obs.issues and self.repeats >= self.stable_frames
        if stable:
            self._commit(obs)
        return State(obs, stable, tuple(tuple(r) for r in self.history),
                     self.temporary_furiten, self.riichi_furiten, self.history_complete,
                     self.kuikae_tile, self.replacement_draw)

    def _commit(self, obs: Observation) -> None:
        old = self.last
        new_round = old is None or obs.round_id != old.round_id or (old.phase == "result" and obs.phase != "result")
        if new_round:
            self.history = [list(r) for r in obs.rivers]
            self.temporary_furiten = self.riichi_furiten = False
            self.kuikae_tile = None
            self.replacement_draw = False
            self.passed_ron = False
            self.pending_replacement = False
            self.history_complete = (not obs.unavailable_fields and not any(obs.rivers) and
                                     not any(obs.melds) and all(n == 0 for n in obs.nuki))
        elif old.fingerprint != obs.fingerprint:
            for player, (before, after) in enumerate(zip(old.rivers, obs.rivers)):
                for tag, _, _, start, end in SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
                    if tag in ("insert", "replace"):
                        self.history[player].extend(after[start:end])
                    if tag == "replace":
                        self.history_complete = False
            gained_meld = len(obs.melds[0]) > len(old.melds[0])
            if gained_meld:
                added = [m for m in obs.melds[0] if m not in old.melds[0]]
                self.kuikae_tile = added[0].tiles[0] if len(added) == 1 and len(added[0].tiles) == 3 else None
            elif len(obs.rivers[0]) > len(old.rivers[0]):
                self.kuikae_tile = None
            gained_nuki = (obs.nuki[0] is not None and old.nuki[0] is not None and
                           obs.nuki[0] > old.nuki[0])
            gained_kan = sum(len(m.tiles) == 4 for m in obs.melds[0]) > sum(len(m.tiles) == 4 for m in old.melds[0])
            self.pending_replacement |= gained_nuki or gained_kan
            # Own-draw reset needs a transition or a changed drawn tile; button
            # flicker in the same turn is not evidence of a new draw.
            new_draw = obs.draw_index is not None and (
                old.draw_index is None or old.phase == "waiting" or gained_nuki or gained_kan
                or tuple(t.tile for t in old.hand) != tuple(t.tile for t in obs.hand))
            if new_draw:
                self.temporary_furiten = False
                self.replacement_draw = self.pending_replacement
                self.pending_replacement = False
            elif len(obs.rivers[0]) > len(old.rivers[0]):
                self.replacement_draw = False
            if "ron" in old.buttons and "ron" not in obs.buttons and obs.phase != "result" and not new_draw:
                self.temporary_furiten = True
                self.riichi_furiten |= old.riichi[0] is not False
        self.last = obs
