"""UI lifecycle independent of score OCR, tile recognition and decision calls."""

from dataclasses import dataclass, field


@dataclass
class RoundLifecycle:
    ready_seconds: float
    round_index: int = 0
    pending: bool = True
    previous_confirm: bool = False
    confirm_since: float | None = None
    table_since: float | None = None

    def update(self, *, confirm: bool, table: bool, now: float):
        events = []
        if confirm:
            if not self.previous_confirm:
                events.append("confirm_appeared")
                self.confirm_since = now
            self.pending = True
            self.table_since = None
        else:
            if self.previous_confirm:
                events.append("confirm_disappeared")
            self.confirm_since = None
            if table:
                if self.table_since is None:
                    self.table_since = now
                if self.pending and now - self.table_since >= self.ready_seconds:
                    self.round_index += 1
                    self.pending = False
                    events.append("table_ready")
            else:
                self.table_since = None
        self.previous_confirm = confirm
        return tuple(events)

    def reset_stability(self):
        self.table_since = self.confirm_since = None
        self.previous_confirm = False


@dataclass
class SessionLifecycle:
    ready_seconds: float
    rounds: RoundLifecycle = field(init=False)
    lobby_since: float | None = None
    session_seen: bool = False
    complete: bool = False

    def __post_init__(self):
        self.rounds = RoundLifecycle(self.ready_seconds)

    def update(self, *, confirm: bool, table: bool, lobby: bool, now: float):
        if self.complete:
            return ()
        events = list(self.rounds.update(confirm=confirm, table=table and not lobby,
                                        now=now))
        self.session_seen |= confirm or table
        if lobby:
            if self.lobby_since is None:
                self.lobby_since = now
            if self.session_seen and now - self.lobby_since >= self.ready_seconds:
                self.complete = True
                events.append("match_complete")
        else:
            self.lobby_since = None
        return tuple(events)

    @property
    def round_index(self):
        return self.rounds.round_index

    @property
    def pending(self):
        return self.rounds.pending

    @property
    def confirm_since(self):
        return self.rounds.confirm_since

    def reset_stability(self):
        self.rounds.reset_stability()
        self.lobby_since = None
