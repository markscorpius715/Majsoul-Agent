"""Shared visual lifecycle and verified client auto-play settings for Agent."""

from pathlib import Path

import cv2

from ..calibration.frames import write_image
from .collector import CollectConfig, CollectError, CollectionUI, TogglePanel, menu_score
from .lifecycle import SessionLifecycle
from .results import ResultDelivery
from .capture import FocusUnavailable


class AgentSession:
    MODAL_ATTEMPTS = 3

    def __init__(self, window, config: CollectConfig, *, execute, capture, sleep,
                 log, directory: Path, notify, modals=None):
        self.window, self.config, self.execute = window, config, execute
        self.capture, self.sleep, self.log = capture, sleep, log
        self.directory, self.notify = directory, notify
        self.ui = CollectionUI(config)
        self.lobby_template = cv2.imread(str(config.menu_template))
        if self.lobby_template is None:
            raise CollectError("Cannot read lobby template")
        self.lifecycle = SessionLifecycle(config.table_ready_seconds)
        self.panel = TogglePanel(window, config, capture=capture, sleep=sleep)
        self.result_image = None
        self.result_delivery = ResultDelivery(directory, mode="agent", match_index=1,
                                              notify=notify, log=log)
        # User contract (2026-10-10): Agent enters with client auto-play OFF.
        # Opening the drawer each hand wastes the live decision window.
        self.options_verified = True
        self.attempts = 0
        self.retry_at = float('-inf')
        self.next_confirm = float('-inf')
        self.confirm_focus_failures = 0
        # Reviewed in-game popups (text AND button templates). The Agent
        # dismisses recoverable ones itself; no other process plays for it.
        self.modals = modals
        self.modal = None
        self.modal_attempts = 0
        self.next_modal = float('-inf')

    def _modal(self, frame, now):
        """Handle a reviewed popup; True while one is visible."""
        name = self.modals.read(frame.image) if self.modals is not None else None
        if name != self.modal:
            self.modal, self.modal_attempts = name, 0
            if name:
                self.log('modal_observed', name=name)
                if not self.modals.recoverable(name):
                    self.notify(f'An in-game popup needs the user: {name}; the Agent will not click it.')
        if not name:
            return False
        if (not self.execute or not self.modals.recoverable(name) or now < self.next_modal
                or self.modal_attempts >= self.MODAL_ATTEMPTS):
            return True
        # Reobserve before input: only the same reviewed popup may be clicked.
        fresh = self.capture()
        if fresh.window != frame.window or self.modals.read(fresh.image) != name:
            return True
        self.next_modal = now+self.config.confirm_click_seconds
        try:
            self.window.click(self.modals.point(name, fresh.image.shape[1::-1]), expected=fresh)
        except FocusUnavailable as exc:
            self.log('modal_click_deferred', name=name, error=str(exc))
            return True
        self.modal_attempts += 1
        self.log('modal_confirm_clicked', name=name, attempt=self.modal_attempts)
        if self.modal_attempts >= self.MODAL_ATTEMPTS:
            self.notify(f'The popup {name} did not close after {self.MODAL_ATTEMPTS} clicks; '
                        'the Agent stops clicking it and keeps watching.')
        return True

    def classify(self, frame):
        lobby = menu_score(frame.image, self.lobby_template, self.config.menu_region,
                           self.config.calibration_size) >= self.config.menu_threshold
        kind = self.ui.confirm(frame.image)
        table = not lobby and not kind and self.ui.table_visible(frame.image)
        return lobby, kind, table

    def observe(self, frame, now):
        """Return (suppress decisions, lifecycle events), with input gated by execute."""
        if self._modal(frame, now):
            return True, []
        lobby, kind, table = self.classify(frame)
        events = self.lifecycle.update(confirm=bool(kind), table=table, lobby=lobby, now=now)
        for event in events:
            self.log(event, round_index=self.lifecycle.round_index)
        if self.result_image is None and self.ui.match_result(frame.image):
            def save(result_frame):
                path = self.directory / 'match_result.png'
                write_image(path, result_frame.image)
                return path
            self.result_image = self.result_delivery.capture(frame, save)
        if 'match_complete' in events:
            write_image(self.directory / 'lobby_verified.png', frame.image)
            self.result_delivery.lobby_verified(frame)
            self.notify('Agent match ended, return to lobby stably recognized'
                        + ('; final result screenshot has been submitted separately' if self.result_image else '; final result screenshot not obtained'))
        if kind:
            held = now - self.lifecycle.confirm_since
            if (self.execute and now >= self.next_confirm and
                    ((kind == 'gold' and held >= self.config.table_ready_seconds)
                     or held >= self.config.confirm_hold_seconds)):
                fresh = self.capture()
                if fresh.window == frame.window and self.ui.confirm(fresh.image) == kind:
                    region = self.config.confirm_region
                    point = ((region[0]+region[2])/2 * fresh.window[2],
                             (region[1]+region[3])/2 * fresh.window[3])
                    try:
                        self.window.click(point, expected=fresh)
                    except FocusUnavailable as exc:
                        # This exception guarantees no input was sent. Later
                        # attempts must reobserve the same result UI first.
                        self.confirm_focus_failures += 1
                        write_image(self.directory / 'confirm_focus_unavailable.png', fresh.image)
                        self.log('confirm_focus_deferred', attempts=self.confirm_focus_failures,
                                 error=str(exc))
                        self.next_confirm = now + self.config.confirm_click_seconds
                        if self.confirm_focus_failures >= self.config.confirm_focus_attempts:
                            raise
                        return True, events
                    self.confirm_focus_failures = 0
                    self.next_confirm = now + self.config.confirm_click_seconds
                    self.log('end_confirm_clicked', kind=kind)
            return True, events
        self.confirm_focus_failures = 0
        if 'table_ready' in events:
            self.attempts = 0
            self.retry_at = now
            self.log('agent_auto_options_assumed_off', source='user_confirmed_default')
        # Menu, matching and loading pages are not tile observations. Unknown
        # pages must keep the cheap UI polling path rather than feed artwork
        # into CNN/OCR and block capture while trying to resolve a seat layout.
        return bool(not table or self.lifecycle.pending), events
