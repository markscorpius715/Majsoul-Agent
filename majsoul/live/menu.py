"""Menu waypoint navigation: ranked lobby -> bronze hall -> sanma east -> match."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable

import cv2
import yaml

from ..window import WindowError
from .capture import FocusUnavailable

if TYPE_CHECKING:
    from .capture import LiveWindow


class NavigationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MenuStep:
    name: str
    click: tuple[float, float] | None
    settle_seconds: float
    mouse_press_seconds: float = 0.05

    def __post_init__(self):
        if not self.name:
            raise ValueError("Menu step name must be non-empty")
        if self.click is not None:
            x, y = self.click
            if not (0 < x < 1 and 0 < y < 1):
                raise ValueError(f"Menu step {self.name!r} click must be normalized to (0, 1)")
        if not math.isfinite(self.settle_seconds) or self.settle_seconds < 0:
            raise ValueError(f"Menu step {self.name!r} settle_seconds must be finite and non-negative")
        if (isinstance(self.mouse_press_seconds, bool) or
                not math.isfinite(self.mouse_press_seconds) or
                not 0 < self.mouse_press_seconds <= 0.2):
            raise ValueError("Menu mouse_press_seconds must be in (0, 0.2]")


@dataclass(frozen=True)
class MenuConfig:
    steps: tuple[MenuStep, ...]
    table_timeout_seconds: float = 120.0
    focus_attempts: int = 3
    focus_retry_seconds: float = .25

    def __post_init__(self):
        if type(self.focus_attempts) is not int or self.focus_attempts<1:
            raise ValueError('Menu focus_attempts must be a positive integer')
        if (type(self.focus_retry_seconds) not in (int,float) or
                not math.isfinite(self.focus_retry_seconds) or self.focus_retry_seconds<=0):
            raise ValueError('Menu focus_retry_seconds must be finite and positive')

    @property
    def ready(self) -> bool:
        return all(step.click is not None for step in self.steps)

    @classmethod
    def load(cls, path: str | Path) -> "MenuConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Malformed menu configuration")
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("Menu configuration requires at least one step")
        try:
            steps = tuple(
                MenuStep(
                    name=str(item["name"]),
                    click=None if item.get("click") is None else tuple(float(v) for v in item["click"]),
                    settle_seconds=float(item.get("settle_seconds", 0.0)),
                    mouse_press_seconds=item.get("mouse_press_seconds", 0.05),
                )
                for item in raw_steps
            )
            timeout = float(data.get("table_timeout_seconds", 120.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed menu step") from exc
        names = [step.name for step in steps]
        if len(set(names)) != len(names):
            raise ValueError("Menu step names must be unique")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("table_timeout_seconds must be finite and positive")
        return cls(steps=steps, table_timeout_seconds=timeout,
                   focus_attempts=data.get('focus_attempts',3),
                   focus_retry_seconds=data.get('focus_retry_seconds',.25))


class MenuNavigator:
    def __init__(self, window: "LiveWindow", config: MenuConfig, *, sleep: Callable[[float], None] = time.sleep,
                 poll_seconds: float | None = None, now: Callable[[], float] = time.monotonic):
        self.window = window
        self.config = config
        self.sleep = sleep
        if poll_seconds is not None and (not math.isfinite(poll_seconds) or poll_seconds <= 0):
            raise ValueError("Menu poll_seconds must be finite and positive")
        self.poll_seconds = poll_seconds
        self.now = now

    def _settle(self, seconds: float) -> None:
        if self.poll_seconds is None:
            self.sleep(seconds)
            return
        deadline = self.now() + seconds
        while self.now() < deadline:
            # Actual captures prove responsiveness and honour the stop key;
            # a timer thread emitting heartbeats could hide a blocked capture.
            self.window.capture()
            remaining = deadline - self.now()
            if remaining > 0:
                self.sleep(min(self.poll_seconds, remaining))

    def run(self) -> None:
        if not self.config.ready:
            raise NavigationError("Menu coordinates are not calibrated; run calibrate-menu and fill in the click positions first")
        for step in self.config.steps:
            try:
                for attempt in range(self.config.focus_attempts):
                    frame = self.window.capture()
                    _, _, width, height = frame.window
                    point = (round(step.click[0] * width), round(step.click[1] * height))
                    try:
                        self.window.click(point, expected=frame, hold_seconds=step.mouse_press_seconds)
                        break
                    except FocusUnavailable:
                        # This exception guarantees no input was sent. Other
                        # failures have uncertain outcomes and are not retried.
                        if attempt+1>=self.config.focus_attempts:raise
                        self._settle(self.config.focus_retry_seconds)
                self._settle(step.settle_seconds)
            except WindowError as exc:
                raise NavigationError(f"Menu step {step.name!r} failed: {exc}") from exc


def calibrate_menu(window: "LiveWindow", output_dir: str | Path, *,
                   prompt: Callable[[str], str] = input,
                   sleep: Callable[[float], None] = time.sleep) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print("manually operate the Mahjong Soul menu to the next page then press Enter to capture, press Enter directly to start, type q and press Enter to end")
    saved: list[Path] = []
    while True:
        answer = prompt("> ").strip().lower()
        if answer == "q":
            break
        frame = window.capture()
        path = output / f"step-{len(saved) + 1:02d}.png"
        cv2.imwrite(str(path), frame.image)
        saved.append(path)
        print(f"saved {path}")
    return saved
