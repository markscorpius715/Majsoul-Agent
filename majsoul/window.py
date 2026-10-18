import ctypes
import math
from ctypes import wintypes
from dataclasses import dataclass
import sys
import time


@dataclass
class Frame:
    image: object
    captured_at: float
    window: tuple[int, int, int, int]


class WindowError(RuntimeError):
    pass


def _user32():
    if sys.platform != "win32":
        raise WindowError("Live window control requires Windows; replay works on other platforms")
    api = ctypes.WinDLL("user32", use_last_error=True)
    api.GetForegroundWindow.restype = wintypes.HWND
    api.IsWindow.argtypes = [wintypes.HWND]
    api.IsWindowVisible.argtypes = [wintypes.HWND]
    api.IsIconic.argtypes = [wintypes.HWND]
    api.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    api.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    api.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    api.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    api.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    api.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
    api.GetAsyncKeyState.argtypes = [ctypes.c_int]
    api.GetAsyncKeyState.restype = ctypes.c_short
    api.SetProcessDpiAwarenessContext.argtypes = [wintypes.HANDLE]
    # Per-monitor v2 coordinates, before screenshot or window geometry reads.
    api.SetProcessDpiAwarenessContext(wintypes.HANDLE(-4))
    return api


def list_windows() -> list[dict]:
    api = _user32()
    result = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def callback(hwnd, _):
        if api.IsWindowVisible(hwnd) and (length := api.GetWindowTextLengthW(hwnd)):
            text = ctypes.create_unicode_buffer(length + 1)
            api.GetWindowTextW(hwnd, text, len(text))
            class_name = ctypes.create_unicode_buffer(256)
            api.GetClassNameW(hwnd, class_name, len(class_name))
            result.append({"hwnd": int(hwnd), "title": text.value,
                           "class_name": class_name.value})
        return True

    api.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    api.EnumWindows(callback, 0)
    return result


class WindowsClient:
    def __init__(self, hwnd: int):
        self.api = _user32()
        self.hwnd = hwnd
        if not self.api.IsWindow(hwnd):
            raise WindowError("Window handle does not exist")

    def bounds(self) -> tuple[int, int, int, int]:
        if not self.api.IsWindow(self.hwnd) or self.api.IsIconic(self.hwnd):
            raise WindowError("Window closed or minimized")
        rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
        if not self.api.GetClientRect(self.hwnd, ctypes.byref(rect)) or not self.api.ClientToScreen(self.hwnd, ctypes.byref(origin)):
            raise WindowError("Cannot get client bounds")
        if rect.right <= 0 or rect.bottom <= 0:
            raise WindowError("Empty client area")
        return origin.x, origin.y, rect.right, rect.bottom

    def check(self, expected: tuple[int, int, int, int] | None = None) -> None:
        if self.api.GetAsyncKeyState(0x1B) & 0x8000:
            raise KeyboardInterrupt
        if self.api.GetForegroundWindow() != self.hwnd:
            raise WindowError("Target window is not foreground")
        if expected is not None and self.bounds() != expected:
            raise WindowError("Window moved or resized since observation")

    def capture(self) -> Frame:
        from PIL import ImageGrab
        self.check()
        bounds = self.bounds()
        x, y, width, height = bounds
        timestamp = time.monotonic()
        image = ImageGrab.grab(bbox=(x, y, x + width, y + height), all_screens=True)
        self.check(bounds)
        return Frame(image, timestamp, bounds)

    def click(self, point: tuple[float, float], expected: tuple[int, int, int, int], *,
              hold_seconds: float = 0.05, click_count: int = 1,
              inter_click_seconds: float = 0.05) -> None:
        if (isinstance(hold_seconds, bool) or not isinstance(hold_seconds, (int, float)) or
                not math.isfinite(hold_seconds) or not 0 < hold_seconds <= 0.2):
            raise WindowError('Mouse press duration must be in (0, 0.2] seconds')
        if type(click_count) is not int or click_count not in (1, 2):
            raise WindowError('Mouse click count must be 1 or 2')
        if (isinstance(inter_click_seconds, bool) or
                not isinstance(inter_click_seconds, (int, float)) or
                not math.isfinite(inter_click_seconds) or not 0 < inter_click_seconds <= 0.2):
            raise WindowError('Mouse inter-click duration must be in (0, 0.2] seconds')
        self.check(expected)
        x, y = point
        if not 0 <= x < expected[2] or not 0 <= y < expected[3]:
            raise WindowError("Click target is outside the client area")
        if not self.api.SetCursorPos(round(expected[0] + x), round(expected[1] + y)):
            raise WindowError("Failed to position mouse")
        for index in range(click_count):
            if index:
                time.sleep(inter_click_seconds)
            self.check(expected)
            # Recheck focus/geometry before each press; release even on interruption.
            self.api.mouse_event(0x0002, 0, 0, 0, 0)
            try:
                time.sleep(hold_seconds)
            finally:
                self.api.mouse_event(0x0004, 0, 0, 0, 0)
