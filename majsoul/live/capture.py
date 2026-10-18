# Steam-client live capture: background PrintWindow frames (occlusion-proof),
# focus-stealing clicks, and baking a reviewed ROI projection into frame
# coordinates.
import ctypes
from ctypes import wintypes
from dataclasses import dataclass, replace
import re
import time

import numpy as np

from ..calibration.roi_config import PROJECTION_NAMES, ROIConfig
from ..window import WindowError, WindowsClient, _user32, list_windows

_PW_RENDERFULLCONTENT = 2
_SW_SHOWNOACTIVATE = 4


class FocusUnavailable(WindowError):
    """Focus could not be acquired; no input has been sent."""


@dataclass
class LiveFrame:
    image: np.ndarray
    captured_at: float
    window: tuple[int, int, int, int]
    client_offset: tuple[int, int]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


def _print_window(hwnd: int, width: int, height: int) -> np.ndarray:
    """Capture the window (incl. title bar) even while occluded. BGR uint8."""
    user32 = _user32()
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    # Explicit pointer-sized signatures are required on 64-bit Windows.
    user32.GetWindowDC.argtypes = [wintypes.HWND]
    user32.GetWindowDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    for name in ("CreateCompatibleDC", "DeleteDC"):
        getattr(gdi32, name).argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HANDLE
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
    gdi32.SelectObject.restype = wintypes.HANDLE
    gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
    gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HANDLE, wintypes.UINT,
                               wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p,
                               wintypes.UINT]
    window_dc = user32.GetWindowDC(hwnd)
    if not window_dc:
        raise WindowError("GetWindowDC failed")
    mem_dc = bitmap = previous_bitmap = None
    try:
        mem_dc = gdi32.CreateCompatibleDC(window_dc)
        if not mem_dc:
            raise WindowError("CreateCompatibleDC failed")
        bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
        if not bitmap:
            raise WindowError("CreateCompatibleBitmap failed")
        previous_bitmap = gdi32.SelectObject(mem_dc, bitmap)
        if not previous_bitmap or previous_bitmap == ctypes.c_void_p(-1).value:
            raise WindowError("SelectObject failed")
        if not user32.PrintWindow(hwnd, mem_dc, _PW_RENDERFULLCONTENT):
            raise WindowError("PrintWindow failed")
        header = _BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        header.biWidth = width
        header.biHeight = -height  # top-down rows
        header.biPlanes = 1
        header.biBitCount = 32
        buffer = (ctypes.c_ubyte * (width * height * 4))()
        gdi32.SelectObject(mem_dc, previous_bitmap)
        previous_bitmap = None
        if gdi32.GetDIBits(mem_dc, bitmap, 0, height, buffer,
                           ctypes.byref(header), 0) != height:
            raise WindowError("GetDIBits failed")
        return np.frombuffer(buffer, dtype=np.uint8).reshape(
            height, width, 4)[:, :, :3].copy()
    finally:
        if previous_bitmap and previous_bitmap != ctypes.c_void_p(-1).value:
            gdi32.SelectObject(mem_dc, previous_bitmap)
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if mem_dc:
            gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, window_dc)


class LiveWindow:
    def __init__(self, hwnd: int):
        self.client = WindowsClient(hwnd)
        self.hwnd = hwnd
        self.api = _user32()
        self.api.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.api.SetForegroundWindow.argtypes = [wintypes.HWND]
        self.api.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self.api.GetForegroundWindow.restype = wintypes.HWND
        self.api.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.api.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.api.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
        self.api.BringWindowToTop.argtypes = [wintypes.HWND]
        self.api.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]

    @classmethod
    def find(cls, title_regex: str = "雀魂", window_class_regex: str | None = None) -> int:
        pattern = re.compile(title_regex)
        candidates = [w for w in list_windows() if pattern.search(w["title"])]
        if window_class_regex is not None:
            class_pattern = re.compile(window_class_regex)
            candidates = [w for w in candidates
                          if class_pattern.search(w.get("class_name", ""))]
        if len(candidates) != 1:
            titles = ", ".join(w["title"] for w in candidates) or "(none)"
            raise WindowError(
                f"Expected exactly one window matching {title_regex!r} "
                f"(class={window_class_regex!r}), "
                f"found {len(candidates)}: {titles}")
        return int(candidates[0]["hwnd"])

    def activate(self) -> None:
        if self.api.IsIconic(self.hwnd):
            self.api.ShowWindow(self.hwnd, 9)  # SW_RESTORE
        if not self.api.SetForegroundWindow(self.hwnd):
            raise WindowError("Failed to bring window to foreground")

    def restore_if_minimized(self) -> None:
        if self.api.IsIconic(self.hwnd):
            self.api.ShowWindow(self.hwnd, _SW_SHOWNOACTIVATE)

    def window_rect(self) -> tuple[int, int, int, int]:
        if not self.api.IsWindow(self.hwnd):
            raise WindowError("Window closed")
        if self.api.IsIconic(self.hwnd):
            raise WindowError(
                "Window is minimized; restore it (occlusion by other windows is fine)")
        rect = wintypes.RECT()
        if not self.api.GetWindowRect(self.hwnd, ctypes.byref(rect)):
            raise WindowError("Cannot get window rect")
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top

    def _check_capture(self) -> None:
        if not getattr(self,'ignore_stop_key',False) and self.api.GetAsyncKeyState(0x1B) & 0x8000:  # ESC stops Agent; protector finishes match.
            raise KeyboardInterrupt
        if not self.api.IsWindow(self.hwnd):
            raise WindowError("Window closed")

    def capture(self) -> LiveFrame:
        self._check_capture()
        self.restore_if_minimized()
        rect = self.window_rect()
        x, y, width, height = rect
        timestamp = time.monotonic()
        image = _print_window(self.hwnd, width, height)
        self._check_capture()
        if self.window_rect() != rect:
            raise WindowError("Window moved or resized since observation")
        client = self.client.bounds()
        return LiveFrame(image, timestamp, rect, (client[0] - x, client[1] - y))

    def _force_foreground(self) -> int | None:
        previous = self.api.GetForegroundWindow()
        if previous == self.hwnd:
            return previous
        foreground_tid = self.api.GetWindowThreadProcessId(previous, None) if previous else 0
        current_tid = ctypes.windll.kernel32.GetCurrentThreadId()
        attached = bool(foreground_tid and foreground_tid != current_tid and
                        self.api.AttachThreadInput(current_tid, foreground_tid, True))
        try:
            self.api.BringWindowToTop(self.hwnd)
            self.api.SetForegroundWindow(self.hwnd)
            time.sleep(0.05)
        finally:
            if attached:
                self.api.AttachThreadInput(current_tid, foreground_tid, False)
        if self.api.GetForegroundWindow() != self.hwnd:
            raise FocusUnavailable("Failed to focus the game window; refusing to click blind")
        return previous

    def _restore_foreground(self, previous: int | None) -> None:
        if not previous or previous == self.hwnd:
            return
        # Respect a user's focus change during the brief click.
        if self.api.GetForegroundWindow() != self.hwnd:
            return
        if self.api.IsWindow(previous) and not self.api.IsIconic(previous):
            previous_tid = self.api.GetWindowThreadProcessId(previous, None)
            current_tid = ctypes.windll.kernel32.GetCurrentThreadId()
            attached = bool(previous_tid != current_tid and
                            self.api.AttachThreadInput(current_tid, previous_tid, True))
            try:
                self.api.SetForegroundWindow(previous)
            finally:
                if attached:
                    self.api.AttachThreadInput(current_tid, previous_tid, False)
            if self.api.GetForegroundWindow() == self.hwnd:
                raise WindowError("Failed to restore the previous foreground window")

    def click(self, point: tuple[float, float], *, expected: LiveFrame | None = None,
              max_age_seconds: float = 2.0, hold_seconds: float = 0.05,
              click_count: int = 1, inter_click_seconds: float = 0.05) -> None:
        started = time.monotonic()
        self.last_click_timing = {'started_at': started, 'input_completed': False}
        self._check_capture()
        rect = self.window_rect()
        if expected is not None:
            if rect != expected.window:
                raise WindowError("Window moved or resized since observation")
            if time.monotonic() - expected.captured_at > max_age_seconds:
                raise WindowError("UI observation is stale; refusing to click")
        bounds = self.client.bounds()
        x, y = point
        if not 0 <= x < rect[2] or not 0 <= y < rect[3]:
            raise WindowError("Click target is outside the window")
        offset_x, offset_y = bounds[0] - rect[0], bounds[1] - rect[1]
        previous = self.api.GetForegroundWindow()
        cursor = wintypes.POINT()
        have_cursor = self.api.GetCursorPos(ctypes.byref(cursor))
        clicked_cursor = (round(rect[0] + x), round(rect[1] + y))
        try:
            self._force_foreground()
            self.last_click_timing['foreground_acquired_at'] = time.monotonic()
            time.sleep(0.08)
            self._check_capture()
            if self.window_rect() != rect:
                raise WindowError("Window moved or resized while focusing")
            if expected is not None and time.monotonic() - expected.captured_at > max_age_seconds:
                raise WindowError("UI observation expired while focusing")
            options = {'hold_seconds': hold_seconds}
            if click_count != 1:
                options.update(click_count=click_count, inter_click_seconds=inter_click_seconds)
            self.client.click((x - offset_x, y - offset_y), bounds, **options)
            self.last_click_timing['click_count'] = click_count
            self.last_click_timing['input_completed'] = True
            self.last_click_timing['input_finished_at'] = time.monotonic()
            time.sleep(0.08)
        finally:
            try:
                current = wintypes.POINT()
                if (have_cursor and self.api.GetCursorPos(ctypes.byref(current)) and
                        (current.x, current.y) == clicked_cursor):
                    self.api.SetCursorPos(cursor.x, cursor.y)
            finally:
                try:
                    self._restore_foreground(previous)
                finally:
                    ended = time.monotonic()
                    self.last_click_timing.update(ended_at=ended,
                        total_seconds=ended-started,
                        foreground_seconds=(ended-self.last_click_timing['foreground_acquired_at']
                            if 'foreground_acquired_at' in self.last_click_timing else None),
                        previous_foreground_restored=(self.api.GetForegroundWindow()==previous))


def roi_for_frame(roi: ROIConfig, size: tuple[int, int],
                  projection: str = "auto") -> ROIConfig:
    """Bake the reviewed projection into a ROIConfig whose reference is the frame."""
    if projection == "auto":
        if roi.assess_target_size(size).requires_trials:
            raise ValueError(
                "Target aspect ratio requires explicit projection selection; "
                "review the trials and pass one of: " + ", ".join(PROJECTION_NAMES))
        return roi
    width, height = size
    projected = roi.projection(size, projection)
    if projected.clipped_regions:
        raise ValueError(
            f"Projection {projection} clips calibrated regions: "
            + ", ".join(projected.clipped_regions))
    regions = {}
    for name, rect in roi.regions.items():
        if rect is None:
            regions[name] = None
            continue
        x1, y1, x2, y2 = projected.boxes[name]
        regions[name] = [x1 / width, y1 / height, x2 / width, y2 / height]
    rw, rh = roi.reference_size
    polygons = {name: [[(x*rw*projected.scale_x+projected.offset_x)/width,
                        (y*rh*projected.scale_y+projected.offset_y)/height]
                       for x, y in points]
                for name, points in roi.region_polygons.items()}
    baked = replace(roi, reference_size=[width, height], regions=regions,
                    region_polygons=polygons)
    baked.validate()
    return baked
