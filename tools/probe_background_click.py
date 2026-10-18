# Probe: does PostMessage(WM_LBUTTONDOWN/UP) reach the majsoul Unity client
# while the window is occluded? Clicks ranked room, checks for a visual change via
# PrintWindow, then clicks return to restore the main menu.
import ctypes
from ctypes import wintypes
import time

import numpy as np

from majsoul.window import list_windows

user32 = ctypes.windll.user32

WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
MK_LBUTTON = 0x0001


def find_hwnd(keyword):
    for w in list_windows():
        if keyword in w["title"]:
            return w["hwnd"]
    raise SystemExit(f"window {keyword!r} not found")


def window_rect(hwnd):
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def client_origin_offset(hwnd):
    """(client_screen_x - window_x, client_screen_y - window_y)."""
    wx, wy, _, _ = window_rect(hwnd)
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x - wx, pt.y - wy


def post_click(hwnd, window_point):
    off_x, off_y = client_origin_offset(hwnd)
    cx = int(window_point[0] - off_x)
    cy = int(window_point[1] - off_y)
    lparam = (cy << 16) | (cx & 0xFFFF)
    user32.PostMessageW(hwnd, WM_MOUSEMOVE, 0, lparam)
    time.sleep(0.05)
    user32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
    time.sleep(0.05)
    user32.PostMessageW(hwnd, WM_LBUTTONUP, 0, lparam)


def grab(hwnd):
    from probe_background_capture import grab_printwindow
    return grab_printwindow(hwnd)[1]


def main():
    hwnd = find_hwnd("雀魂")
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 4)  # SW_SHOWNOACTIVATE: un-minimize without stealing focus
        time.sleep(1.0)
    rect = window_rect(hwnd)
    print("rect:", rect, "client offset:", client_origin_offset(hwnd))
    w, h = rect[2], rect[3]

    before = grab(hwnd)

    # ranked room banner center, normalized from the calibrated menu config
    target = (0.725 * w, 0.320 * h)
    print("post click at window point", target)
    post_click(hwnd, target)
    time.sleep(2.5)
    after = grab(hwnd)
    diff = np.abs(after.astype(int) - before.astype(int)).mean()
    print("mean diff after click =", round(diff, 2))
    if diff > 5:
        print("RESULT: background PostMessage click WORKS")
    else:
        print("RESULT: no visible change; PostMessage click likely ignored")

    from majsoul.calibration.frames import write_image
    write_image("artifacts/probe_click_after.png", after)

    if diff > 5:
        # click return at top of the ranked page to go back
        back = (0.655 * w, 0.265 * h)
        print("post click return at", back)
        post_click(hwnd, back)
        time.sleep(2.5)
        restored = grab(hwnd)
        write_image("artifacts/probe_click_restored.png", restored)
        print("diff vs main menu =",
              round(np.abs(restored.astype(int) - before.astype(int)).mean(), 2))


if __name__ == "__main__":
    main()
