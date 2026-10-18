# Probe round 2: inspect majsoul window class/children, try harder background
# click variants (child window, SendMessage, WM_MOUSEACTIVATE ceremony).
import ctypes
from ctypes import wintypes
import time

import numpy as np

from majsoul.window import list_windows
from probe_background_capture import grab_printwindow

user32 = ctypes.windll.user32

WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_MOUSEACTIVATE = 0x0021
WM_SETFOCUS = 0x0007
WM_ACTIVATE = 0x0006
MK_LBUTTON = 0x0001
WA_ACTIVE = 1
HTCLIENT = 1
MA_ACTIVATE = 1


def find_hwnd(keyword):
    for w in list_windows():
        if keyword in w["title"]:
            return w["hwnd"]
    raise SystemExit("not found")


def class_name(hwnd):
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def list_children(hwnd):
    result = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(child, _):
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(child, buf, 256)
        result.append((child, class_name(child), buf.value))
        return True

    user32.EnumChildWindows(hwnd, callback, 0)
    return result


def window_rect(hwnd):
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def client_point(hwnd, window_point):
    wx, wy, _, _ = window_rect(hwnd)
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return int(window_point[0] - (pt.x - wx)), int(window_point[1] - (pt.y - wy))


def lparam(cx, cy):
    return (cy << 16) | (cx & 0xFFFF)


def diff(a, b):
    return round(float(np.abs(a.astype(int) - b.astype(int)).mean()), 2)


def try_click(hwnd, window_point, mode, send=False, ceremony=False):
    cx, cy = client_point(hwnd, window_point)
    lp = lparam(cx, cy)
    poster = user32.SendMessageW if send else user32.PostMessageW
    if ceremony:
        poster(hwnd, WM_MOUSEACTIVATE, hwnd, lparam(HTCLIENT, WM_LBUTTONDOWN))
        poster(hwnd, WM_ACTIVATE, WA_ACTIVE, 0)
        poster(hwnd, WM_SETFOCUS, 0, 0)
    poster(hwnd, WM_MOUSEMOVE, 0, lp)
    time.sleep(0.05)
    poster(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lp)
    time.sleep(0.08)
    poster(hwnd, WM_LBUTTONUP, 0, lp)
    print(f"  tried {mode} on hwnd={hwnd}")


def main():
    hwnd = find_hwnd("雀魂")
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 4)
        time.sleep(1.0)
    print("class:", class_name(hwnd))
    children = list_children(hwnd)
    print("children:", children)

    w, h = window_rect(hwnd)[2], window_rect(hwnd)[3]
    target = (0.725 * w, 0.320 * h)  # ranked room banner

    variants = []
    if children:
        variants.append((children[0][0], "child-window PostMessage", False, False))
    variants += [
        (hwnd, "SendMessage", True, False),
        (hwnd, "PostMessage + activate ceremony", False, True),
    ]
    for target_hwnd, mode, send, ceremony in variants:
        before = grab_printwindow(hwnd)[1]
        try_click(target_hwnd, target, mode, send, ceremony)
        time.sleep(2.5)
        after = grab_printwindow(hwnd)[1]
        d = diff(before, after)
        print(f"  diff = {d}")
        if d > 5:
            print(f"RESULT: {mode} WORKS")
            from majsoul.calibration.frames import write_image
            write_image("artifacts/probe_click2_after.png", after)
            return
    print("RESULT: all background click variants failed")


if __name__ == "__main__":
    main()
