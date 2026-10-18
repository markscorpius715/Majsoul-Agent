# Probe: can we capture the majsoul window with PrintWindow while it is
# occluded by another window? Compares ImageGrab (screen copy) vs PrintWindow.
import ctypes
from ctypes import wintypes
import time

import numpy as np

from majsoul.window import list_windows

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

PW_RENDERFULLCONTENT = 2


def find_hwnd(keyword):
    for w in list_windows():
        if keyword in w["title"]:
            return w["hwnd"]
    raise SystemExit(f"window {keyword!r} not found")


def window_rect(hwnd):
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def grab_screen(hwnd):
    from PIL import ImageGrab
    x, y, w, h = window_rect(hwnd)
    return np.asarray(ImageGrab.grab(bbox=(x, y, x + w, y + h), all_screens=True))


def grab_printwindow(hwnd, mode=PW_RENDERFULLCONTENT):
    x, y, w, h = window_rect(hwnd)
    hwnd_dc = user32.GetWindowDC(hwnd)
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
    gdi32.SelectObject(mem_dc, bitmap)
    ok = user32.PrintWindow(hwnd, mem_dc, mode)
    # bitmap -> numpy
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD)]
    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h  # top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    buf = (ctypes.c_ubyte * (w * h * 4))()
    gdi32.GetDIBits(mem_dc, bitmap, 0, h, buf, ctypes.byref(bmi), 0)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, hwnd_dc)
    image = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3][:, :, ::-1]
    return ok, image


def stats(name, img):
    print(f"{name}: shape={img.shape} mean_bgr={img.mean(axis=(0, 1)).round(1)}")


def main():
    hwnd = find_hwnd("雀魂")
    print("hwnd:", hwnd, "rect:", window_rect(hwnd))

    ok, pw1 = grab_printwindow(hwnd)
    sg1 = grab_screen(hwnd)
    stats("printwindow(foreground)", pw1)
    stats("imagegrab(foreground)", sg1)
    print("PrintWindow ok =", ok)
    same = np.abs(pw1.astype(int) - sg1.astype(int)).mean()
    print("mean|pw-screen| =", round(same, 2), "(small means window is foreground)")

    # occlude: bring another window to front
    others = [w for w in list_windows()
              if w["hwnd"] != hwnd and w["title"] and "雀魂" not in w["title"]]
    blocker = None
    for w in others:
        if "Program Manager" in w["title"]:
            continue
        blocker = w
        break
    print("activating blocker:", blocker["title"][:40])
    user32.SetForegroundWindow(blocker["hwnd"])
    time.sleep(1.0)

    ok2, pw2 = grab_printwindow(hwnd)
    sg2 = grab_screen(hwnd)
    stats("printwindow(occluded)", pw2)
    stats("imagegrab(occluded)", sg2)
    print("PrintWindow occluded ok =", ok2)
    diff_pw = np.abs(pw2.astype(int) - pw1.astype(int)).mean()
    diff_sg = np.abs(sg2.astype(int) - sg1.astype(int)).mean()
    print("pw(occluded) vs pw(foreground) mean diff =", round(diff_pw, 2))
    print("screen(occluded) vs screen(foreground) mean diff =", round(diff_sg, 2))
    if diff_pw < 5 and diff_sg > 10:
        print("RESULT: PrintWindow works under occlusion; ImageGrab does not.")
    else:
        print("RESULT: inconclusive, inspect numbers above")

    from majsoul.calibration.frames import write_image
    write_image("artifacts/probe_pw_occluded.png", pw2)
    write_image("artifacts/probe_screen_occluded.png", sg2)
    user32.SetForegroundWindow(hwnd)
    print("restored foreground to majsoul; wrote artifacts/probe_*.png")


if __name__ == "__main__":
    main()
