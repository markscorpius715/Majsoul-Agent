"""Interactive-ish browser controller for maj-soul web client testing.

Uses Playwright with a persistent profile (artifacts/browser-profile) so login
state survives across CLI invocations. Each invocation launches a visible
(headless=False) Chromium window, performs one staged action, screenshots, and
exits. Credentials come from CLI args or env vars MAJSOUL_EMAIL /
MAJSOUL_PASSWORD; they are never written to disk by this tool.

Subcommands:
  open      --url URL [--wait S]                 open page, screenshot
  inspect   [--url URL]                          list buttons/inputs/links text
  login     --email E --password P               full login flow with checkpoints
  eval-js   --expr JS                            evaluate JS in page, print result
  click-text --text STR                          click first element containing text
  shot      [--out PATH]                         screenshot current page

Artifacts default to artifacts/browser-lab/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402

PROFILE_DIR = REPO_ROOT / "artifacts" / "browser-profile"
OUT_DIR = REPO_ROOT / "artifacts" / "browser-lab"
GAME_URL = "https://game.maj-soul.com/1/"


def _stamp() -> str:
    return time.strftime("%H%M%S")


def launch(pw, headless: bool = False):
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    proxy_url = os.environ.get("BROWSER_PROXY") or None
    return pw.chromium.launch_persistent_context(
        str(PROFILE_DIR),
        headless=headless,
        viewport={"width": 1920, "height": 1080},
        proxy={"server": proxy_url} if proxy_url else None,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--window-position=0,0",
        ],
        ignore_default_args=["--enable-automation"],
    )


def shot(page, name: str, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_stamp()}_{name}.png"
    page.screenshot(path=str(path))
    print(f"[shot] {path}")
    return path


def cmd_open(args):
    with sync_playwright() as pw:
        ctx = launch(pw, headless=args.headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(int(args.wait * 1000))
        print("[title]", page.title())
        print("[url]", page.url)
        shot(page, args.name)
        ctx.close()


def cmd_inspect(args):
    with sync_playwright() as pw:
        ctx = launch(pw, headless=args.headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        if args.url:
            page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(int(args.wait * 1000))
        info = page.evaluate(
            """() => {
                const els = [...document.querySelectorAll(
                    'button, input, a, [role=button], select')];
                return els.slice(0, 200).map(e => ({
                    tag: e.tagName.toLowerCase(),
                    type: e.type || null,
                    text: (e.innerText || e.value || e.placeholder || '')
                        .trim().slice(0, 60),
                    id: e.id || null,
                    cls: (e.className || '').toString().slice(0, 80),
                    visible: !!(e.offsetWidth || e.offsetHeight),
                }));
            }"""
        )
        print(json.dumps(info, ensure_ascii=False, indent=1))
        shot(page, "inspect")
        ctx.close()


def cmd_eval(args):
    with sync_playwright() as pw:
        ctx = launch(pw, headless=args.headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        if args.url:
            page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(int(args.wait * 1000))
        result = page.evaluate(args.expr)
        print(json.dumps(result, ensure_ascii=False, indent=1, default=str)[:4000])
        ctx.close()


def cmd_click_text(args):
    with sync_playwright() as pw:
        ctx = launch(pw, headless=args.headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        if args.url:
            page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(int(args.wait * 1000))
        page.get_by_text(args.text, exact=False).first.click(timeout=10000)
        page.wait_for_timeout(int(args.wait * 1000))
        shot(page, f"clicked_{args.text[:12]}")
        ctx.close()


def _page_pixel(page, fx: float, fy: float) -> tuple[int, int, int]:
    """Screenshot the page and return the RGB pixel at fractional position."""
    import io
    from PIL import Image
    buf = page.screenshot()
    with Image.open(io.BytesIO(buf)) as img:
        img = img.convert("RGB")
        x = min(max(int(fx * img.width), 0), img.width - 1)
        y = min(max(int(fy * img.height), 0), img.height - 1)
        return img.getpixel((x, y))


def _region_match(page, fx, fy, predicate, radius_frac=0.008, need=0.2):
    """True if >= `need` fraction of a small grid around (fx, fy) matches."""
    import io
    from PIL import Image
    buf = page.screenshot()
    with Image.open(io.BytesIO(buf)) as img:
        img = img.convert("RGB")
        w, h = img.size
        cx, cy = int(fx * w), int(fy * h)
        rx, ry = max(2, int(radius_frac * w)), max(2, int(radius_frac * h))
        hits = total = 0
        for yy in range(max(0, cy - ry), min(h, cy + ry + 1), max(1, ry // 2)):
            for xx in range(max(0, cx - rx), min(w, cx + rx + 1), max(1, rx // 2)):
                total += 1
                if predicate(*img.getpixel((xx, yy))):
                    hits += 1
        return total > 0 and hits / total >= need


def wait_for_pixel(page, fx, fy, predicate, timeout=300, interval=3.0, label=""):
    """Poll a fractional region until enough pixels satisfy predicate."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _region_match(page, fx, fy, predicate):
            print(f"[wait-pixel {label}] ready")
            return True
        page.wait_for_timeout(int(interval * 1000))
    print(f"[wait-pixel {label}] TIMEOUT after {timeout}s")
    return False


def _is_gold(r, g, b):
    return r > 170 and g > 130 and b < 120 and r > b + 70


def cmd_login(args):
    """Canvas-based login: the game is WebGL, inputs are focused by clicking
    their on-canvas positions (fractions of the 1920x1080 viewport)."""
    email = args.email or os.environ.get("MAJSOUL_EMAIL")
    password = args.password or os.environ.get("MAJSOUL_PASSWORD")
    if not email or not password:
        raise SystemExit("email/password required (args or MAJSOUL_EMAIL/MAJSOUL_PASSWORD)")
    with sync_playwright() as pw:
        ctx = launch(pw, headless=args.headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(GAME_URL, wait_until="domcontentloaded", timeout=60000)
        # wait for resource download + login screen: the gold 進入遊戲 button
        page.wait_for_timeout(int(args.load_wait * 1000))
        if not wait_for_pixel(page, *args.submit_xy, _is_gold,
                              timeout=420, label="login-button"):
            shot(page, "login_fail_no_button")
            ctx.close()
            return
        shot(page, "login_00_start")

        def click_frac(fx, fy, settle=1.0):
            page.mouse.click(fx * 1920, fy * 1080)
            page.wait_for_timeout(int(settle * 1000))

        click_frac(*args.email_xy, settle=1.5)
        page.keyboard.type(email, delay=40)
        shot(page, "login_01_email")
        click_frac(*args.password_xy, settle=1.5)
        page.keyboard.type(password, delay=40)
        shot(page, "login_02_password")
        click_frac(*args.submit_xy, settle=10.0)
        shot(page, "login_03_submitted")
        print("[url after login]", page.url)
        ctx.close()


def cmd_click_xy(args):
    with sync_playwright() as pw:
        ctx = launch(pw, headless=args.headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        if args.url:
            page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(int(args.pre_wait * 1000))
        # coordinates are fractions of the 1920x1080 viewport when <= 1.5
        x = args.x * 1920 if args.x <= 1.5 else args.x
        y = args.y * 1080 if args.y <= 1.5 else args.y
        page.mouse.click(x, y)
        page.wait_for_timeout(int(args.wait * 1000))
        shot(page, args.name)
        ctx.close()


def cmd_type(args):
    with sync_playwright() as pw:
        ctx = launch(pw, headless=args.headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        if args.url:
            page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(int(args.pre_wait * 1000))
        if args.key:
            page.keyboard.press(args.key)
        else:
            page.keyboard.type(args.text, delay=40)
        page.wait_for_timeout(int(args.wait * 1000))
        shot(page, args.name)
        ctx.close()


def cmd_serve(args):
    """Long-running single-browser session driven by a command file.

    Each line appended to --cmd-file is one JSON command:
      {"cmd": "goto", "url": "..."}
      {"cmd": "shot", "name": "..."}
      {"cmd": "click", "x": 0.5, "y": 0.5, "settle": 1.0}   # fractions of viewport
      {"cmd": "type", "text": "..."} / {"cmd": "key", "key": "Enter"}
      {"cmd": "eval", "expr": "..."}
      {"cmd": "wait_gold", "x": 0.74, "y": 0.59, "timeout": 120}
      {"cmd": "sleep", "seconds": 5}
      {"cmd": "quit"}
    Results are appended to serve_log.jsonl in the output dir.
    """
    cmd_file = Path(args.cmd_file)
    log_path = OUT_DIR / "serve_log.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    processed = 0

    def log(record):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    with sync_playwright() as pw:
        ctx = launch(pw, headless=args.headless)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        if page.url.startswith("about:"):
            page.goto(GAME_URL, wait_until="domcontentloaded", timeout=60000)
        log({"event": "serve_start", "url": page.url})
        while True:
            if cmd_file.exists():
                lines = cmd_file.read_text(encoding="utf-8").splitlines()
                new = [l for l in lines[processed:] if l.strip()]
                processed = len(lines)
                for line in new:
                    try:
                        cmd = json.loads(line)
                    except json.JSONDecodeError:
                        log({"error": "bad json", "line": line})
                        continue
                    op = cmd.get("cmd")
                    log({"cmd": cmd})
                    try:
                        if op == "quit":
                            log({"event": "serve_stop"})
                            ctx.close()
                            return
                        elif op == "goto":
                            page.goto(cmd["url"], wait_until="domcontentloaded", timeout=60000)
                        elif op == "shot":
                            path = shot(page, cmd.get("name", "serve"))
                            log({"result": str(path)})
                        elif op == "click":
                            page.mouse.click(cmd["x"] * 1920, cmd["y"] * 1080)
                            page.wait_for_timeout(int(cmd.get("settle", 1.0) * 1000))
                        elif op == "type":
                            page.keyboard.type(cmd["text"], delay=cmd.get("delay", 40))
                        elif op == "key":
                            page.keyboard.press(cmd["key"])
                        elif op == "eval":
                            log({"result": page.evaluate(cmd["expr"])})
                        elif op == "wait_gold":
                            ok = wait_for_pixel(page, cmd["x"], cmd["y"], _is_gold,
                                                timeout=cmd.get("timeout", 120),
                                                label="serve")
                            log({"result": ok})
                        elif op == "sleep":
                            page.wait_for_timeout(int(cmd.get("seconds", 1) * 1000))
                        else:
                            log({"error": f"unknown cmd {op}"})
                    except Exception as exc:  # noqa: BLE001
                        log({"error": f"{type(exc).__name__}: {exc}"})
            time.sleep(0.7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true",
                        help="debug only; default is a visible window")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("open")
    p.add_argument("--url", default=GAME_URL)
    p.add_argument("--wait", type=float, default=8)
    p.add_argument("--name", default="open")
    p.set_defaults(fn=cmd_open)

    p = sub.add_parser("inspect")
    p.add_argument("--url", default=None)
    p.add_argument("--wait", type=float, default=5)
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("eval-js")
    p.add_argument("--expr", required=True)
    p.add_argument("--url", default=None)
    p.add_argument("--wait", type=float, default=3)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("click-text")
    p.add_argument("--text", required=True)
    p.add_argument("--url", default=None)
    p.add_argument("--wait", type=float, default=3)
    p.set_defaults(fn=cmd_click_text)

    p = sub.add_parser("login")
    p.add_argument("--email", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--load-wait", type=float, default=10,
                   help="initial settle before polling for the login screen")
    p.add_argument("--email-xy", type=float, nargs=2, default=[0.755, 0.287])
    p.add_argument("--password-xy", type=float, nargs=2, default=[0.755, 0.395])
    p.add_argument("--submit-xy", type=float, nargs=2, default=[0.741, 0.589])
    p.set_defaults(fn=cmd_login)

    p = sub.add_parser("click-xy")
    p.add_argument("--x", type=float, required=True)
    p.add_argument("--y", type=float, required=True)
    p.add_argument("--url", default=None)
    p.add_argument("--pre-wait", type=float, default=3)
    p.add_argument("--wait", type=float, default=2)
    p.add_argument("--name", default="clickxy")
    p.set_defaults(fn=cmd_click_xy)

    p = sub.add_parser("type")
    p.add_argument("--text", default=None)
    p.add_argument("--key", default=None, help="press a key instead of typing")
    p.add_argument("--url", default=None)
    p.add_argument("--pre-wait", type=float, default=3)
    p.add_argument("--wait", type=float, default=2)
    p.add_argument("--name", default="type")
    p.set_defaults(fn=cmd_type)

    p = sub.add_parser("serve")
    p.add_argument("--cmd-file", default=str(OUT_DIR / "commands.jsonl"))
    p.set_defaults(fn=cmd_serve)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
