import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from .domain import Box, Detection, Observation, State, TileView
from .rules import RuleEngine, Rules
from .strategy import DeepSeekPolicy, EfficiencyPolicy, ModelSettings
from .jev import JevPolicy, JevSettings, FallbackPolicy
from .tiles import parse_hand, tile_name
from .tracker import StateTracker
from .vision import Layout, ObservationBuilder, REQUIRED_REGIONS


ROOT = Path(__file__).resolve().parent.parent


def dump(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def make_policy(args):
    if args.policy == "efficiency":
        return EfficiencyPolicy()
    deepseek = lambda: DeepSeekPolicy(ModelSettings.load(ROOT, args.reference_env))
    if args.policy == "deepseek":
        return deepseek()
    settings = JevSettings.load(ROOT)
    jev = lambda: JevPolicy(settings)
    return jev() if args.policy == "jev" else FallbackPolicy(jev, deepseek, settings.timeout)


def calibrate(args):
    import cv2
    import numpy as np
    from PIL import Image
    image = np.array(Image.open(args.image).convert("RGB"))[:, :, ::-1].copy()
    height, width = image.shape[:2]
    regions = {}
    try:
        for name in REQUIRED_REGIONS:
            print(f"Select {name}; Enter accepts, Esc cancels. Seat 0=self, 1=top, 2=left.", flush=True)
            x, y, w, h = cv2.selectROI(f"Calibrate {name}", image, showCrosshair=True)
            cv2.destroyAllWindows()
            if not w or not h:
                raise ValueError("Calibration cancelled; output not changed")
            regions[name] = {"rect": [x/width, y/height, (x+w)/width, (y+h)/height],
                             "row_axis": "y", "row_direction": 1, "tile_direction": 1}
    finally:
        cv2.destroyAllWindows()
    rules = Rules.load(args.rules)
    labels = {tile_name(i): "tile:" + tile_name(i) for i in rules.available}
    labels.update({t: "tile:" + t for t in rules.data["red_tiles"]})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(Layout(regions, labels)), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {target}. Adjust river sort orientation and map your detector's metadata/button classes.")


def replay(args, engine):
    layout = Layout.load(args.layout)
    builder, tracker, policy = ObservationBuilder(layout, engine.rules), StateTracker(layout.stable_frames), make_policy(args)
    last = None
    try:
        with Path(args.input).open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                detections = [Detection(d["label"], d["confidence"], Box(*d["box"])) for d in item["detections"]]
                obs = builder.build(detections, item["captured_at"], tuple(item["window"]))
                state = tracker.update(obs)
                actions = engine.legal_actions(state)
                output = {"time": obs.captured_at, "stable": state.stable, "issues": obs.issues,
                          "candidate_count": len(actions)}
                if actions and obs.fingerprint != last:
                    output["choice"] = policy.choose(state, actions).for_model()
                    last = obs.fingerprint
                dump(output)
    finally:
        if hasattr(policy, "close"):
            policy.close()


def run(args, engine):
    from .controller import AgentLoop
    from .vision import YoloDetector
    from .window import WindowsClient
    layout = Layout.load(args.layout)
    # Keep optional library settings in the workspace.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "artifacts" / "ultralytics"))
    detector = YoloDetector(Path(args.weights), layout, args.device)
    policy = make_policy(args)
    try:
        agent = AgentLoop(WindowsClient(args.hwnd), detector, ObservationBuilder(layout, engine.rules),
                          engine, policy, layout, Path(args.artifacts))
        print(f"execute={args.execute}; focus the target window within {args.start_delay:g}s. Esc or Ctrl+C stops.", flush=True)
        time.sleep(args.start_delay)
        while True:
            action = agent.step(args.execute)
            if args.once and action:
                break
            time.sleep(layout.poll_seconds)
    finally:
        if hasattr(policy, "close"):
            policy.close()


def main():
    parser = argparse.ArgumentParser(description="Mahjong Soul visual sanma agent")
    parser.add_argument("--rules", default=str(ROOT / "configs" / "sanma.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("windows", help="List visible Windows handles; does not activate them")
    doctor = sub.add_parser("doctor", help="Check local model configuration; optionally call API")
    doctor.add_argument("--check-api", action="store_true")
    doctor.add_argument("--policy", choices=["jev", "deepseek"], default="deepseek")
    bench = sub.add_parser("benchmark", help="Time actual API decisions on synthetic mahjong snapshots; never clicks")
    bench.add_argument("--rounds", type=int, default=3)
    bench.add_argument("--modes", nargs="+", choices=["stream", "buffered"], default=["stream", "buffered"])
    bench.add_argument("--seed", type=int, default=20260920)
    bench.add_argument("--output", default=None)
    cap = sub.add_parser("capture", help="Collect foreground client screenshots for annotation")
    cap.add_argument("--hwnd", type=lambda v: int(v, 0), required=True)
    cap.add_argument("--output", default="artifacts/captures")
    cap.add_argument("--count", type=int, default=1)
    cap.add_argument("--interval", type=float, default=1)
    cap.add_argument("--start-delay", type=float, default=3)
    cal = sub.add_parser("calibrate", help="Interactively select layout regions in a captured image")
    cal.add_argument("--image", required=True)
    cal.add_argument("--output", default="configs/layout.local.json")
    analyze = sub.add_parser("analyze", help="Offline closed-hand discard features")
    analyze.add_argument("--hand", required=True)
    score = sub.add_parser("score", help="Score a known closed hand using configured sanma assumptions")
    score.add_argument("--hand", required=True)
    score.add_argument("--win-index", type=int, required=True)
    score.add_argument("--seat", choices=["1z", "2z", "3z"], required=True)
    score.add_argument("--round-wind", choices=["1z", "2z", "3z"], default="1z")
    score.add_argument("--tsumo", action="store_true")
    score.add_argument("--riichi", action="store_true")
    score.add_argument("--rinshan", action="store_true")
    score.add_argument("--nuki", type=int, default=0)
    score.add_argument("--indicators", default="")
    for name in ("run", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--layout", required=True)
        p.add_argument("--policy", choices=["auto", "jev", "deepseek", "efficiency"], default="auto" if name == "run" else "efficiency")
        p.add_argument("--reference-env", type=Path, default=None)
        if name == "replay":
            p.add_argument("--input", required=True)
        else:
            p.add_argument("--weights", required=True)
            p.add_argument("--hwnd", type=lambda v: int(v, 0), required=True)
            p.add_argument("--device", default="cpu")
            p.add_argument("--artifacts", default="artifacts/run")
            p.add_argument("--start-delay", type=float, default=3)
            p.add_argument("--execute", action="store_true", help="Execute and verify clicks")
            p.add_argument("--once", action="store_true", help="Stop after one decision")
    args = parser.parse_args()
    try:
        if hasattr(args, "start_delay") and not 0 <= args.start_delay <= 60:
            raise ValueError("start-delay must be between 0 and 60 seconds")
        engine = RuleEngine(Rules.load(args.rules))
        if args.command == "benchmark":
            from .benchmark import run_benchmark
            policy = DeepSeekPolicy(ModelSettings.load(ROOT))
            try:
                output = Path(args.output) if args.output else ROOT / "artifacts" / f"benchmark-{time.time_ns()}"
                report = run_benchmark(engine, policy, rounds=args.rounds, modes=args.modes, seed=args.seed, output=output)
                dump(report)
                print(f"Report: {output / 'summary.json'}")
            finally:
                policy.close()
        elif args.command == "doctor":
            from urllib.parse import urlsplit
            from importlib.metadata import version, PackageNotFoundError
            settings = JevSettings.load(ROOT) if args.policy == "jev" else ModelSettings.load(ROOT)
            result = {"model": settings.model, "api_host": urlsplit(settings.base_url).hostname,
                      "key_configured": bool(settings.api_key)}
            for package in ("mahjong", "ultralytics", "torch"):
                try:
                    result[package] = version(package)
                except PackageNotFoundError:
                    result[package] = "missing"
            if args.check_api:
                from .domain import Action
                policy = JevPolicy(settings) if args.policy == "jev" else DeepSeekPolicy(settings)
                try:
                    diagnostic = Action("diagnostic", "diagnostic")
                    probe = State(Observation(time.monotonic(), (0,0,1,1)), True, ((),(),()))
                    started = time.monotonic()
                    result["api_choice"] = policy.choose(probe, [diagnostic]).id
                    result["latency_seconds"] = round(time.monotonic()-started, 3)
                    result["stream_timing"] = policy.last_timing
                    result["decision_result"] = getattr(policy, "last_result", None)
                finally:
                    policy.close()
            dump(result)
        elif args.command == "windows":
            from .window import list_windows
            dump(list_windows())
        elif args.command == "capture":
            from .window import WindowsClient
            if args.count < 1 or not 0 < args.interval <= 60:
                raise ValueError("count must be positive; interval must be in (0, 60]")
            window = WindowsClient(args.hwnd)
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            print(f"Focus target window within {args.start_delay:g}s; Esc stops.", flush=True)
            time.sleep(args.start_delay)
            for index in range(args.count):
                frame = window.capture()
                path = output / f"{time.time_ns()}-{index}.png"
                frame.image.save(path)
                print(path, flush=True)
                if index + 1 < args.count:
                    time.sleep(args.interval)
        elif args.command == "calibrate":
            calibrate(args)
        elif args.command == "analyze":
            hand = parse_hand(args.hand)
            if len(hand) != 14:
                raise ValueError("analyze expects 14 concealed tiles; melds are supported through the Python API")
            obs = Observation(time.monotonic(), (0, 0, 1, 1),
                              hand=tuple(TileView(t, Box(0, 0, 1, 1)) for t in hand), phase="discard")
            dump([a.for_model() for a in engine.legal_actions(State(obs, True, ((), (), ())))])
        elif args.command == "score":
            dump(engine.score(parse_hand(args.hand), args.win_index, seat=args.seat,
                              round_wind=args.round_wind, tsumo=args.tsumo, riichi=args.riichi,
                              rinshan=args.rinshan, nuki=args.nuki, indicators=tuple(parse_hand(args.indicators))))
        elif args.command == "replay":
            replay(args, engine)
        else:
            run(args, engine)
    except KeyboardInterrupt:
        print("Stopped.")
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Stopped: {exc}")
        return 1
    return 0
