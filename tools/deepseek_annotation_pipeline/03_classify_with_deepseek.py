"""Step 3: classify every pending crop with DeepSeek Flash vision."""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time


def locate_project_root() -> Path:
    """Find the checkout when this script is run from tools/ or an extracted ZIP."""
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if ((candidate / "pyproject.toml").is_file() and
                    (candidate / "majsoul" / "__init__.py").is_file()):
                return candidate
    raise RuntimeError(
        "Cannot find the majsoul project root. Keep this package inside the "
        "majsoul checkout or run it from that checkout.")


PROJECT_ROOT = locate_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from majsoul.continual.deepseek_classifier import (
    ClassificationProgress, run_classification)


class ConsoleProgressBar:
    """Dependency-free progress bar with a heartbeat before the first batch."""

    SPINNER = "|/-\\"

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.total: int | None = None
        self.completed = 0
        self.failed = 0
        self.started_at = time.monotonic()
        self.phase = "loading corpus manifest"
        self.maximum_line = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def update(self, value: ClassificationProgress) -> None:
        with self.lock:
            self.total = value.total
            self.completed = value.completed
            self.failed = value.failed
            self.started_at = value.started_at
            self.phase = "waiting for DeepSeek / classifying"

    @staticmethod
    def _duration(seconds: float) -> str:
        value = max(0, int(seconds))
        hours, remainder = divmod(value, 3600)
        minutes, seconds = divmod(remainder, 60)
        return (f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                if hours else f"{minutes:02d}:{seconds:02d}")

    def _line(self, spinner: str) -> str:
        with self.lock:
            total = self.total
            completed = self.completed
            failed = self.failed
            started_at = self.started_at
            phase = self.phase
        elapsed = max(0.001, time.monotonic() - started_at)
        if total is None:
            return f"[{spinner}] {phase} | elapsed {self._duration(elapsed)}"
        fraction = 1.0 if total == 0 else min(1.0, completed / total)
        width = 30
        filled = int(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        rate = completed / elapsed
        return (
            f"[{bar}] {fraction * 100:6.2f}%  {completed}/{total}  "
            f"failed={failed}  {rate:.2f} img/s  "
            f"elapsed={self._duration(elapsed)}  {spinner}")

    def _run(self) -> None:
        index = 0
        while not self.stop_event.wait(0.2):
            line = self._line(self.SPINNER[index % len(self.SPINNER)])
            index += 1
            self.maximum_line = max(self.maximum_line, len(line))
            sys.stdout.write("\r" + line.ljust(self.maximum_line))
            sys.stdout.flush()

    def finish(self, *, message: str) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
        line = self._line("done")
        self.maximum_line = max(self.maximum_line, len(line))
        sys.stdout.write("\r" + line.ljust(self.maximum_line) + "\n")
        print(message, flush=True)


def main() -> int:
    parser = ArgumentParser(
        description="Step 3/4: one-pass DeepSeek Flash crop classification")
    default_corpus = (
        PROJECT_ROOT / "datasets" / "raw" / "captured" /
        "captured_yonma_20260923_005221" / "tile_crops" / "manifest.json")
    parser.add_argument(
        "corpus_manifest", nargs="?", default=str(default_corpus),
        help="Defaults to the current 32,784-crop corpus")
    parser.add_argument("--datasets", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--maximum", type=int,
                        help="Optional smoke-test limit; omit for all crops")
    parser.add_argument(
        "--proxy", default=None,
        help="Optional explicit proxy URL; omitted means forced direct connection")
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "artifacts" / "deepseek-classification.json"))
    args = parser.parse_args()

    progress_bar = ConsoleProgressBar()
    progress_bar.start()
    try:
        result = run_classification(
            args.corpus_manifest, args.datasets,
            project_root=args.project_root, batch_size=args.batch_size,
            concurrency=args.concurrency, maximum=args.maximum,
            proxy=args.proxy, progress=progress_bar.update)
    except BaseException:
        progress_bar.finish(message="classification interrupted or failed")
        raise
    progress_bar.finish(message="classification pass finished")
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report)
    print(json.dumps({**result, "report": str(report)}, ensure_ascii=False))
    return 2 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
