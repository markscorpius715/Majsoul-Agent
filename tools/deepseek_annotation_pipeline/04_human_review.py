"""Step 4: review DeepSeek results with 1=Pass and 2=Reject."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys


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

from majsoul.continual.corpus_label_gui import run_corpus_label_gui


def main() -> int:
    parser = ArgumentParser(
        description="Step 4/4: human confirmation of AI-reviewed crops")
    default_corpus = (
        PROJECT_ROOT / "datasets" / "raw" / "captured" /
        "captured_yonma_20260923_005221" / "tile_crops" / "manifest.json")
    parser.add_argument(
        "corpus_manifest", nargs="?", default=str(default_corpus),
        help="Defaults to the current 32,784-crop corpus")
    parser.add_argument("--datasets", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument(
        "--cases", default=str(PROJECT_ROOT / "artifacts" / "unknown_cases"))
    parser.add_argument("--reviewer", default="human")
    parser.add_argument("--all-candidates", action="store_true",
                        help="Include candidates not yet visually reviewed by AI")
    args = parser.parse_args()
    run_corpus_label_gui(
        args.corpus_manifest, args.datasets,
        cases_root=args.cases, reviewer=args.reviewer,
        ai_reviewed_only=not args.all_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
