"""Launch the complete offline human-in-the-loop image labeler."""

from argparse import ArgumentParser

from majsoul.continual.label_gui import run_label_gui
from majsoul.continual.corpus_label_gui import (
    discover_crop_manifests, run_corpus_label_gui)
from majsoul.continual.schemas import ReviewKind


def main():
    parser = ArgumentParser(description="Open the manual annotation interface for Mahjong Soul all images to be annotated")
    parser.add_argument("--cases", default="artifacts/unknown_cases")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--reviewer", default="human_gui")
    parser.add_argument(
        "--corpus",
        help="Raw crop corpus manifest; when omitted, automatically discover the complete corpus under datasets")
    parser.add_argument(
        "--cases-only", action="store_true",
        help="Only open a small number of unknown cases, do not open the raw crop corpus")
    parser.add_argument(
        "--kind", default="tile",
        choices=("all", *(item.value for item in ReviewKind)))
    args = parser.parse_args()
    if not args.cases_only:
        manifests = ((args.corpus,) if args.corpus else
                     discover_crop_manifests(args.datasets))
        if manifests:
            run_corpus_label_gui(
                manifests[0], args.datasets,
                cases_root=args.cases, reviewer=args.reviewer)
            return
    run_label_gui(
        args.cases, args.datasets,
        reviewer=args.reviewer, initial_kind=args.kind)


if __name__ == "__main__":
    main()
