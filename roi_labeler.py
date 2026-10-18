r"""Convenient launcher for the offline ROI GUI.

Run from the project root:
    .venv\Scripts\python roi_labeler.py
"""

from argparse import ArgumentParser

from majsoul.calibration.roi_gui import run_roi_gui


def main():
    parser = ArgumentParser(description="Open Mahjong Soul screenshots and interactively annotate fixed ROIs")
    parser.add_argument("image", nargs="?", default="artifacts/roi/reference.png",
                        help="Reference screenshot; default artifacts/roi/reference.png")
    parser.add_argument("--config", default="configs/roi.yaml", help="ROI YAML save path")
    parser.add_argument("--names", nargs="+", help="Only display and edit the specified ROI")
    args = parser.parse_args()
    run_roi_gui(args.image, args.config, names=args.names)


if __name__ == "__main__":
    main()
