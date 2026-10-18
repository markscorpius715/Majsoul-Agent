"""Export misclassified test samples for a trained tile classifier checkpoint.

For every test-set error writes the crop image plus GT / prediction /
confidence / top-3 probabilities, as JSON + CSV + a contact sheet.

    D:/pScript/MiniConda/envs/bottle_yolo/python.exe tools/export_tile_errors.py \
        --checkpoint artifacts/model-select-20260929/convnext_tiny_best.pt \
        --split artifacts/model-select-20260929/split_manifest.json \
        --data-dir artifacts/ai-and-human-reviewed \
        --out-dir artifacts/model-select-20260929/errors_convnext_tiny
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tile_model_select import build_model, build_transforms  # noqa: E402


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    labels = ckpt["labels"]
    model, _ = build_model(ckpt["model_name"], len(labels))
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    _, eval_tf = build_transforms()
    data_dir = Path(args.data_dir)
    test_paths = json.loads(Path(args.split).read_text(encoding="utf-8"))["test"]

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    paths = test_paths
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start:start + args.batch_size]
        tensors = []
        for rel in batch_paths:
            with Image.open(data_dir / rel) as img:
                tensors.append(eval_tf(img.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        probs = torch.softmax(model(batch).float(), dim=1).cpu()
        for rel, prob in zip(batch_paths, probs):
            gt = Path(rel).parent.name
            top3_prob, top3_idx = prob.topk(3)
            top3 = [(labels[i], round(float(p), 5)) for p, i in zip(top3_prob, top3_idx)]
            pred, conf = top3[0]
            if pred != gt:
                errors.append({
                    "image": rel,
                    "gt": gt,
                    "prediction": pred,
                    "confidence": conf,
                    "top3": top3,
                })

    records = []
    for i, err in enumerate(errors):
        src = data_dir / err["image"]
        name = f"err{i:03d}_gt-{err['gt']}_pred-{err['prediction']}{src.suffix}"
        shutil.copy2(src, images_dir / name)
        records.append({**err, "exported_image": f"images/{name}"})

    (out_dir / "errors.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out_dir / "errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "gt", "prediction", "confidence",
                         "top1", "top1_prob", "top2", "top2_prob", "top3", "top3_prob"])
        for rec in records:
            top3 = rec["top3"] + [("", "")] * (3 - len(rec["top3"]))
            writer.writerow([rec["exported_image"], rec["gt"], rec["prediction"],
                             rec["confidence"],
                             *(x for pair in top3 for x in pair)])

    if records:
        thumb = 160
        cols = min(6, len(records))
        rows = (len(records) + cols - 1) // cols
        label_h = 52
        sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(sheet)
        for i, rec in enumerate(records):
            with Image.open(out_dir / rec["exported_image"]) as img:
                img = img.convert("RGB").resize((thumb, thumb))
            x, y = (i % cols) * thumb, (i // cols) * (thumb + label_h)
            sheet.paste(img, (x, y))
            lines = [f"GT:{rec['gt']} P:{rec['prediction']} {rec['confidence']:.3f}",
                     " ".join(f"{l}:{p:.2f}" for l, p in rec["top3"])]
            draw.text((x + 4, y + thumb + 4), "\n".join(lines), fill="black")
        sheet.save(out_dir / "contact_sheet.png")

    print(f"test={len(test_paths)} errors={len(records)} -> {out_dir}", flush=True)
    for rec in records:
        print(json.dumps(rec, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
