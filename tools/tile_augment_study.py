"""Augmentation ablation study driver for the 37-class tile classifier.

Trains ConvNeXt-Tiny (ImageNet1K_V1) under the tile_model_select protocol
(75% of backbone blocks frozen, AdamW + CosineAnnealingLR + CrossEntropy,
bf16 autocast, early stop on val acc, best state kept) while varying only the
augmentation preset (TileAugment PRESETS A/B/C/D/F) and the seed. Data comes
from a fixed split manifest; the test split is only touched when
``--evaluate-test`` is passed.

Run with an environment that has a CUDA-enabled torch, e.g.:

    D:/pScript/MiniConda/envs/bottle_yolo/python.exe tools/tile_augment_study.py \
        --split-manifest artifacts/tile-augment-20261002/split_manifest.json \
        --canvas-dir artifacts/tile-augment-20261002/canvases \
        --experiment D --seed 42 \
        --out-dir artifacts/tile-augment-20261002/runs/D_s42
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import PIL
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import tile_model_select as tms  # noqa: E402
from majsoul.perception.tile_augment import PRESETS, TileAugment  # noqa: E402

LABELS = list(tms.TILE_LABELS)
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# Directed red-five confusion pairs to report: (gt, pred) -> count.
RED_FIVE_PAIRS = [("0m", "5m"), ("5m", "0m"),
                  ("0p", "5p"), ("5p", "0p"),
                  ("0s", "5s"), ("5s", "0s")]

TOP_ERRORS = 50


def resolve_path(rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute():
        return path
    candidate = REPO_ROOT / rel
    return candidate if candidate.exists() else Path(rel)


class TrainTransform:
    def __init__(self, augment: TileAugment, rng: random.Random):
        self.augment = augment
        self.rng = rng
        self.post = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(tms.IMAGENET_MEAN, tms.IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image, sample_id=None):
        img, _ops = self.augment(img, sample_id=sample_id, rng=self.rng)
        return self.post(img)


class EvalTransform:
    def __init__(self):
        self.post = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(tms.IMAGENET_MEAN, tms.IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image, sample_id=None):
        return self.post(img)


class ManifestTileDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with Image.open(resolve_path(sample["path"])) as img:
            tensor = self.transform(img.convert("RGB"), sample["sample_id"])
        return tensor, LABEL_TO_IDX[sample["label"]], sample["sample_id"]


def make_canvas_lookup(canvas_dir: Path):
    @functools.lru_cache(maxsize=4096)
    def lookup(sample_id: str):
        path = canvas_dir / f"{sample_id}.png"
        if not path.is_file():
            return None
        with Image.open(path) as img:
            return img.convert("RGB")

    return lookup


def short_side_bucket(width: int, height: int) -> str:
    short = min(width, height)
    if short < 64:
        return "<64"
    if short <= 128:
        return "64-128"
    return ">128"


def macro_f1_from_cm(cm: np.ndarray) -> float:
    f1s = []
    for cls in range(cm.shape[0]):
        tp = int(cm[cls, cls])
        predicted = int(cm[:, cls].sum())
        support = int(cm[cls, :].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def eval_epoch(model, loader, criterion, device):
    """One deterministic pass; returns loss, acc, macro-F1."""
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.inference_mode():
        for images, targets, _sids in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            preds = logits.float().argmax(dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()
            for gt, pred in zip(targets_np, preds):
                cm[gt, pred] += 1
            total_loss += loss.item() * len(targets)
            total_acc += float((preds == targets_np).sum())
            total_n += len(targets)
    return total_loss / total_n, total_acc / total_n, macro_f1_from_cm(cm)


@torch.inference_mode()
def full_eval(model, loader, device, meta_by_id, out_dir: Path, split_name: str,
              primary: bool):
    """Full metric dump on a split: overall acc, macro-F1, per-class recall,
    confusion matrix, red-five pairs, stratified acc, top-confidence errors.

    The primary split writes error images to ``out_dir/errors/`` plus
    ``out_dir/errors.json``; the secondary split uses
    ``out_dir/errors_<split>/`` and ``errors_<split>.json``.
    """
    errors_dir = out_dir / ("errors" if primary else f"errors_{split_name}")
    errors_json_path = out_dir / ("errors.json" if primary else f"errors_{split_name}.json")
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    records = []
    for images, targets, sids in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1).cpu()
        confs, preds = probs.max(dim=1)
        top3_probs, top3_idx = probs.topk(3, dim=1)
        for i, sid in enumerate(sids):
            gt = int(targets[i])
            pred = int(preds[i])
            cm[gt, pred] += 1
            records.append({
                "sample_id": sid,
                "gt": gt,
                "pred": pred,
                "confidence": round(float(confs[i]), 6),
                "top3": [[LABELS[int(c)], round(float(p), 6)]
                         for p, c in zip(top3_probs[i], top3_idx[i])],
            })

    total = int(cm.sum())
    overall_acc = float(np.trace(cm)) / total if total else 0.0
    per_class = {}
    for cls, label in enumerate(LABELS):
        support = int(cm[cls, :].sum())
        correct = int(cm[cls, cls])
        per_class[label] = {
            "recall": round(correct / support, 4) if support else None,
            "support": support,
        }
    red_five = {f"{gt}->{pred}": int(cm[LABEL_TO_IDX[gt], LABEL_TO_IDX[pred]])
                for gt, pred in RED_FIVE_PAIRS}

    strata = {"region": defaultdict(lambda: [0, 0]),
              "orientation": defaultdict(lambda: [0, 0]),
              "short_side": defaultdict(lambda: [0, 0])}
    for rec in records:
        meta = meta_by_id[rec["sample_id"]]
        keys = {"region": meta.get("region") or "unknown",
                "orientation": meta.get("orientation") or "unknown",
                "short_side": short_side_bucket(meta["width"], meta["height"])}
        for name, key in keys.items():
            bucket = strata[name][key]
            bucket[1] += 1
            bucket[0] += int(rec["pred"] == rec["gt"])
    stratified = {name: {key: {"acc": round(c / n, 4) if n else None, "support": n}
                         for key, (c, n) in sorted(buckets.items())}
                  for name, buckets in strata.items()}

    errors = [r for r in records if r["pred"] != r["gt"]]
    errors.sort(key=lambda r: r["confidence"], reverse=True)
    top_errors = errors[:TOP_ERRORS]
    errors_json = []
    errors_dir.mkdir(parents=True, exist_ok=True)
    for rank, rec in enumerate(top_errors):
        meta = meta_by_id[rec["sample_id"]]
        src = resolve_path(meta["path"])
        dest_name = f"{rank:02d}_{rec['sample_id']}.png"
        if src.exists():
            shutil.copy2(src, errors_dir / dest_name)
        errors_json.append({
            "rank": rank,
            "sample_id": rec["sample_id"],
            "path": meta["path"],
            "error_image": f"{errors_dir.name}/{dest_name}",
            "gt": LABELS[rec["gt"]],
            "pred": LABELS[rec["pred"]],
            "confidence": rec["confidence"],
            "top3": rec["top3"],
        })
    errors_json_path.write_text(
        json.dumps(errors_json, ensure_ascii=False, indent=1), encoding="utf-8")

    return {
        "overall_acc": round(overall_acc, 5),
        "macro_f1": round(macro_f1_from_cm(cm), 5),
        "support": total,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "red_five_confusions": red_five,
        "stratified": stratified,
        "top_errors": errors_json,
        "errors_json": str(errors_json_path),
    }


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest",
                        default="artifacts/tile-augment-20261002/split_manifest.json")
    parser.add_argument("--canvas-dir",
                        default="artifacts/tile-augment-20261002/canvases")
    parser.add_argument("--experiment", required=True, choices=sorted(PRESETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0,
                        help="Windows: workers>0 is known to deadlock")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="also evaluate on the test split (model selection "
                             "must use val only; leave off during selection)")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    augment_rng = random.Random(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas_dir = Path(args.canvas_dir)

    split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    samples = split["samples"]
    unknown_labels = sorted({s["label"] for s in samples} - set(LABELS))
    if unknown_labels:
        raise SystemExit(f"labels not in TILE_LABELS: {unknown_labels}")
    meta_by_id = {s["sample_id"]: s for s in samples}
    splits = {name: [s for s in samples if s["split"] == name]
              for name in ("train", "val", "test")}
    if not splits["train"] or not splits["val"]:
        raise SystemExit("split manifest must contain non-empty train and val splits")

    augment = TileAugment(PRESETS[args.experiment],
                          canvas_lookup=make_canvas_lookup(canvas_dir))
    train_ds = ManifestTileDataset(splits["train"], TrainTransform(augment, augment_rng))
    eval_tf = EvalTransform()
    val_ds = ManifestTileDataset(splits["val"], eval_tf)
    test_ds = ManifestTileDataset(splits["test"], eval_tf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} "
          f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}", flush=True)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                         pin_memory=device.type == "cuda",
                         persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed),
                              **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model, param_info = tms.build_model("convnext_tiny", NUM_CLASSES)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc, best_epoch, epochs_without_improve = -1.0, -1, 0
    best_state = None
    history = []
    early_stopped = False
    started = time.perf_counter()
    for epoch in range(args.epochs):
        train_loss, train_acc = tms.run_epoch(
            model, _strip_sids(train_loader), criterion, optimizer, device, True)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "train_acc": round(train_acc, 5), "val_loss": round(val_loss, 5),
                        "val_acc": round(val_acc, 5), "val_macro_f1": round(val_f1, 5)})
        print(f"[{args.experiment}/s{args.seed}] epoch {epoch + 1}/{args.epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}", flush=True)
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            epochs_without_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"early stop at epoch {epoch + 1}", flush=True)
                early_stopped = True
                break
    train_seconds = time.perf_counter() - started

    model.load_state_dict(best_state)
    model = model.to(device)
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({"model_name": "convnext_tiny", "labels": LABELS,
                "state_dict": best_state, "best_val_acc": best_val_acc,
                "experiment": args.experiment, "seed": args.seed}, ckpt_path)

    evaluate_test = args.evaluate_test and bool(splits["test"])
    print("evaluating best state on val ...", flush=True)
    val_metrics = full_eval(model, val_loader, device, meta_by_id, out_dir,
                            "val", primary=not evaluate_test)
    if evaluate_test:
        print("evaluating best state on test (--evaluate-test) ...", flush=True)
        test_metrics = full_eval(model, test_loader, device, meta_by_id, out_dir,
                                 "test", primary=True)
    else:
        test_metrics = None

    gpu_latency_ms = tms.measure_latency(model, device) if device.type == "cuda" else None
    cpu_model = model.to("cpu")
    cpu_latency_ms = tms.measure_latency(cpu_model, torch.device("cpu"), warmup=5, iters=30)
    del cpu_model

    canvas_coverage = (
        sum(1 for s in splits["train"] if (canvas_dir / f"{s['sample_id']}.png").is_file())
        / len(splits["train"])
    )

    result = {
        "experiment": args.experiment,
        "seed": args.seed,
        "config": {
            "model": "convnext_tiny",
            "pretrained": "ImageNet1K_V1",
            "backbone_freeze_ratio": tms.BACKBONE_FREEZE_RATIO,
            "param_info": param_info,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": "CosineAnnealingLR",
            "loss": "CrossEntropyLoss",
            "augment_preset": args.experiment,
            "augment_config": asdict(PRESETS[args.experiment]),
        },
        "data": {
            "split_manifest": args.split_manifest,
            "counts": {name: len(ss) for name, ss in splits.items()},
            "label_histogram": {name: dict(sorted(Counter(s["label"] for s in ss).items()))
                                for name, ss in splits.items()},
        },
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "early_stopped": early_stopped,
        "train_seconds": round(train_seconds, 1),
        "history": history,
        "val": val_metrics,
        "test": test_metrics if test_metrics is not None else {
            "note": "not evaluated (selection on val only)"},
        "latency_ms_batch1": {"gpu": round(gpu_latency_ms, 3) if gpu_latency_ms else None,
                              "cpu": round(cpu_latency_ms, 3)},
        "environment": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "PIL": PIL.__version__,
            "cv2": cv2.__version__,
        },
        "git_commit": git_commit(),
        "argv": sys.argv,
        "canvas_coverage_train": round(canvas_coverage, 4),
        "canvas_dir": str(canvas_dir),
        "timestamp": datetime.now().astimezone().isoformat(),
        "checkpoint": str(ckpt_path),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done: best_epoch={best_epoch} best_val_acc={best_val_acc:.4f} "
          f"val_acc={val_metrics['overall_acc']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
          + (f" test_acc={test_metrics['overall_acc']:.4f}" if test_metrics else "")
          + f" -> {out_dir / 'result.json'}", flush=True)


def _strip_sids(loader):
    """Adapt (image, target, sample_id) batches to tile_model_select.run_epoch."""
    for images, targets, _sids in loader:
        yield images, targets


if __name__ == "__main__":
    main()
