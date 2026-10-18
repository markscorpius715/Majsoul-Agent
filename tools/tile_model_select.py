"""Model-selection benchmark for the 37-class majsoul tile classifier.

Trains MobileNetV3-Large / EfficientNet-B0 / ConvNeXt-Tiny (ImageNet
pretrained) under identical conditions:

- identical stratified train/val/test split (persisted to disk)
- 224x224 input, identical augmentation
- AdamW + cosine schedule + CrossEntropy
- classifier head + last ~25% of backbone blocks trainable, rest frozen
- identical early stopping on validation accuracy

The dataset is the merged reviewed tile corpus
(``artifacts/ai-and-human-reviewed``). Per user decision on 2026-09-29 this
benchmark may consume AI-reviewed labels as training truth for model
selection only; the result is a candidate baseline and must not be
promoted as a human-gold model.

Run with an environment that has a CUDA-enabled torch, e.g.:

    D:/pScript/MiniConda/envs/bottle_yolo/python.exe tools/tile_model_select.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

TILE_LABELS = tuple(
    [f"{rank}{suit}" for suit in "mps" for rank in "123456789"]
    + [f"0{suit}" for suit in "mps"]
    + [f"{rank}z" for rank in "1234567"]
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Fraction of top-level backbone blocks kept frozen; the classifier head and
# the remaining ~25% of blocks are trained. Same rule for every model.
BACKBONE_FREEZE_RATIO = 0.75

MODEL_SPECS = {
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "efficientnet_b0": models.efficientnet_b0,
    "convnext_tiny": models.convnext_tiny,
}


def stratified_split(labels, seed, val_frac=0.15, test_frac=0.15):
    """Deterministic stratified train/val/test split.

    Every class with >= 3 samples contributes at least one sample to val and
    test so per-class metrics stay defined for rare classes.
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)
    train_idx, val_idx, test_idx = [], [], []
    for label in sorted(by_label):
        indices = by_label[label]
        rng.shuffle(indices)
        n = len(indices)
        n_test = max(1, round(n * test_frac)) if n >= 3 else 0
        n_val = max(1, round(n * val_frac)) if n - n_test >= 2 else 0
        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test:n_test + n_val])
        train_idx.extend(indices[n_test + n_val:])
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


class TileDataset(Dataset):
    def __init__(self, root: Path, samples, transform):
        self.root = root
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label_idx = self.samples[idx]
        with Image.open(self.root / rel_path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, label_idx


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def _feature_blocks(model_name: str, model: nn.Module) -> list[nn.Module]:
    # All three torchvision backbones expose the feature extractor as a
    # top-level Sequential named ``features``.
    return list(model.features.children())


def build_model(model_name: str, num_classes: int):
    model = MODEL_SPECS[model_name](weights="IMAGENET1K_V1" if model_name != "convnext_tiny" else "IMAGENET1K_V1")
    blocks = _feature_blocks(model_name, model)
    n_freeze = round(len(blocks) * BACKBONE_FREEZE_RATIO)
    n_freeze = min(max(n_freeze, 1), len(blocks) - 1)
    for block in blocks[:n_freeze]:
        for param in block.parameters():
            param.requires_grad_(False)

    if model_name in ("mobilenet_v3_large", "efficientnet_b0"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "convnext_tiny":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:  # pragma: no cover - guarded by MODEL_SPECS
        raise ValueError(model_name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, {"total_params": total, "trainable_params": trainable,
                   "backbone_blocks": len(blocks), "frozen_blocks": n_freeze}


def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            total_acc += accuracy(logits.float(), targets) * len(targets)
            total_n += len(targets)
    return total_loss / total_n, total_acc / total_n


@torch.inference_mode()
def evaluate_per_class(model, loader, device, num_classes):
    model.eval()
    correct = torch.zeros(num_classes, dtype=torch.long)
    support = torch.zeros(num_classes, dtype=torch.long)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(images)
        preds = logits.float().argmax(dim=1).cpu()
        for cls in range(num_classes):
            mask = targets == cls
            support[cls] += mask.sum()
            correct[cls] += (preds[mask] == cls).sum()
    return correct, support


@torch.inference_mode()
def measure_latency(model, device, batch_size=1, warmup=20, iters=100):
    model.eval()
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)
    for _ in range(warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1000.0  # ms per batch


def train_model(model_name, samples_by_idx, label_names, out_dir: Path, args, device):
    train_tf, eval_tf = build_transforms()
    root = Path(args.data_dir)
    train_ds = TileDataset(root, samples_by_idx["train"], train_tf)
    val_ds = TileDataset(root, samples_by_idx["val"], eval_tf)
    test_ds = TileDataset(root, samples_by_idx["test"], eval_tf)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                         pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model, param_info = build_model(model_name, len(label_names))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc, best_epoch, epochs_without_improve = -1.0, -1, 0
    best_state = None
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "train_acc": round(train_acc, 5), "val_loss": round(val_loss, 5),
                        "val_acc": round(val_acc, 5)})
        print(f"[{model_name}] epoch {epoch + 1}/{args.epochs} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            epochs_without_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"[{model_name}] early stop at epoch {epoch + 1}", flush=True)
                break
    train_seconds = time.perf_counter() - started

    model.load_state_dict(best_state)
    model = model.to(device)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, False)
    correct, support = evaluate_per_class(model, test_loader, device, len(label_names))
    per_class = {
        label_names[i]: {
            "correct": int(correct[i]), "support": int(support[i]),
            "acc": round(int(correct[i]) / int(support[i]), 4) if support[i] else None,
        }
        for i in range(len(label_names))
    }

    gpu_latency_ms = measure_latency(model, device) if device.type == "cuda" else None
    cpu_model = model.to("cpu")
    cpu_latency_ms = measure_latency(cpu_model, torch.device("cpu"), warmup=5, iters=30)
    del cpu_model
    if device.type == "cuda":
        model = model.to(device)

    ckpt_path = out_dir / f"{model_name}_best.pt"
    torch.save({"model_name": model_name, "labels": label_names,
                "state_dict": best_state, "test_acc": test_acc}, ckpt_path)

    return {
        "model": model_name,
        "params": param_info,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 5),
        "test_acc": round(test_acc, 5),
        "test_loss": round(test_loss, 5),
        "epochs_ran": len(history),
        "train_seconds": round(train_seconds, 1),
        "gpu_latency_ms_batch1": round(gpu_latency_ms, 3) if gpu_latency_ms else None,
        "cpu_latency_ms_batch1": round(cpu_latency_ms, 3),
        "per_class_test": per_class,
        "history": history,
        "checkpoint": str(ckpt_path),
    }


def collect_samples(data_dir: Path):
    tiles_root = data_dir / "tiles"
    labels = sorted(p.name for p in tiles_root.iterdir() if p.is_dir())
    if set(labels) != set(TILE_LABELS):
        raise SystemExit(f"label mismatch: {sorted(set(TILE_LABELS) - set(labels))} missing")
    samples = []
    for idx, label in enumerate(TILE_LABELS):
        for png in sorted((tiles_root / label).glob("*.png")):
            samples.append((str(png.relative_to(data_dir)), idx))
    return TILE_LABELS, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="artifacts/ai-and-human-reviewed")
    parser.add_argument("--out-dir", default="artifacts/model-select-20260929")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="debug: cap total samples (stratified subset)")
    parser.add_argument("--models", nargs="*", default=list(MODEL_SPECS),
                        choices=list(MODEL_SPECS))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_names, samples = collect_samples(Path(args.data_dir))
    if args.max_samples and args.max_samples < len(samples):
        rng = random.Random(args.seed)
        by_label = defaultdict(list)
        for sample in samples:
            by_label[sample[1]].append(sample)
        per = max(1, args.max_samples // len(label_names))
        samples = [s for cls in sorted(by_label) for s in rng.sample(by_label[cls], min(per, len(by_label[cls])))]
        samples.sort()

    label_idx_of_sample = [label for _, label in samples]
    train_idx, val_idx, test_idx = stratified_split(label_idx_of_sample, args.seed)
    samples_by_idx = {
        "train": [samples[i] for i in train_idx],
        "val": [samples[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
    }
    split_manifest = {
        "seed": args.seed, "total": len(samples),
        "counts": {k: len(v) for k, v in samples_by_idx.items()},
        "train": [s[0] for s in samples_by_idx["train"]],
        "val": [s[0] for s in samples_by_idx["val"]],
        "test": [s[0] for s in samples_by_idx["test"]],
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"samples={len(samples)} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}", flush=True)

    config = {
        "data_dir": str(args.data_dir),
        "models": args.models, "input_size": 224,
        "epochs": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "optimizer": "AdamW",
        "lr": args.lr, "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR", "loss": "CrossEntropyLoss",
        "pretrained": "ImageNet1K_V1",
        "backbone_freeze_ratio": BACKBONE_FREEZE_RATIO,
        "seed": args.seed,
        "torch": torch.__version__, "device": str(device),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    results = []
    for model_name in args.models:
        print(f"=== {model_name} ===", flush=True)
        result = train_model(model_name, samples_by_idx, list(label_names), out_dir, args, device)
        results.append(result)
        (out_dir / f"result_{model_name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = {
        "config": config,
        "results": [
            {k: v for k, v in r.items() if k not in ("history", "per_class_test")}
            for r in results
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary["results"], indent=1), flush=True)


if __name__ == "__main__":
    main()
