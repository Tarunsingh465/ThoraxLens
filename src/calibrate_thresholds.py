"""Choose per-label probability-score thresholds on the training pool's validation split.

This script deliberately does not load or inspect the held-out test list. It reconstructs
train.py's patient-level split (default validation_fraction=0.15, seed=42), runs the
checkpoint on the validation patients, and selects one threshold per label that maximizes
validation F1. These thresholds are for research/education only and are not clinical cutoffs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from .data import ChestXrayDataset, load_metadata, patient_split
from .labels import LABELS
from .model import load_checkpoint


def threshold_for_label(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict]:
    """Return threshold maximizing F1, with safe fallback for single-class validation labels."""
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return 0.5, {
            "threshold": 0.5,
            "validation_positive_count": positives,
            "validation_negative_count": negatives,
            "validation_f1_at_threshold": float(f1_score(y_true, y_score >= 0.5, zero_division=0)),
            "validation_precision_at_threshold": float("nan"),
            "validation_recall_at_threshold": float("nan"),
            "validation_auroc": float("nan"),
            "validation_average_precision": float("nan"),
            "threshold_method": "fallback_0.5_single_class_validation",
        }

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        threshold = 0.5
        method = "fallback_0.5_no_thresholds"
    else:
        f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        best_idx = int(np.nanargmax(f1))
        threshold = float(thresholds[best_idx])
        method = "validation_max_f1"

    pred = y_score >= threshold
    return threshold, {
        "threshold": threshold,
        "validation_positive_count": positives,
        "validation_negative_count": negatives,
        "validation_f1_at_threshold": float(f1_score(y_true, pred, zero_division=0)),
        "validation_precision_at_threshold": float(np.sum(pred & (y_true == 1)) / max(int(pred.sum()), 1)),
        "validation_recall_at_threshold": float(np.sum(pred & (y_true == 1)) / max(positives, 1)),
        "validation_auroc": float(roc_auc_score(y_true, y_score)),
        "validation_average_precision": float(average_precision_score(y_true, y_score)),
        "threshold_method": method,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate per-label thresholds on the patient-level validation split")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--train-val-list", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", default="reports/validation_thresholds.json")
    parser.add_argument("--output-csv", default="reports/validation_threshold_metrics.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be >= 1 and --workers must be >= 0")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_metadata(args.csv, image_list=args.train_val_list)
    _, valid_frame = patient_split(frame, validation_fraction=args.validation_fraction, seed=args.seed)
    model, meta = load_checkpoint(args.checkpoint, device)
    image_size = int(meta.get("image_size", 224))
    normalization = meta.get("input_normalization", "imagenet")

    dataset = ChestXrayDataset(valid_frame, args.image_dir, image_size, train=False, normalization=normalization)
    loader_kwargs = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    if args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    use_amp = bool(args.amp and device.type == "cuda")
    print(f"Validation images: {len(valid_frame):,} | device: {device} | checkpoint: {args.checkpoint}")
    print(f"Validation split: patient-level, fraction={args.validation_fraction}, seed={args.seed} | AMP: {use_amp}")

    truths, scores = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
            truths.append(y.numpy())
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())

    y_true = np.concatenate(truths, axis=0).astype(np.int32)
    y_score = np.concatenate(scores, axis=0).astype(np.float64)
    thresholds: dict[str, float] = {}
    rows = []
    for i, label in enumerate(LABELS):
        threshold, metrics = threshold_for_label(y_true[:, i], y_score[:, i])
        thresholds[label] = threshold
        rows.append({"label": label, **metrics})

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Per-label score thresholds selected by maximizing F1 on this project's patient-level validation split. Research-only; not clinical decision thresholds.",
        "checkpoint": str(args.checkpoint),
        "validation_images": int(len(valid_frame)),
        "validation_fraction": float(args.validation_fraction),
        "seed": int(args.seed),
        "method": "per-label maximum F1 on validation precision-recall curve",
        "labels": LABELS,
        "thresholds": thresholds,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(pd.DataFrame(rows)[["label", "threshold", "validation_f1_at_threshold", "validation_auroc", "validation_average_precision"]].to_string(index=False))
    print(f"Wrote thresholds: {json_path}")
    print(f"Wrote validation metrics: {csv_path}")
    print("Do not use these validation metrics as final test performance; keep the held-out test set untouched.")


if __name__ == "__main__":
    main()
