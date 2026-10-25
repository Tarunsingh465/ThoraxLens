from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .data import ChestXrayDataset, load_metadata, read_image_list
from .labels import LABELS
from .model import load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate validation-selected per-label thresholds on a held-out test set."
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--test-list", required=True)
    p.add_argument("--exclude-patient-overlap-with-list", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--thresholds-json", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--output", default="reports/patient_disjoint_test_threshold_metrics.csv")
    args = p.parse_args()

    if args.batch_size < 1 or args.workers < 0:
        p.error("--batch-size must be >= 1 and --workers must be >= 0")

    thresholds_path = Path(args.thresholds_json)
    if not thresholds_path.is_file():
        raise FileNotFoundError(f"Threshold JSON not found: {thresholds_path}")
    threshold_payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    threshold_map = threshold_payload.get("thresholds", threshold_payload)
    missing_thresholds = [label for label in LABELS if label not in threshold_map]
    if missing_thresholds:
        raise ValueError(f"Threshold JSON is missing labels: {missing_thresholds}")
    thresholds = np.array([float(threshold_map[label]) for label in LABELS], dtype=np.float64)
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("Threshold JSON contains non-finite values")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_metadata(args.csv, image_list=args.test_list)

    if args.exclude_patient_overlap_with_list:
        train_names = read_image_list(args.exclude_patient_overlap_with_list)
        metadata = pd.read_csv(args.csv, usecols=["Image Index", "Patient ID"])
        metadata["Image Index"] = metadata["Image Index"].astype(str)
        train_patients = set(
            metadata.loc[metadata["Image Index"].isin(train_names), "Patient ID"].astype(str)
        )
        before = len(frame)
        frame = frame[~frame["Patient ID"].astype(str).isin(train_patients)].copy().reset_index(drop=True)
        print(
            f"Patient-overlap exclusion: removed {before - len(frame):,} test images; "
            f"retained {len(frame):,} patient-disjoint test images"
        )

    if args.limit is not None and args.limit > 0 and len(frame) > args.limit:
        frame = frame.sample(args.limit, random_state=args.seed).sort_index().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No test images remain after filters")

    model, meta = load_checkpoint(args.checkpoint, device)
    image_size = args.image_size or int(meta.get("image_size", 224))
    normalization = meta.get("input_normalization", "imagenet")
    dataset = ChestXrayDataset(
        frame, args.image_dir, image_size, train=False, normalization=normalization
    )
    loader_kwargs = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    if args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    use_amp = bool(args.amp and device.type == "cuda")
    print(
        f"Test images: {len(frame):,} | device: {device} | backbone: {meta.get('backbone', 'imagenet')} "
        f"| image size: {image_size} | AMP: {use_amp}"
    )
    print("Thresholds loaded from validation-calibration JSON; test labels are not used to tune thresholds.")

    truths: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
            truths.append(y.cpu().numpy())
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())

    y_true = np.concatenate(truths, axis=0).astype(np.int32)
    y_score = np.concatenate(scores, axis=0).astype(np.float64)
    y_pred = (y_score >= thresholds[None, :]).astype(np.int32)

    rows = []
    for i, label in enumerate(LABELS):
        truth, score, pred = y_true[:, i], y_score[:, i], y_pred[:, i]
        tp = int(np.sum((truth == 1) & (pred == 1)))
        tn = int(np.sum((truth == 0) & (pred == 0)))
        fp = int(np.sum((truth == 0) & (pred == 1)))
        fn = int(np.sum((truth == 1) & (pred == 0)))
        auc = float(roc_auc_score(truth, score)) if np.unique(truth).size == 2 else float("nan")
        ap = float(average_precision_score(truth, score)) if truth.sum() > 0 else float("nan")
        rows.append({
            "label": label,
            "positive_count": int(truth.sum()),
            "prevalence": float(truth.mean()),
            "threshold_from_validation": float(thresholds[i]),
            "AUROC": auc,
            "average_precision": ap,
            "test_precision_at_threshold": float(precision_score(truth, pred, zero_division=0)),
            "test_recall_at_threshold": float(recall_score(truth, pred, zero_division=0)),
            "test_f1_at_threshold": float(f1_score(truth, pred, zero_division=0)),
            "test_accuracy_at_threshold": float(accuracy_score(truth, pred)),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out[[
        "label", "threshold_from_validation", "test_precision_at_threshold",
        "test_recall_at_threshold", "test_f1_at_threshold", "AUROC", "average_precision"
    ]].to_string(index=False))
    print(f"Wrote thresholded test metrics: {out_path}")
    print(f"Macro AUROC: {out['AUROC'].mean(skipna=True):.4f}")
    print(f"Macro average precision: {out['average_precision'].mean(skipna=True):.4f}")
    print(f"Macro precision at validation thresholds: {out['test_precision_at_threshold'].mean():.4f}")
    print(f"Macro recall at validation thresholds: {out['test_recall_at_threshold'].mean():.4f}")
    print(f"Macro F1 at validation thresholds: {out['test_f1_at_threshold'].mean():.4f}")
    print("Research/education evaluation only; thresholds are not clinical cutoffs.")


if __name__ == "__main__":
    main()
