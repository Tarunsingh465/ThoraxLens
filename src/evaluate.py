from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from .data import load_metadata, read_image_list, ChestXrayDataset
from .model import load_checkpoint
from .labels import LABELS


def main():
    p = argparse.ArgumentParser(description="Evaluate per-label AUROC and average precision")
    p.add_argument("--csv", required=True); p.add_argument("--image-dir", required=True)
    p.add_argument("--test-list", required=True, help="Image list for evaluation; do not use it during training or tuning")
    p.add_argument("--exclude-patient-overlap-with-list", default=None,
                   help="Optional train/validation image list. All test images whose Patient ID appears in that list are excluded.")
    p.add_argument("--checkpoint", required=True); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=None); p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--workers", type=int, default=2)
    p.add_argument("--output", default="reports/per_label_metrics.csv")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_metadata(args.csv, image_list=args.test_list)

    if args.exclude_patient_overlap_with_list:
        train_names = read_image_list(args.exclude_patient_overlap_with_list)
        all_meta = pd.read_csv(args.csv, usecols=["Image Index", "Patient ID"])
        train_patients = set(all_meta.loc[all_meta["Image Index"].astype(str).isin(train_names), "Patient ID"].astype(str))
        before = len(frame)
        frame = frame[~frame["Patient ID"].astype(str).isin(train_patients)].copy().reset_index(drop=True)
        print(f"Patient-overlap exclusion: removed {before-len(frame):,} test images; retained {len(frame):,} patient-disjoint test images")
    if args.limit and len(frame) > args.limit:
        frame = frame.sample(args.limit, random_state=args.seed).sort_index().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No evaluation images remain. Check test list and patient-overlap exclusion.")
    missing = [name for name in frame["Image Index"] if not (Path(args.image_dir) / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} evaluation images are missing. First: {missing[0]}")

    model, meta = load_checkpoint(args.checkpoint, device)
    image_size = args.image_size or int(meta.get("image_size", 224))
    norm = meta.get("input_normalization", "imagenet")
    loader = DataLoader(
        ChestXrayDataset(frame, args.image_dir, image_size, train=False, normalization=norm),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    print(f"Evaluation images: {len(frame):,} | device: {device} | backbone: {meta.get('backbone','imagenet')} | size: {image_size}")
    ys, scores = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            scores.append(torch.sigmoid(logits.float()).cpu().numpy()); ys.append(y.numpy())
    y_true, y_score = np.concatenate(ys), np.concatenate(scores)
    rows = []
    for i, label in enumerate(LABELS):
        auc = roc_auc_score(y_true[:, i], y_score[:, i]) if np.unique(y_true[:, i]).size == 2 else np.nan
        ap = average_precision_score(y_true[:, i], y_score[:, i]) if y_true[:, i].sum() else np.nan
        rows.append({"label": label, "positive_count": int(y_true[:, i].sum()),
                     "prevalence": float(y_true[:, i].mean()), "AUROC": float(auc), "average_precision": float(ap)})
    out = pd.DataFrame(rows)
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True); out.to_csv(out_path, index=False)
    print(out.to_string(index=False)); print(f"Wrote {out_path}")
    if np.isfinite(out["AUROC"]).any(): print(f"Macro AUROC: {out['AUROC'].mean(skipna=True):.4f}")
    if np.isfinite(out["average_precision"]).any(): print(f"Macro average precision: {out['average_precision'].mean(skipna=True):.4f}")

if __name__ == "__main__": main()
