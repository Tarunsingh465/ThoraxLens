from __future__ import annotations
import argparse, csv, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from .data import load_metadata, patient_split, ChestXrayDataset
from .model import build_model
from .labels import LABELS


def reference_weighted_bce(logits, targets, pos_weights, neg_weights):
    """Coursera/reference-style weighted BCE: balance positive and negative contribution per label."""
    positive = F.softplus(-logits) * targets * pos_weights
    negative = F.softplus(logits) * (1.0 - targets) * neg_weights
    # Sum label-wise batch means, matching the reference implementation's scale.
    return (positive + negative).mean(dim=0).sum()


def validation_metrics(model, loader, device, use_amp, pos_weights, neg_weights):
    model.eval()
    total_loss, count = 0.0, 0
    ys, probs = [], []
    with torch.inference_mode():
        for x, target in tqdm(loader, desc="validation", leave=False):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss = reference_weighted_bce(logits.float(), target, pos_weights, neg_weights)
            total_loss += loss.item() * x.size(0)
            count += x.size(0)
            ys.append(target.cpu().numpy())
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    y_true, y_score = np.concatenate(ys), np.concatenate(probs)
    aucs, aps, rows = [], [], []
    for i, label in enumerate(LABELS):
        auc = np.nan
        ap = np.nan
        if np.unique(y_true[:, i]).size == 2:
            auc = float(roc_auc_score(y_true[:, i], y_score[:, i]))
            aucs.append(auc)
        if y_true[:, i].sum() > 0:
            ap = float(average_precision_score(y_true[:, i], y_score[:, i]))
            aps.append(ap)
        rows.append({"label": label, "positive_count": int(y_true[:, i].sum()),
                     "AUROC": auc, "average_precision": ap})
    return total_loss / max(count, 1), float(np.mean(aucs)) if aucs else float("nan"), \
        float(np.mean(aps)) if aps else float("nan"), rows


def main():
    p = argparse.ArgumentParser(description="Train DenseNet121 using the reference project's weighted BCE and validation callbacks")
    p.add_argument("--csv", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--train-val-list", default=None, help="Optional NIH list. Leave out to use all CSV rows (audit patient split before evaluation).")
    p.add_argument("--output", default="models/densenet121_chestxray.pth")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4, help="Windows: start with 2-4 workers; use 0 if multiprocessing causes problems")
    p.add_argument("--amp", action="store_true", help="Mixed precision on CUDA")
    p.add_argument("--backbone", choices=["imagenet", "xrv-nih"], default="xrv-nih",
                   help="xrv-nih starts from medical-domain pretrained weights; imagenet is compatible with the earlier project checkpoint")
    p.add_argument("--init-checkpoint", default=None,
                   help="Initialize from an earlier project's checkpoint (same backbone only; optimizer state starts fresh)")
    p.add_argument("--resume", default=None, help="Resume a v3 checkpoint including optimizer/scheduler/scaler state")
    p.add_argument("--patience", type=int, default=3, help="Early-stopping patience on validation AUROC")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32 or args.workers < 0:
        p.error("epochs/batch-size/image-size must be positive and workers must be >= 0")
    if args.backbone == "xrv-nih" and args.image_size != 224:
        p.error("The xrv-nih pretrained backbone expects 224x224 inputs. Use --image-size 224.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | backbone: {args.backbone} | CUDA AMP: {args.amp and device.type == 'cuda'}")
    if device.type == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")

    frame = load_metadata(args.csv, args.limit, args.seed, image_list=args.train_val_list)
    train_df, valid_df = patient_split(frame, seed=args.seed)
    print(f"Training images: {len(train_df):,} | validation images: {len(valid_df):,} | patients: {frame['Patient ID'].nunique():,}")
    # Weight formula follows the source project: positive weight = negative frequency; negative weight = positive frequency.
    y = train_df[LABELS].to_numpy(dtype=np.float32)
    pos_freq = np.clip(y.mean(axis=0), 1e-5, 1.0 - 1e-5)
    neg_freq = 1.0 - pos_freq
    pos_weights = torch.tensor(neg_freq, dtype=torch.float32, device=device)
    neg_weights = torch.tensor(pos_freq, dtype=torch.float32, device=device)

    norm = "xrv" if args.backbone == "xrv-nih" else "imagenet"
    train_ds = ChestXrayDataset(train_df, args.image_dir, args.image_size, train=True, normalization=norm)
    valid_ds = ChestXrayDataset(valid_df, args.image_dir, args.image_size, train=False, normalization=norm)
    loader_kwargs = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    if args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = build_model(pretrained=args.init_checkpoint is None and args.resume is None, backbone=args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_auc, stale_epochs = 1, -float("inf"), 0

    def load_source_checkpoint(checkpoint_path: str, resume_full: bool):
        nonlocal start_epoch, best_auc
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        meta_backbone = payload.get("backbone", "imagenet") if isinstance(payload, dict) else "imagenet"
        if meta_backbone != args.backbone:
            raise ValueError(f"Checkpoint backbone={meta_backbone!r}, but --backbone={args.backbone!r}. Use matching backbone.")
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        # Support both legacy bare torchvision DenseNet checkpoints (keys such as
        # "features.conv0.weight") and wrapped project checkpoints (keys such as
        # "base.features.conv0.weight" plus the wrapper's "label_indices" buffer).
        # Do not inspect only the first key: state_dict buffers can appear before "base.*".
        if args.backbone == "imagenet" and isinstance(state, dict) and state:
            has_wrapper_keys = any(key.startswith("base.") for key in state) or "label_indices" in state
            if has_wrapper_keys:
                model.load_state_dict(state, strict=True)
            else:
                model.base.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(state, strict=True)
        if resume_full and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            if payload.get("scheduler_state_dict"): scheduler.load_state_dict(payload["scheduler_state_dict"])
            if payload.get("scaler_state_dict"): scaler.load_state_dict(payload["scaler_state_dict"])
            start_epoch = int(payload.get("epoch", 0)) + 1
            best_auc = float(payload.get("best_val_auroc", -float("inf")))
        print(f"Loaded weights from {checkpoint_path}; full optimizer resume={resume_full and 'optimizer_state_dict' in payload}")

    if args.init_checkpoint:
        load_source_checkpoint(args.init_checkpoint, False)
    if args.resume:
        load_source_checkpoint(args.resume, True)

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    last_path = output.with_name(output.stem + "_last" + output.suffix)
    history_path = Path("reports/training_history.csv"); history_path.parent.mkdir(parents=True, exist_ok=True)
    history_exists = history_path.exists() and start_epoch > 1
    use_amp_msg = "enabled" if use_amp else "disabled"
    print(f"AMP: {use_amp_msg} | batch-size: {args.batch_size} | workers: {args.workers} | train batches/epoch: {len(train_loader):,}")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.time()
        model.train(); total_loss = 0.0; seen = 0
        scaler_is_enabled = use_amp
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train")
        for x, target in progress:
            x = x.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(x)
                loss = reference_weighted_bce(logits.float(), target, pos_weights, neg_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item() * x.size(0); seen += x.size(0)
            if seen and (progress.n % 100 == 0):
                progress.set_postfix(loss=f"{total_loss/max(seen,1):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        train_loss = total_loss / max(seen, 1)
        val_loss, val_auc, val_ap, per_label = validation_metrics(model, valid_loader, device, use_amp, pos_weights, neg_weights)
        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_auc if np.isfinite(val_auc) else -val_loss)
        elapsed = time.time() - epoch_started
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_macro_auroc={val_auc:.4f} val_macro_ap={val_ap:.4f} lr={optimizer.param_groups[0]['lr']:.2e} time_min={elapsed/60:.1f}")
        if optimizer.param_groups[0]["lr"] < previous_lr:
            print("ReduceLROnPlateau lowered the learning rate.")

        # Record per-label validation metrics at every epoch for honest model selection.
        Path("reports").mkdir(exist_ok=True)
        pd.DataFrame(per_label).to_csv(Path("reports") / f"validation_per_label_epoch_{epoch}.csv", index=False)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_macro_auroc": val_auc, "val_macro_ap": val_ap,
               "lr": optimizer.param_groups[0]["lr"], "seconds": elapsed}
        write_header = not history_path.exists() or (not history_exists and epoch == 1)
        with history_path.open("a" if not write_header else "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header: writer.writeheader()
            writer.writerow(row)

        checkpoint = {
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "labels": LABELS, "backbone": args.backbone, "image_size": args.image_size,
            "input_normalization": norm, "seed": args.seed, "epoch": epoch,
            "train_loss": train_loss, "validation_loss": val_loss,
            "best_val_auroc": max(best_auc, val_auc) if np.isfinite(val_auc) else best_auc,
            "training_pool": str(args.train_val_list or "all rows from CSV; patient-group train/validation split"),
        }
        torch.save(checkpoint, last_path)
        is_improved = np.isfinite(val_auc) and val_auc > best_auc + 1e-4
        if is_improved or (not np.isfinite(best_auc) and not np.isfinite(val_auc) and val_loss < checkpoint.get("best_val_loss", float("inf"))):
            best_auc = val_auc if np.isfinite(val_auc) else best_auc
            checkpoint["best_val_auroc"] = best_auc
            torch.save(checkpoint, output)
            print(f"Saved best checkpoint: {output}")
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(f"Saved resumable checkpoint: {last_path}")
        if stale_epochs >= args.patience:
            print(f"Early stopping: validation AUROC did not improve for {args.patience} epochs.")
            break
    print("Training finished. Research/education use only; not a medical device.")

if __name__ == "__main__":
    main()
