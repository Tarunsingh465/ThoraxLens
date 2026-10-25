from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from .labels import LABELS


def _xrv_scale(tensor: torch.Tensor) -> torch.Tensor:
    # Picklable transform for Windows DataLoader workers (avoid lambda pickling failures).
    return tensor * 2048.0 - 1024.0


def make_transform(image_size: int, train: bool = False, normalization: str = "imagenet"):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        # Modest augmentations appropriate for non-lateral, frontal NIH images.
        ops += [transforms.RandomRotation(5), transforms.RandomHorizontalFlip(p=0.5)]
    ops += [transforms.Grayscale(num_output_channels=3 if normalization == "imagenet" else 1),
            transforms.ToTensor()]
    if normalization == "imagenet":
        ops += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    elif normalization == "xrv":
        # TorchXRayVision expects pixels normalized from [0, 255] into [-1024, 1024].
        ops += [transforms.Lambda(_xrv_scale)]
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    return transforms.Compose(ops)


def read_image_list(list_path: str | Path) -> set[str]:
    path = Path(list_path)
    names = {Path(line.strip()).name for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    if not names:
        raise ValueError(f"Image list is empty: {path}")
    return names


def load_metadata(csv_path: str | Path, limit: int | None = None, seed: int = 42,
                  image_list: str | Path | None = None):
    df = pd.read_csv(csv_path)
    image_col = "Image Index"
    label_col = "Finding Labels"
    if image_col not in df or label_col not in df:
        raise ValueError(f"Expected columns 'Image Index' and 'Finding Labels'; found: {list(df.columns)}")
    df = df.copy()
    df[image_col] = df[image_col].astype(str)
    if image_list is not None:
        allowed = read_image_list(image_list)
        df = df[df[image_col].isin(allowed)].copy()
        matched = set(df[image_col])
        missing_csv = allowed - matched
        print(f"List entries: {len(allowed):,} | CSV matches: {len(df):,} | list entries absent from CSV: {len(missing_csv):,}")
        if len(df) == 0:
            raise ValueError(f"No CSV rows matched filenames in {image_list}")
    if "Patient ID" not in df:
        df["Patient ID"] = df[image_col].astype(str).str.split("_").str[0]
    # NIH CSV uses "Pleural_Thickening", while this project's display label
    # is "Pleural Thickening". Canonicalize this known source-label alias
    # before building multi-hot targets; otherwise that class becomes all zeros.
    source_to_project_label = {"Pleural_Thickening": "Pleural Thickening"}

    def parse_findings(value: object) -> set[str]:
        items = {item.strip() for item in str(value).split("|") if item.strip()}
        return {source_to_project_label.get(item, item) for item in items}

    label_sets = df[label_col].fillna("").map(parse_findings)
    for label in LABELS:
        df[label] = label_sets.map(lambda values: float(label in values))
    if limit is not None and limit > 0 and len(df) > limit:
        # Sampling is restricted to the selected pool before patient split.
        df = df.sample(n=limit, random_state=seed).sort_index()
    return df.reset_index(drop=True)


def patient_split(df: pd.DataFrame, validation_fraction: float = 0.15, seed: int = 42):
    """Patient-group split so multiple studies from one patient never cross train/validation."""
    groups = df["Patient ID"].astype(str)
    if groups.nunique() < 2:
        raise ValueError("At least two distinct patients are required for a patient-level split.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_idx, valid_idx = next(splitter.split(df, groups=groups))
    train_df, valid_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[valid_idx].reset_index(drop=True)
    overlap = set(train_df["Patient ID"].astype(str)) & set(valid_df["Patient ID"].astype(str))
    if overlap:
        raise RuntimeError("Patient leakage detected between train and validation data.")
    return train_df, valid_df


class ChestXrayDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: str | Path, image_size: int = 224,
                 train: bool = False, normalization: str = "imagenet"):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = make_transform(image_size, train, normalization)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = self.image_dir / str(row["Image Index"])
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as image:
            x = self.transform(image.convert("L"))
        y = torch.tensor(row[LABELS].to_numpy(dtype="float32"), dtype=torch.float32)
        return x, y
