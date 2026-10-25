from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_list(path: Path) -> set[str]:
    return {
        Path(line.strip()).name
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate NIH ChestX-ray14 metadata, image inventory, and official split lists."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/nih"),
        help="Dataset directory (default: data/nih).",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Override metadata CSV path.")
    parser.add_argument("--train-val-list", type=Path, default=None, help="Override official train/validation list.")
    parser.add_argument("--test-list", type=Path, default=None, help="Override official test list.")
    parser.add_argument("--image-dir", type=Path, default=None, help="Override extracted PNG image directory.")
    args = parser.parse_args()

    base = args.dataset_dir
    csv_path = args.csv or base / "Data_Entry_2017_v2020.csv"
    train_path = args.train_val_list or base / "train_val_list.txt"
    test_path = args.test_list or base / "test_list.txt"
    image_dir = args.image_dir or base / "images"

    required_files = [csv_path, train_path, test_path]
    missing = [str(path) for path in required_files if not path.is_file()]
    if not image_dir.is_dir():
        missing.append(str(image_dir) + " (image directory)")
    if missing:
        parser.error("Required dataset paths not found: " + ", ".join(missing))

    df = pd.read_csv(csv_path, usecols=["Image Index", "Patient ID"])
    df["Image Index"] = df["Image Index"].astype(str)
    df["Patient ID"] = df["Patient ID"].astype(str)

    train = read_list(train_path)
    test = read_list(test_path)
    lookup = df.drop_duplicates("Image Index").set_index("Image Index")["Patient ID"].to_dict()
    train_patients = {lookup[name] for name in train if name in lookup}
    test_patients = {lookup[name] for name in test if name in lookup}
    actual_images = {path.name for path in image_dir.glob("*.png")}
    known_csv_names = set(lookup)

    print(f"CSV rows: {len(df):,}")
    print(f"Unique CSV image names: {len(known_csv_names):,}")
    print(f"Image files: {len(actual_images):,}")
    print(f"Official train/val filenames: {len(train):,}")
    print(f"Official test filenames: {len(test):,}")
    print(f"Train/val names absent from CSV: {len(train - known_csv_names):,}")
    print(f"Test names absent from CSV: {len(test - known_csv_names):,}")
    print(f"Filename overlap: {len(train & test):,}")
    print(f"Train/val entries missing from image folder: {len(train - actual_images):,}")
    print(f"Test entries missing from image folder: {len(test - actual_images):,}")
    print(f"Patient overlap between official pools: {len(train_patients & test_patients):,}")

    if train & test:
        print("WARNING: official train/validation and test lists overlap by filename.")
    if train_patients & test_patients:
        print("WARNING: patient overlap exists between the official lists.")


if __name__ == "__main__":
    main()
