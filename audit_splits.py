from pathlib import Path
import argparse
import pandas as pd


def read_list(path):
    return {Path(x.strip()).name for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()}


def main():
    p = argparse.ArgumentParser(description="Audit NIH official file lists for image and patient overlap")
    p.add_argument("--csv", required=True)
    p.add_argument("--train-val-list", required=True)
    p.add_argument("--test-list", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.csv, usecols=["Image Index", "Patient ID"])
    df["Image Index"] = df["Image Index"].astype(str)
    df["Patient ID"] = df["Patient ID"].astype(str)
    tr, te = read_list(args.train_val_list), read_list(args.test_list)
    tr_rows = df[df["Image Index"].isin(tr)]
    te_rows = df[df["Image Index"].isin(te)]
    overlap_files = tr & te
    overlap_patients = set(tr_rows["Patient ID"]) & set(te_rows["Patient ID"])
    print(f"Train/val list filenames: {len(tr):,}; CSV matches: {len(tr_rows):,}")
    print(f"Test list filenames: {len(te):,}; CSV matches: {len(te_rows):,}")
    print(f"Image filename overlap: {len(overlap_files):,}")
    print(f"Patients in train/val list: {tr_rows['Patient ID'].nunique():,}")
    print(f"Patients in test list: {te_rows['Patient ID'].nunique():,}")
    print(f"Patient overlap: {len(overlap_patients):,}")
    if overlap_patients:
        print("WARNING: official lists are not patient-disjoint. For a conservative test evaluation, pass --exclude-patient-overlap-with-list train_val_list.txt to src.evaluate.")
    else:
        print("Patient-level split audit passed.")

if __name__ == "__main__": main()
