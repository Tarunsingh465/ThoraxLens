from pathlib import Path
import pandas as pd

base = Path(r"D:\chest_xray_project\Datasets")
csv_path = base / "Data_Entry_2017_v2020.csv"
train_path = base / "train_val_list.txt"
test_path = base / "test_list.txt"
image_dir = base / "images"

df = pd.read_csv(csv_path)
train = {Path(s.strip()).name for s in train_path.read_text(encoding="utf-8-sig").splitlines() if s.strip()}
test = {Path(s.strip()).name for s in test_path.read_text(encoding="utf-8-sig").splitlines() if s.strip()}
lookup = df.set_index("Image Index")["Patient ID"].astype(str).to_dict()
train_patients = {lookup[n] for n in train if n in lookup}
test_patients = {lookup[n] for n in test if n in lookup}
actual_images = {p.name for p in image_dir.glob("*.png")}
print(f"CSV rows: {len(df)}")
print(f"Image files: {len(actual_images)}")
print(f"Official train/val filenames: {len(train)}")
print(f"Official test filenames: {len(test)}")
print(f"Filename overlap: {len(train & test)}")
print(f"Train/val entries missing in images: {len(train - actual_images)}")
print(f"Test entries missing in images: {len(test - actual_images)}")
print(f"Patient overlap between official pools: {len(train_patients & test_patients)}")
