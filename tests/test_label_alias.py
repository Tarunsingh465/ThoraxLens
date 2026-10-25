import pandas as pd
from src.data import load_metadata


def test_nih_pleural_thickening_alias(tmp_path):
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({
        "Image Index": ["a.png", "b.png", "c.png"],
        "Finding Labels": ["Pleural_Thickening", "Pleural_Thickening|Effusion", "No Finding"],
        "Patient ID": [1, 2, 3],
    }).to_csv(csv_path, index=False)
    frame = load_metadata(csv_path)
    assert frame["Pleural Thickening"].tolist() == [1.0, 1.0, 0.0]
    assert frame["Effusion"].tolist() == [0.0, 1.0, 0.0]
