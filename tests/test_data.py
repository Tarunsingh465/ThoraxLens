import pandas as pd
from src.data import load_metadata, patient_split
from src.labels import LABELS


def test_multi_label_parsing(tmp_path):
    p = tmp_path / "labels.csv"
    pd.DataFrame({"Image Index":["a.png", "b.png"], "Finding Labels":["Cardiomegaly|Edema", "No Finding"], "Patient ID":[1,2]}).to_csv(p,index=False)
    df = load_metadata(p)
    assert df.loc[0,"Cardiomegaly"] == 1
    assert df.loc[0,"Edema"] == 1
    assert df.loc[1,"Edema"] == 0
    assert len(LABELS) == 14


def test_patient_split_has_no_overlap():
    df = pd.DataFrame({"Patient ID":[1,1,2,2,3,4,5,6,7,8], "Image Index":[str(i) for i in range(10)]})
    a,b = patient_split(df, validation_fraction=0.3)
    assert not (set(a["Patient ID"]) & set(b["Patient ID"]))
