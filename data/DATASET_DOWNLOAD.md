# Dataset access

The images are not included in this repository because of their size and the dataset's access terms. Obtain them from the [NIH Chest X-ray dataset portal](https://nihcc.app.box.com/v/Chestxray-NIHCC) and follow the provider's terms.

Place the files in this layout so the validation script works with its defaults:

~~~~text
data/nih/
├── Data_Entry_2017_v2020.csv
├── train_val_list.txt
├── test_list.txt
└── images/
    ├── 00000001_000.png
    └── ...
~~~~

Download/extract the image archives (images_001.tar.gz through images_012.tar.gz) into data/nih/images/; put the metadata CSV and official train/validation and test lists in data/nih/. The dataset contains about 112,120 images and requires tens of gigabytes of storage.

Run the inventory and split check from the repository root:

~~~~bash
python validate_dataset.py
~~~~

For a custom location, use python validate_dataset.py --dataset-dir /path/to/nih or pass explicit --csv, --train-val-list, --test-list, and --image-dir arguments. Do not commit dataset archives, extracted images, or patient data to GitHub.
