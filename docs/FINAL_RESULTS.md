# Chest X-ray Multi-label Research Demo — Results Report

**Purpose:** Educational and research prototype for multi-label classification on NIH ChestX-ray14. This project is not a medical device and must not be used for clinical decisions.

## 1. Project summary

The project uses a DenseNet-121 model to produce scores for 14 NIH ChestX-ray14 findings from frontal chest radiographs. The application provides raw model scores, validation-derived threshold flags, and exploratory Grad-CAM-style saliency visualizations.

The implementation is a modern PyTorch reimplementation inspired by the referenced GitHub project; it is not an exact reproduction of the repository's original notebook or pretrained weights.

## 2. Dataset and split validation

- Images extracted: **112,120 PNG files**
- Metadata rows in `Data_Entry_2017_v2020.csv`: **112,120**
- Unique CSV image filenames: **112,120**
- Image filenames absent from CSV: **0**
- CSV entries whose image files were missing: **0**
- Official train/validation filename list: **86,524** (all matched the CSV)
- Official test filename list: **25,596** (all matched the CSV)
- Filename overlap between these lists: **0**
- Patient-level split audit: **passed; no patient overlap reported**
- Patient-group internal split from the train/validation pool: **73,023 training images** and **13,501 validation images**
- Held-out test images retained after overlap exclusion: **25,596**

The metadata's `Pleural_Thickening` label is normalized to the application label `Pleural Thickening`. A bug in an earlier version caused this label to be counted as zero; the corrected evaluation reports **1,143 positive test cases** for Pleural Thickening.

## 3. Training configuration and validation results

Final corrected training run:

- Model: DenseNet-121, 14 outputs
- Device: NVIDIA GeForce RTX 2050 (4 GB VRAM)
- PyTorch: 2.14.1+cu130; Torchvision: 0.29.1+cu130
- Input size: 224 × 224
- Batch size: 2
- Mixed precision (AMP): enabled
- Epochs completed: 2
- Initialization: prior project checkpoint (`densenet121_xray_best.pth`); optimizer state was not resumed
- Checkpoint: `models/densenet121_xray_label_fixed_best.pth`

| Metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Training loss | 0.6699 | 0.6388 |
| Validation loss | 0.7687 | 0.6837 |
| Validation macro AUROC | 0.7742 | 0.7838 |
| Validation macro average precision | 0.1444 | 0.1618 |

## 4. Held-out test results — ranking metrics

The following results come from the corrected evaluation of **25,596 patient-disjoint test images**. AUROC and average precision are ranking metrics; they do not specify performance at a chosen decision threshold.

| Finding | Positive test cases | AUROC | Average precision |
|---|---:|---:|---:|
| Atelectasis | 3,279 | 0.7311 | 0.2821 |
| Cardiomegaly | 1,069 | 0.8697 | 0.3015 |
| Consolidation | 1,815 | 0.6974 | 0.1187 |
| Edema | 925 | 0.8225 | 0.1262 |
| Effusion | 4,658 | 0.8083 | 0.4671 |
| Emphysema | 1,093 | 0.8208 | 0.1927 |
| Fibrosis | 435 | 0.7425 | 0.0435 |
| Hernia | 86 | 0.8039 | 0.0142 |
| Infiltration | 6,112 | 0.6781 | 0.3739 |
| Mass | 1,748 | 0.7711 | 0.2608 |
| Nodule | 1,623 | 0.7263 | 0.1916 |
| Pleural Thickening | 1,143 | 0.7281 | 0.0988 |
| Pneumonia | 555 | 0.6670 | 0.0385 |
| Pneumothorax | 2,665 | 0.8016 | 0.2938 |

**Macro AUROC: 0.7620**  
**Macro average precision: 0.2002** (the earlier unthresholded report printed 0.2003; minor difference may result from evaluation numerics or rounding, so use the threshold-evaluation report value when quoting the complete final report.)

## 5. Validation-calibrated thresholds and held-out test metrics

Each disease has its own threshold selected on validation data by maximizing F1. These thresholds were then held fixed for the test evaluation. The test set was not used to tune the thresholds.

| Finding | Validation threshold | Test precision | Test recall | Test F1 |
|---|---:|---:|---:|---:|
| Atelectasis | 0.6841 | 0.2434 | 0.5657 | 0.3404 |
| Cardiomegaly | 0.9332 | 0.3246 | 0.4041 | 0.3600 |
| Consolidation | 0.7917 | 0.1292 | 0.4871 | 0.2043 |
| Edema | 0.8822 | 0.1183 | 0.6011 | 0.1978 |
| Effusion | 0.7765 | 0.3804 | 0.7306 | 0.5003 |
| Emphysema | 0.8836 | 0.1990 | 0.4163 | 0.2692 |
| Fibrosis | 0.7341 | 0.0575 | 0.2069 | 0.0900 |
| Hernia | 0.9172 | 0.0284 | 0.0581 | 0.0382 |
| Infiltration | 0.5332 | 0.3206 | 0.7546 | 0.4500 |
| Mass | 0.9218 | 0.3060 | 0.3793 | 0.3387 |
| Nodule | 0.8401 | 0.1874 | 0.3845 | 0.2520 |
| Pleural Thickening | 0.7415 | 0.1139 | 0.4094 | 0.1782 |
| Pneumonia | 0.8209 | 0.0441 | 0.2414 | 0.0746 |
| Pneumothorax | 0.8554 | 0.3042 | 0.4533 | 0.3641 |

- **Macro precision at validation thresholds:** 0.1969
- **Macro recall at validation thresholds:** 0.4352
- **Macro F1 at validation thresholds:** 0.2613
- **Macro AUROC:** 0.7620
- **Macro average precision:** 0.2002

Threshold-derived precision and F1 are modest, with substantial variation by disease. Hernia has only 86 positive test examples, so its thresholded metrics are particularly uncertain. No confidence intervals or bootstrap intervals have been produced in the files listed so far.

## 6. Dashboard status

The Streamlit dashboard was launched locally and successfully:

- accepted a PNG chest X-ray upload;
- generated 14 model scores;
- loaded the corrected checkpoint specified in the dashboard;
- displayed validation-derived thresholds and above/below-threshold flags;
- generated an exploratory Grad-CAM-style heatmap.

An example image labelled `No Finding` received some high scores in the demo. That disagreement should be recorded as a potential false-positive/error-analysis case, not interpreted as a clinical finding. NIH report-mined labels can be imperfect, and a single image does not establish overall model quality.

## 7. Existing report artifacts

The following files were present in the project's `reports` directory:

- `patient_disjoint_test_metrics.csv`
- `patient_disjoint_test_metrics.csv`
- `patient_disjoint_test_threshold_metrics.csv`
- `training_history.csv`
- `validation_per_label_epoch_1.csv`
- `validation_per_label_epoch_2.csv`
- `validation_thresholds.json`
- `validation_threshold_metrics.csv`

Use `patient_disjoint_test_threshold_metrics.csv` for the threshold-specific test precision/recall/F1 report and `patient_disjoint_test_metrics.csv` for the corrected per-finding ranking metrics.

## 8. Limitations and next improvements

1. This is an educational/research prototype, not a clinically validated diagnostic system.
2. Scores displayed by the dashboard are raw, uncalibrated model scores—not calibrated disease probabilities.
3. Validation-derived F1 thresholds are not clinical cutoffs and may not generalize across sites, scanners, populations, or prevalence shifts.
4. Some findings are rare and have low average precision; report per-disease metrics rather than only an aggregate number.
5. The reference publication's radiologist comparison cannot be reproduced using NIH labels alone. A properly annotated evaluation set and suitable comparison protocol would be required.
6. Before further performance claims, preserve the held-out test set, report confidence intervals (e.g., bootstrap by patient), inspect calibration, consider threshold selection for the intended operating point, and review false-positive/false-negative examples.
7. This PyTorch implementation is inspired by the reference repository but is not an exact reproduction of its original notebook, preprocessing, or pretrained weights.

## 9. Reproducible paths in the local project

Dataset: `D:\chest_xray_project\Datasets`  
Project: `D:\chest_xray_project_v3`  
Final corrected checkpoint: `models\densenet121_xray_label_fixed_best.pth`  
Validation thresholds: `reports\validation_thresholds.json`  
Thresholded test metrics: `reports\patient_disjoint_test_threshold_metrics.csv`
