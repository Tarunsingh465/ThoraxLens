<div align="center">

# ThoraxLens

### Explainable AI for Multi-Label Chest X-Ray Analysis

**DenseNet-121 · NIH ChestX-ray14 · PyTorch · Streamlit · Saliency Visualization**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DenseNet--121-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Research Prototype](https://img.shields.io/badge/Status-Research%20Prototype-orange)

*An end-to-end research implementation for 14-label chest radiograph classification, patient-aware evaluation, validation-selected thresholds, and visual model inspection.*

</div>

> [!WARNING]
> **Research and education only — not a medical device.** ThoraxLens has not been clinically validated. Its scores, threshold flags, and saliency maps are not diagnoses and must not be used to make clinical decisions.

![Chest X-ray project header from the upstream reference repository](https://raw.githubusercontent.com/LaurentVeyssier/Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning/main/asset/xray-header-image.png)

---

## Project overview

ThoraxLens predicts scores for **14 findings** from a frontal chest X-ray. It is a multi-label task: more than one finding can receive a score for the same image.

The project is inspired by [Laurent Veyssier's Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning repository](https://github.com/LaurentVeyssier/Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning), which explores DenseNet-121, class-imbalance handling, ROC/AUC evaluation, and Grad-CAM. ThoraxLens is a separate PyTorch implementation that adds patient-split auditing, saved per-label evaluation reports, validation-derived decision thresholds, and a Streamlit research dashboard.

**This is not an exact reproduction** of the upstream TensorFlow/Keras notebook, training protocol, or its bundled model weights. Metrics below come from the ThoraxLens experiments.

## Key features

- DenseNet-121 multi-label classifier with 14 outputs.
- Dataset file and patient split auditing.
- Label alias handling for the NIH metadata label `Pleural_Thickening`.
- Held-out test evaluation using AUROC and average precision.
- Per-label thresholds selected on validation data, then held fixed for test evaluation.
- Streamlit dashboard for image upload, score inspection, threshold flags, and saliency visualization.
- CSV and JSON reports for reproducible inspection of the reported experiments.

## Dataset and split audit

ThoraxLens uses the NIH ChestX-ray dataset. Obtain the dataset from the [NIH ChestX-ray portal](https://nihcc.app.box.com/v/Chestxray-NIHCC) and review the current access terms. The dataset images and original archives are **not included** in this repository.

The local dataset audit recorded:

| Audit item | Result |
|---|---:|
| Extracted PNG images | 112,120 |
| Rows in `Data_Entry_2017_v2020.csv` | 112,120 |
| Image/metadata filename mismatches | 0 |
| Official train/validation list | 86,524 images |
| Official test list | 25,596 images |
| Filename overlap between official lists | 0 |
| Patient overlap reported by split audit | 0 |
| Internal training partition | 73,023 images |
| Internal validation partition | 13,501 images |
| Held-out test images evaluated | 25,596 |

The corrected label normalizer maps `Pleural_Thickening` to the display name **Pleural Thickening**. The corrected held-out test report contains 1,143 positive test cases for this finding.

Expected local dataset layout:

```text
Datasets/
├── Data_Entry_2017_v2020.csv
├── train_val_list.txt
├── test_list.txt
└── images/                     # extracted PNG images
```

See `data/DATASET_DOWNLOAD.md` for dataset setup notes. Local dataset paths can differ from machine to machine.

## Model architecture

ThoraxLens uses DenseNet-121, a convolutional neural network with densely connected blocks. Each layer in a dense block receives feature maps from earlier layers, encouraging feature reuse and providing short paths for gradient flow. The classification head is replaced with **14 outputs**.

![DenseNet architecture from the upstream reference repository](https://raw.githubusercontent.com/LaurentVeyssier/Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning/main/asset/densenet.png)

For the original author's explanation and architecture visuals, see the [upstream README](https://github.com/LaurentVeyssier/Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning#densenet-architecture). Those images remain hosted by and credited to the upstream project.

## Training setup

The final corrected fine-tuning run used:

| Setting | Value |
|---|---|
| Architecture | DenseNet-121, 14 output labels |
| Image size | 224 × 224 |
| Batch size | 2 |
| Epochs completed | 2 |
| Mixed precision | Enabled (AMP) |
| GPU | NVIDIA GeForce RTX 2050, 4 GB VRAM |
| PyTorch / Torchvision | `2.14.1+cu130` / `0.29.1+cu130` |
| Initialization | Earlier project checkpoint `densenet121_xray_best.pth` |
| Final best checkpoint (local) | `models/densenet121_xray_label_fixed_best.pth` |

The final corrected run fine-tuned from an earlier project checkpoint; it was not trained from scratch. The run completed two epochs.

| Epoch | Training loss | Validation loss | Validation macro AUROC | Validation macro AP |
|---:|---:|---:|---:|---:|
| 1 | 0.6699 | 0.7687 | 0.7742 | 0.1444 |
| 2 | 0.6388 | 0.6837 | 0.7838 | 0.1618 |

## Held-out test results

Evaluation used **25,596 held-out test images**. AUROC and average precision measure ranking quality; they do not establish clinical utility or guarantee performance at a particular decision threshold.

| Metric | Result |
|---|---:|
| Macro AUROC | **0.7620** |
| Macro average precision | **0.2002** |
| Macro precision at validation-selected thresholds | 0.1969 |
| Macro recall at validation-selected thresholds | 0.4352 |
| Macro F1 at validation-selected thresholds | 0.2613 |

### Per-finding ranking metrics

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

### Validation-selected thresholds

Each label threshold was selected by maximizing F1 on the validation partition, then kept fixed for held-out test evaluation. The test set was **not** used to choose these thresholds. These values are experimental operating points, not medical cutoffs.

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

Detailed artifacts live in `reports/`, including corrected per-finding test metrics, thresholded test metrics, training history, and validation threshold JSON/CSV files.

## Dashboard and saliency visualization

The Streamlit dashboard supports X-ray upload, the 14 model scores, comparisons against validation-derived thresholds, and a saliency heatmap for exploratory model inspection.

Run from the repository root, once a compatible local checkpoint and its matching threshold JSON are available:

```powershell
python -m streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in a browser and upload a PNG/JPG chest X-ray.

The displayed scores are raw model outputs, **not calibrated probabilities**. A threshold flag is not a diagnosis. A saliency map does not prove that the model has found a clinically valid feature or correctly localized a finding. Individual predictions can disagree with the report-derived NIH labels.

## Environment and setup

The project uses Python 3.11, PyTorch/Torchvision, pandas, NumPy, scikit-learn, Pillow, Matplotlib, and Streamlit.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA, install a CUDA-enabled PyTorch/Torchvision build compatible with your machine using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Repository structure

```text
ThoraxLens/
├── app.py
├── audit_splits.py
├── validate_dataset.py
├── requirements.txt
├── data/
│   └── DATASET_DOWNLOAD.md
├── models/                         # trained checkpoints are not committed
├── reports/
│   ├── patient_disjoint_test_metrics_fixed.csv
│   ├── patient_disjoint_test_threshold_metrics.csv
│   ├── training_history.csv
│   ├── validation_thresholds.json
│   └── validation_threshold_metrics.csv
├── src/
│   ├── data.py
│   ├── labels.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_thresholds.py
│   ├── calibrate_thresholds.py
│   └── inference.py
└── tests/
```

## Limitations and responsible use

- NIH ChestX-ray labels are report-derived and can be noisy or incomplete.
- Class prevalence is highly imbalanced; average precision varies substantially by finding.
- The model has only been evaluated in this project setup and has not undergone external clinical validation.
- Saliency visualization is exploratory and should not be interpreted as a verified lesion localization.
- The model is **not for diagnosis, triage, treatment, or other clinical decisions**.
- The reported two-epoch fine-tuning run is a research baseline, not evidence of clinical readiness.

## Reference

- Laurent Veyssier, [Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning](https://github.com/LaurentVeyssier/Chest-X-Ray-Medical-Diagnosis-with-Deep-Learning). This project is inspired by the repository's task framing and explanatory material, but it uses a separate PyTorch implementation and its own evaluation results.
- NIH Clinical Center, [Chest X-ray dataset portal](https://nihcc.app.box.com/v/Chestxray-NIHCC).

---

<div align="center">

**ThoraxLens — an explainable-AI research prototype for chest X-ray analysis.**

</div>
