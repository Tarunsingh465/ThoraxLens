# Evaluation Protocol

## Dataset accounting

The local audit documented 112,120 extracted PNG images and 112,120 rows in `Data_Entry_2017_v2020.csv`. The official train/validation list contained 86,524 entries, while the official test list contained 25,596 entries. The lists had zero filename overlap.

The internal train/validation split was created within the official train/validation pool:

- Training: 73,023 images
- Validation: 13,501 images
- Held-out test: 25,596 images

The patient-split audit reported no patient overlap between the internal training/validation group and the held-out test group.

## Label normalization

NIH metadata uses the underscore label `Pleural_Thickening`. The project normalizes that alias to the display label `Pleural Thickening` so that test positives are counted correctly. The corrected test report contains 1,143 positives for this label.

## Training run in the project report

| Setting | Value |
|---|---|
| Model | DenseNet-121, 14 labels |
| Input size | 224 × 224 |
| Batch size | 2 |
| Epochs completed | 2 |
| AMP | Enabled |
| Initialization | Earlier ThoraxLens project checkpoint |
| GPU | NVIDIA GeForce RTX 2050, 4 GB VRAM |

| Epoch | Train loss | Validation loss | Validation macro AUROC | Validation macro AP |
|---:|---:|---:|---:|---:|
| 1 | 0.6699 | 0.7687 | 0.7742 | 0.1444 |
| 2 | 0.6388 | 0.6837 | 0.7838 | 0.1618 |

## Metrics

- **AUROC**: area under the receiver operating characteristic curve; measures ranking across thresholds.
- **Average precision (AP)**: summarizes the precision-recall curve and is important for rare labels.
- **Precision, recall, and F1**: calculated after binarizing each label using its validation-selected threshold.

Macro metrics are the unweighted average of the per-label metric values, not sample-level metrics.

## Threshold calibration

For each finding, F1 was maximized on the validation split. The chosen threshold was saved to `reports/validation_thresholds.json` and then applied unchanged to the held-out test set. This avoids using the test set for threshold tuning.

## Corrected held-out test summary

| Metric | Value |
|---|---:|
| Macro AUROC | 0.7620 |
| Macro AP | 0.2002 |
| Macro precision | 0.1969 |
| Macro recall | 0.4352 |
| Macro F1 | 0.2613 |

## Reproducibility notes

The reported run represents two fine-tuning epochs and initializes from an earlier project checkpoint; it is not a from-scratch training experiment. The exact training environment, data version, preprocessing settings, and checkpoint must be preserved to reproduce results. Dataset and checkpoint files are not currently distributed in this repository.

## Interpretation caveats

NIH labels are extracted from report text and can be noisy or incomplete. Metrics describe this experimental dataset and protocol only. They do not demonstrate clinical validity, utility, fairness, or safety.
