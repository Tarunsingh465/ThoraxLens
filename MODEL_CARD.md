# ThoraxLens Model Card

## Overview

ThoraxLens is a research prototype for multi-label classification of chest radiographs using a DenseNet-121 backbone and the NIH ChestX-ray14 labels. It also provides per-label thresholding and a dashboard for inspecting scores and saliency maps.

**Status:** research / educational prototype. Not clinically validated and not for clinical use.

## Intended use

- Explore a multi-label medical-imaging ML workflow.
- Study preprocessing, patient-aware splits, model training, threshold calibration and held-out evaluation.
- Inspect model outputs in a dashboard as part of software and research experimentation.

## Out-of-scope use

Do not use ThoraxLens to diagnose, triage, rule out disease, recommend treatment, or make care decisions. It is not a substitute for a qualified clinician.

## Data

The project uses the NIH ChestX-ray dataset and report-derived labels for 14 findings. The labels may be noisy, incomplete, and subject to uncertainty. Dataset images are not redistributed by this repository. Users must obtain the dataset from its official portal and follow its current terms.

## Model and training

- Architecture: DenseNet-121 with 14 outputs.
- Input resolution in the final documented run: 224 × 224.
- Final corrected fine-tuning run: two epochs, batch size 2, mixed precision enabled.
- Initialization: earlier project checkpoint, not training from scratch.
- Device used in the documented run: NVIDIA GeForce RTX 2050 (4 GB VRAM).

## Evaluation summary

The corrected patient-disjoint test evaluation retained 25,596 held-out images and reported:

| Metric | Value |
|---|---:|
| Macro AUROC | 0.7620 |
| Macro average precision | 0.2002 |
| Macro precision at validation-selected thresholds | 0.1969 |
| Macro recall at validation-selected thresholds | 0.4352 |
| Macro F1 at validation-selected thresholds | 0.2613 |

These are project-reported experimental metrics, not estimates of clinical performance. See `reports/` for per-label outputs when the project files are present.

## Threshold policy

The decision threshold for each finding is selected on the validation split by maximizing F1, and then held fixed for test evaluation. The test set is not used to tune those thresholds. The resulting thresholds are experiment-specific operating points, not medical cutoffs.

## Known limitations and risks

- Report-derived labels can be noisy or incomplete.
- Strong class imbalance affects precision and average precision, especially for rare findings.
- A high AUROC does not guarantee useful precision at an operational threshold.
- Saliency maps are exploratory visualization, not evidence of verified disease localization or causal reasoning.
- The documented evaluation is not an independent external validation study.
- Performance may shift across institutions, devices, patient groups, acquisition protocols, and prevalence.

## Human oversight

Any use beyond educational experimentation would require appropriate clinical, regulatory, privacy, security, bias, and external validation review. Do not expose identifiable patient data or use this prototype on real clinical workflows.
