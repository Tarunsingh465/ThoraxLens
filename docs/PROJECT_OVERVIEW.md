# Project Overview

ThoraxLens is a PyTorch research implementation for multi-label classification of frontal chest radiographs from the NIH ChestX-ray14 dataset. It uses a DenseNet-121 backbone with 14 finding outputs, patient-aware train/validation splitting, held-out test evaluation, validation-calibrated decision thresholds, and a Streamlit interface for exploratory score and saliency inspection.

The training pipeline supports two initialization modes: an ImageNet-initialized DenseNet-121 and the optional TorchXRayVision NIH medical-pretrained backbone (xrv-nih). Checkpoint loading supports fine-tuning from compatible checkpoints as well as resuming a saved training state. The evaluation tools report AUROC and average precision, with separate scripts for ranking metrics and threshold-based classification metrics.

The final reported experiment was a two-epoch fine-tuning run initialized from an earlier project checkpoint; it was not trained from scratch. Results, split accounting, and known caveats are documented in [FINAL_RESULTS.md](FINAL_RESULTS.md) and [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).

**Research/education only:** the model has not been clinically validated and must not be used for diagnosis, triage, treatment, or other clinical decisions.
