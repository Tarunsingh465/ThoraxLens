# Project Overview — Performance Upgrade

This version keeps the original repository's core research framing: DenseNet-121 for 14-label NIH ChestX-ray14 classification, class-frequency weighting, AUROC evaluation and Grad-CAM. It modernizes the implementation for current PyTorch/CUDA and adds optional TorchXRayVision NIH medical-pretrained initialization, scheduler/early stopping, per-label validation metrics, robust checkpoints, improved data loading, and split-overlap audit.

**Important distinctions:** The original reference is TensorFlow/Keras and provides its own weights inside `densenet.7z`; this PyTorch project does not directly load that 7z archive. The `xrv-nih` mode uses a separately published PyTorch medical-pretrained checkpoint. External pretraining may influence evaluation independence, so patient overlap and checkpoint provenance must be audited before reporting benchmark claims.

The model is an educational/research prototype and is not clinically validated.
