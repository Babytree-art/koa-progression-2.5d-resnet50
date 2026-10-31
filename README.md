# KOA Progression Prediction (2.5D ResNet-50)

Predicting knee osteoarthritis (KOA) progression from MRI using a 2.5D ResNet-50.

Three consecutive slices covering the thigh muscle regions (ROI1–3) are stacked as three channels and fed into an ImageNet-pretrained ResNet-50. The model outputs a patient-level risk probability of KOA progression.

> Patient data, NIfTI files, labels, and pretrained weights are not included.

## Installation

    pip install -r requirements.txt

Python 3.9.1 recommended. Place ImageNet ResNet-50 weights at `./weights/resnet50-19c8e357.pth`, or set `RESNET50_WEIGHTS`.

## Data Layout

    data/
    ├── train/<patient>/*t1*.nii*  *seg*.nii*  *label*.txt
    ├── val/<patient>/...
    └── test/<patient>/...

Each patient folder needs one T1 file, one segmentation file with the same shape, and a label file containing `0` (no progression) or `1` (progression).

## Usage

    export KOA_DATA_DIR=./data
    export RESNET50_WEIGHTS=./weights/resnet50-19c8e357.pth
    export KOA_OUTPUT_DIR=./results
    python train.py

## Method Summary

- Normalization: training-set ROI voxels clipped at 1st/99th percentiles, scaled to [0,1]
- Augmentation: rotation ±15°, horizontal/vertical flip, brightness/contrast, Gaussian blur
- Loss: label-smoothed BCE (smoothing = 0.1)
- Optimizer: AdamW (lr = 1e-4, weight decay = 1e-4), mixed precision
- Scheduler: ReduceLROnPlateau (factor = 0.5, patience = 5, monitoring val AUC)
- Early stopping: patience = 15, min_delta = 0.005
- Class imbalance handled by `WeightedRandomSampler`

## Outputs

All results are saved to `KOA_OUTPUT_DIR` (default `./results`), including `best_model.pth`, `training_history.csv`, `training_plots.png`, train/val/test reports, confusion matrices, predicted probabilities, and `best_threshold.txt`.
