"""
KOA progression prediction with a 2.5D ResNet-50.

Environment variables (optional):
    KOA_DATA_DIR          Root data directory. Default: ./data
    RESNET50_WEIGHTS      Path to ImageNet ResNet-50 weights. Default: ./weights/resnet50-19c8e357.pth
    KOA_OUTPUT_DIR        Output directory. Default: ./results

Expected data layout:
    <KOA_DATA_DIR>/
        train/<patient_dir>/...t1*.nii*  ...seg*.nii*  *label*.txt
        val/<patient_dir>/...
        test/<patient_dir>/...
"""

import os
import json
import time
import copy
import glob
import pickle
import warnings

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# AMP: legacy API, compatible with PyTorch 1.6~2.x
from torch.cuda.amp import GradScaler, autocast as cuda_autocast

import torchvision.transforms as transforms
import timm
import albumentations as A

from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


# ==================== Configuration ====================
class Config:
    data_dir = os.environ.get("KOA_DATA_DIR", "./data")
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    batch_size = 16
    # Use 4 workers on Linux to reduce concurrent NIfTI I/O pressure.
    # Keep 0 on Windows because spawn-based multiprocessing is fragile.
    num_workers = 0 if os.name == 'nt' else 4
    image_size = 224
    num_classes = 1
    lr = 1e-4
    epochs = 100
    early_stop_patience = 15
    min_delta = 0.005
    weight_decay = 1e-4
    dropout = 0.3
    smoothing = 0.1

    model_name = "resnet50"
    # Local ImageNet-pretrained weights; timm pretrained=False, loaded manually.
    pretrained_path = os.environ.get(
        "RESNET50_WEIGHTS", "./weights/resnet50-19c8e357.pth"
    )
    freeze_backbone = False
    unfreeze_after = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    use_amp = True

    output_dir = os.environ.get("KOA_OUTPUT_DIR", "./results")
    os.makedirs(output_dir, exist_ok=True)

    norm_stats_path = os.path.join(output_dir, "norm_stats.pkl")
    model_save_path = os.path.join(output_dir, "best_model.pth")
    history_save_path = os.path.join(output_dir, "training_history.csv")
    plot_save_path = os.path.join(output_dir, "training_plots.png")
    config_save_path = os.path.join(output_dir, "config.json")

    roi_labels = [1, 2, 3]
    seg_suffix = "seg"
    t1_suffix = "t1"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(Config.seed)


# ==================== AMP compatibility layer ====================
def amp_autocast(enabled):
    try:
        return torch.cuda.amp.autocast(enabled=enabled)
    except (TypeError, AttributeError):
        return torch.amp.autocast(device_type='cuda', enabled=enabled)


def make_grad_scaler(enabled):
    try:
        return torch.cuda.amp.GradScaler(enabled=enabled)
    except (TypeError, AttributeError):
        return torch.amp.GradScaler('cuda', enabled=enabled)


# ==================== Module-level Resize (picklable) ====================
class ResizeLambda:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        return F.interpolate(
            x.unsqueeze(0),
            size=(self.size, self.size),
            mode='bilinear', align_corners=False
        ).squeeze(0)


# ==================== Save Config ====================
def save_config():
    cfg = {k: v for k, v in Config.__dict__.items() if not k.startswith("_")}
    with open(Config.config_save_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)


# ==================== Label-smoothed BCE ====================
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert 0.0 <= smoothing < 0.5, "smoothing should be in [0, 0.5)"
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        targets = targets * (1.0 - 2.0 * self.smoothing) + self.smoothing
        return self.bce(outputs, targets)


# ==================== Safe AUC ====================
def safe_roc_auc(labels, probs):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, probs)


# ==================== Best threshold on validation set ====================
def find_best_threshold(labels, probs):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    return float(thresholds[best_idx])


# ==================== seg -> ROI ====================
def seg_to_roi(seg_slice, roi_labels):
    roi = np.zeros_like(seg_slice, dtype=np.float32)
    for lab in roi_labels:
        roi[seg_slice == lab] = 1.0
    return roi


# ==================== Training-set normalization statistics ====================
class DatasetStatsCalculator:
    """
    Clip ROI voxel intensities using the 1st/99th percentiles of the
    training set, then linearly map to [0, 1].
    """
    def __init__(self, dataset, sample_pixels=1_000_000, force_recompute=False):
        self.sample_pixels = sample_pixels
        self.stats_path = Config.norm_stats_path

        if os.path.exists(self.stats_path) and not force_recompute:
            print(f"Loading existing normalization stats from {self.stats_path}")
            with open(self.stats_path, 'rb') as f:
                stats = pickle.load(f)
            self.low = stats['low']
            self.high = stats['high']
            self.mean = stats.get('mean', None)
            self.std = stats.get('std', None)
            print(f"Loaded: low={self.low:.4f}, high={self.high:.4f}")
            return

        print("Calculating global statistics from training set ...")
        all_pixels = []

        for i in tqdm(range(len(dataset)), desc="Computing stats"):
            sample = dataset.samples[i]
            t1_3d = nib.load(sample["t1"]).get_fdata().astype(np.float32)
            seg_3d = nib.load(sample["seg"]).get_fdata().astype(np.float32)

            for d in range(t1_3d.shape[-1]):
                img_slice = t1_3d[:, :, d]
                seg_slice = seg_3d[:, :, d]
                roi = seg_to_roi(seg_slice, Config.roi_labels)
                roi_mask = roi > 0
                valid = img_slice[roi_mask]
                if len(valid) > 0:
                    all_pixels.append(valid)

        if len(all_pixels) == 0:
            raise RuntimeError("No valid pixels found in training set!")

        all_pixels = np.concatenate(all_pixels)
        print(f"Total valid pixels collected: {len(all_pixels):,}")

        if len(all_pixels) > self.sample_pixels:
            print(f"Sampling {self.sample_pixels:,} pixels for statistics ...")
            all_pixels = np.random.choice(all_pixels, size=self.sample_pixels, replace=False)

        self.low = float(np.percentile(all_pixels, 1))
        self.high = float(np.percentile(all_pixels, 99))
        self.mean = float(np.mean(all_pixels))
        self.std = float(np.std(all_pixels))

        print("\nGlobal stats computed (min-max to [0, 1]):")
        print(f"  Low  (1%):  {self.low:.4f}")
        print(f"  High (99%): {self.high:.4f}")

        with open(self.stats_path, 'wb') as f:
            pickle.dump({
                'low': self.low, 'high': self.high,
                'mean': self.mean, 'std': self.std
            }, f)
        print(f"Saved normalization stats to {self.stats_path}")


# ==================== Fixed normalization to [0, 1] ====================
class FixedNormalize:
    def __init__(self, low, high, mean=None, std=None):
        self.low = float(low)
        self.high = float(high)
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = torch.clamp(x, self.low, self.high)
        x = (x - self.low) / (self.high - self.low + 1e-8)
        return x


# ==================== 2.5D dataset ====================
class KneeMRIDataset(Dataset):
    def __init__(self, data_dir, transform=None, return_path=False):
        self.data_dir = data_dir
        self.transform = transform
        self.return_path = return_path
        self.samples = []
        self.labels = []

        patient_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
        print(f"Found {len(patient_dirs)} patient directories in {data_dir}")

        skipped = 0
        for patient_dir in patient_dirs:
            if not os.path.isdir(patient_dir):
                continue

            t1_files = sorted(glob.glob(
                os.path.join(patient_dir, f"*{Config.t1_suffix}*.nii*")))
            seg_files = sorted(glob.glob(
                os.path.join(patient_dir, f"*{Config.seg_suffix}*.nii*")))
            label_files = glob.glob(os.path.join(patient_dir, "*label*.txt"))

            if len(t1_files) == 1 and len(seg_files) == 1 and len(label_files) >= 1:
                valid = True
                try:
                    t1_img = nib.load(t1_files[0]).get_fdata().astype(np.float32)
                    seg_img = nib.load(seg_files[0]).get_fdata().astype(np.float32)

                    if t1_img.shape != seg_img.shape:
                        print(f"Skipping {patient_dir}: shape mismatch "
                              f"t1={t1_img.shape}, seg={seg_img.shape}")
                        valid = False
                    elif t1_img.ndim != 3 or t1_img.shape[-1] != 3:
                        print(f"Skipping {patient_dir}: t1 shape is "
                              f"{t1_img.shape}, expected (H, W, 3)")
                        valid = False
                    else:
                        has_roi = False
                        for d in range(3):
                            roi = seg_to_roi(seg_img[:, :, d], Config.roi_labels)
                            if roi.sum() > 0:
                                has_roi = True
                                break
                        if not has_roi:
                            print(f"Skipping {patient_dir}: no ROI labels "
                                  f"{Config.roi_labels} found in seg")
                            valid = False
                except Exception as e:
                    print(f"Skipping {patient_dir}: error - {e}")
                    valid = False

                if not valid:
                    skipped += 1
                    continue

                with open(label_files[0], 'r') as f:
                    label = int(f.read().strip())

                self.samples.append({
                    "t1": t1_files[0],
                    "seg": seg_files[0],
                    "label": label,
                    "patient_dir": patient_dir,
                })
                self.labels.append(label)
            else:
                skipped += 1

        print(f"Loaded {len(self.samples)} valid samples (skipped {skipped})")
        if self.labels:
            print(f"Class distribution: {np.bincount(self.labels)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        t1_3d = nib.load(sample["t1"]).get_fdata().astype(np.float32)
        seg_3d = nib.load(sample["seg"]).get_fdata().astype(np.float32)

        masked_slices = []
        original_slices = []

        for d in range(t1_3d.shape[-1]):
            t1_slice = t1_3d[:, :, d]
            seg_slice = seg_3d[:, :, d]
            roi = seg_to_roi(seg_slice, Config.roi_labels)
            masked_slices.append(t1_slice * roi)
            original_slices.append(t1_slice)

        stacked = np.stack(masked_slices, axis=0).astype(np.float32)
        stacked_tensor = torch.tensor(stacked, dtype=torch.float32)

        original_imgs = torch.tensor(
            np.stack(original_slices, axis=0), dtype=torch.float32
        )

        label = torch.tensor(sample["label"], dtype=torch.float32)

        if self.transform:
            stacked_tensor = self.transform(stacked_tensor)

        if self.return_path:
            return stacked_tensor, label, original_imgs, sample["t1"]
        return stacked_tensor, label, original_imgs


# ==================== Data augmentation ====================
class RandomAugment:
    def __init__(self, size):
        self.transform = A.Compose([
            A.Resize(size, size),
            A.Rotate(limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        ])

    def __call__(self, x):
        # x is a normalized tensor in [0, 1]; convert HWC to numpy first.
        x_np = x.permute(1, 2, 0).contiguous().numpy()
        augmented = self.transform(image=x_np)
        img_np = augmented["image"]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return img_tensor


# ==================== Model ====================
def load_pretrained_resnet():
    model = timm.create_model(
        Config.model_name,
        pretrained=False,
        num_classes=1000,
        in_chans=3
    )

    weight_path = Config.pretrained_path

    if not os.path.isfile(weight_path):
        if os.path.isdir(weight_path):
            candidates = (
                glob.glob(os.path.join(weight_path, "*.pth")) +
                glob.glob(os.path.join(weight_path, "*.bin"))
            )
            if len(candidates) == 0:
                raise FileNotFoundError(
                    f"No weight file (.pth/.bin) found in directory: {weight_path}"
                )
            weight_path = sorted(candidates)[0]
        else:
            if os.path.isfile(weight_path + ".pth"):
                weight_path = weight_path + ".pth"
            else:
                raise FileNotFoundError(f"Pretrained weight not found: {weight_path}")

    print(f"Loading local ImageNet pretrained weights: {weight_path}")

    try:
        checkpoint = torch.load(weight_path, map_location="cpu")
    except Exception as e:
        print(f"torch.load default failed ({e}), retrying with weights_only=False")
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Local weights loaded. Missing keys: {len(missing)}, "
          f"Unexpected keys: {len(unexpected)}")
    if missing:
        print("  Missing keys (first 5):", missing[:5])
    if unexpected:
        print("  Unexpected keys (first 5):", unexpected[:5])

    num_features = model.num_features
    model.fc = nn.Sequential(
        nn.Dropout(Config.dropout),
        nn.Linear(num_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(Config.dropout / 2),
        nn.Linear(256, Config.num_classes)
    )

    if Config.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith('fc.'):
                param.requires_grad = False

    return model.to(Config.device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== Train one epoch ====================
def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_labels, all_probs = [], []

    if Config.freeze_backbone and (epoch + 1) == Config.unfreeze_after:
        print("Unfreezing backbone layers ...")
        for param in model.parameters():
            param.requires_grad = True

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(Config.use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    if total_samples == 0:
        return 0.0, 0.0, 0.5

    avg_loss = total_loss / total_samples
    preds = (np.array(all_probs) > 0.5).astype(float)
    acc = accuracy_score(all_labels, preds)
    auc = safe_roc_auc(all_labels, all_probs)
    if np.isnan(auc):
        auc = 0.5
    return avg_loss, acc, auc


# ==================== Evaluation ====================
def evaluate(model, loader, criterion, device, set_name="validation",
             save_report=True, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in loader:
            images, labels, _ = batch
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    if total_samples == 0:
        print(f"\n[{set_name}] empty loader, skipping evaluation.")
        return 0.0, 0.0, 0.5, np.zeros((2, 2), dtype=int), "", np.array([]), np.array([])

    avg_loss = total_loss / total_samples
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    preds = (all_probs > threshold).astype(float)

    acc = accuracy_score(all_labels, preds)
    auc = safe_roc_auc(all_labels, all_probs)
    if np.isnan(auc):
        auc = 0.5
    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    report = classification_report(
        all_labels, preds, labels=[0, 1],
        target_names=['No Progress', 'Progress'],
        zero_division=0
    )

    print(f"\n{set_name.capitalize()} Set Evaluation (threshold={threshold:.4f}):")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC: {auc:.4f}")
    print("  Confusion Matrix:\n", cm)
    print("  Classification Report:\n", report)

    if save_report:
        np.save(os.path.join(Config.output_dir, f"{set_name}_probs.npy"), all_probs)
        np.save(os.path.join(Config.output_dir, f"{set_name}_labels.npy"), all_labels)

        report_path = os.path.join(Config.output_dir, f"{set_name}_report.txt")
        with open(report_path, 'w', encoding="utf-8") as f:
            f.write(f"{set_name.capitalize()} Set Evaluation (threshold={threshold:.4f}):\n")
            f.write(f"Loss: {avg_loss:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\nClassification Report:\n")
            f.write(report)

        cm_path = os.path.join(Config.output_dir, f"{set_name}_confusion_matrix.png")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Progress', 'Progress'],
                    yticklabels=['No Progress', 'Progress'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{set_name.capitalize()} Set Confusion Matrix '
                  f'(threshold={threshold:.2f})')
        plt.savefig(cm_path)
        plt.close()

    return avg_loss, acc, auc, cm, report, all_probs, all_labels


# ==================== Training loop ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    best_val_auc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_auc': [], 'val_auc': [], 'lr': []
    }
    scaler = make_grad_scaler(Config.use_amp)

    for epoch in range(Config.epochs):
        print(f"\nEpoch {epoch + 1}/{Config.epochs}")
        start_time = time.time()

        train_loss, train_acc, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, scaler, Config.device, epoch
        )

        val_loss, val_acc, val_auc, _, _, _, _ = evaluate(
            model, val_loader, criterion, Config.device,
            set_name="validation", save_report=False
        )

        current_lr = optimizer.param_groups[0]['lr']
        old_lr = current_lr
        scheduler.step(val_auc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"  [LR] ReduceLROnPlateau: {old_lr:.6f} -> {new_lr:.6f}")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)
        pd.DataFrame(history).to_csv(Config.history_save_path, index=False)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1} | Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}")
        print(f"  Train AUC:  {train_auc:.4f} | Val AUC:  {val_auc:.4f}")
        print(f"  LR: {current_lr:.6f}")

        if val_auc > best_val_auc + Config.min_delta:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
                'val_loss': val_loss,
            }, Config.model_save_path)
            print(f"Saved new best model with AUC: {best_val_auc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{Config.early_stop_patience} epochs")
            if epochs_no_improve >= Config.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_wts)
    print(f"Loaded best model with AUC: {best_val_auc:.4f}")
    return history


# ==================== Training curves ====================
def plot_history(history):
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['epoch'], history['train_auc'], label='Train AUC')
    plt.plot(history['epoch'], history['val_auc'], label='Val AUC')
    plt.title('AUC'); plt.xlabel('Epoch'); plt.ylabel('AUC')
    plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
    plt.plot(history['epoch'], history['val_acc'], label='Val Acc')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['epoch'], history['lr'], label='LR')
    plt.title('Learning Rate'); plt.xlabel('Epoch'); plt.ylabel('LR')
    plt.yscale('log'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(Config.plot_save_path)
    plt.close()


# ==================== Main ====================
def main():
    save_config()

    print("=" * 60)
    print("STEP 1: Computing normalization statistics from training set")
    print("=" * 60)
    dummy_transform = transforms.Compose([transforms.Lambda(lambda x: x)])
    temp_dataset = KneeMRIDataset(Config.train_dir, transform=dummy_transform)

    stats_calc = DatasetStatsCalculator(
        temp_dataset,
        sample_pixels=1_000_000,
        force_recompute=False
    )

    print("\n" + "=" * 60)
    print("STEP 2: Building data transforms (min-max to [0,1])")
    print("=" * 60)
    fixed_norm = FixedNormalize(low=stats_calc.low, high=stats_calc.high)

    # Normalize to [0,1] first so that brightness/contrast augmentation
    # operates on a meaningful intensity range.
    train_transform = transforms.Compose([
        fixed_norm,
        RandomAugment(Config.image_size),
    ])

    val_transform = transforms.Compose([
        ResizeLambda(Config.image_size),
        fixed_norm,
    ])
    test_transform = val_transform

    print("\n" + "=" * 60)
    print("STEP 3: Creating final datasets")
    print("=" * 60)
    train_dataset = KneeMRIDataset(Config.train_dir, train_transform)
    train_dataset_eval = KneeMRIDataset(Config.train_dir, val_transform)
    val_dataset = KneeMRIDataset(Config.val_dir, val_transform)
    test_dataset_eval = KneeMRIDataset(Config.test_dir, test_transform, return_path=False)

    print(f"Train samples (aug): {len(train_dataset)}")
    print(f"Train samples (eval): {len(train_dataset_eval)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test eval samples: {len(test_dataset_eval)}")

    print("\n" + "=" * 60)
    print("STEP 4: Creating data loaders")
    print("=" * 60)

    class_counts = np.bincount(train_dataset.labels)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    samples_weights = weights[train_dataset.labels]
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )

    pin_memory = Config.device.type == 'cuda'
    persistent = Config.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        sampler=sampler,
        num_workers=Config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent,
    )

    train_loader_eval = DataLoader(
        train_dataset_eval,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    test_loader_eval = DataLoader(
        test_dataset_eval,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    print(f"Train loader: {len(train_loader)} batches (bs={Config.batch_size})")
    print(f"Val   loader: {len(val_loader)} batches")
    print(f"Test  loader: {len(test_loader_eval)} batches")

    print("\n" + "=" * 60)
    print("STEP 5: Training model")
    print("=" * 60)
    model = load_pretrained_resnet()
    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")

    with torch.no_grad():
        dummy = torch.randn(2, 3, Config.image_size, Config.image_size).to(Config.device)
        dummy_out = model(dummy)
        print(f"Sanity check output shape: {tuple(dummy_out.shape)}  (expected (2, 1))")

    criterion = LabelSmoothingBCE(smoothing=Config.smoothing)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=Config.lr, weight_decay=Config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    print(f"Device: {Config.device}")
    print(f"Model: {Config.model_name}, freeze_backbone={Config.freeze_backbone}")
    print(f"Label smoothing: {Config.smoothing}, AMP: {Config.use_amp}")

    history = train_model(model, train_loader, val_loader,
                          criterion, optimizer, scheduler)
    plot_history(history)

    print("\n" + "=" * 60)
    print("STEP 6: Evaluating best model")
    print("=" * 60)
    try:
        checkpoint = torch.load(Config.model_save_path, map_location=Config.device)
    except Exception:
        checkpoint = torch.load(Config.model_save_path, map_location=Config.device,
                                weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} "
          f"with val AUC {checkpoint['val_auc']:.4f}")

    _, _, _, _, _, val_probs, val_labels = evaluate(
        model, val_loader, criterion, Config.device,
        set_name="validation_tmp", save_report=False, threshold=0.5
    )

    best_threshold = find_best_threshold(val_labels, val_probs)
    print(f"\nBest threshold from validation set (Youden): {best_threshold:.4f}")
    with open(os.path.join(Config.output_dir, "best_threshold.txt"), "w") as f:
        f.write(f"{best_threshold:.6f}\n")

    evaluate(model, train_loader_eval, criterion, Config.device,
             set_name="train", save_report=True, threshold=best_threshold)

    evaluate(model, val_loader, criterion, Config.device,
             set_name="validation", save_report=True, threshold=best_threshold)

    evaluate(model, test_loader_eval, criterion, Config.device,
             set_name="test", save_report=True, threshold=best_threshold)

    print("\n" + "=" * 60)
    print("All done! Results saved in:", Config.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()