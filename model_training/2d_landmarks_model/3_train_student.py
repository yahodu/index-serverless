# 3_train_student_2d106.py
"""
Step 3: Train a student 2D 106-point landmark model via knowledge distillation.

TASK
────
  Teacher : 2d106det.onnx  MobileNet-0.5  1.2 M params  5 MB
  Student : Tinier CNN  output (N, 106, 2) raw pixel coordinates
  Labels  : decoded teacher landmarks stored by 2_prepare_dataset_2d106.py

KEY DIFFERENCES vs 1k3d68 distillation
────────────────────────────────────────
  - Output is (N, 106, 2) not (N, 68, 3) — no z dimension
  - No z weighting needed in the loss
  - Teacher is already small (MobileNet-0.5) — student goal is to be TINIER
    (MobileNet-0.25 / ShuffleNet / custom ultra-lite backbone)
  - NME normalised by inter-ocular distance (2D)
    106-point eye corners: pts 60,61,63,64 (left) and 68,69,71,72 (right)
    (InsightFace 106-pt indexing — see 2d106markup.jpg)

LOSS
────
  Wing loss  on xy  — standard face alignment loss
  L1  loss   on xy  — robust to outliers
  MSE loss   on xy  — hard penalty on large errors

METRIC
──────
  NME% — Normalised Mean Error, inter-ocular distance normalised.
  Teacher on aligned crops: ~1–2% (already a lightweight model).
  Good student target: NME < 3%.

AUGMENTATION NOTE
─────────────────
  Horizontal flip OMITTED — requires remapping 106 landmark pairs
  (e.g. left-eye pts ↔ right-eye pts), non-trivial.
  ColorJitter and GaussianBlur are safe.
"""

import os
import gc
import glob
import math
import json
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime


# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_SIZE = 192
NUM_LMK    = 106
LMK_DIM    = 2   # 2D only


# ══════════════════════════════════════════════════════════════════════════════
# Student Models
# Three options targeting different speed/accuracy tradeoffs:
#   ultra : ~0.3 M params — beats MobileNet-0.25 teacher on edge devices
#   small : ~1.0 M params — matches teacher size with different architecture
#   medium: ~3.5 M params — surpasses teacher accuracy
# ══════════════════════════════════════════════════════════════════════════════

class ConvBnAct(nn.Module):
    """Conv2d -> BN -> ReLU6 (mobile-friendly activation)"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparable(nn.Module):
    """
    Depthwise-separable convolution (MobileNet-style).
    Depthwise (groups=in_ch) + pointwise (1×1).
    ~8–9× fewer FLOPs than a standard 3×3 conv.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = ConvBnAct(in_ch, in_ch,  k=3, s=stride, p=1, groups=in_ch)
        self.pw = ConvBnAct(in_ch, out_ch, k=1, s=1,      p=0)

    def forward(self, x):
        return self.pw(self.dw(x))


class InvertedResidual(nn.Module):
    """
    MobileNetV2 inverted residual block.
    Expand -> depthwise -> project (squeeze).
    Residual connection only when stride=1 and in_ch==out_ch.
    """
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        mid_ch      = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBnAct(in_ch, mid_ch, k=1, s=1, p=0))
        layers += [
            ConvBnAct(mid_ch, mid_ch, k=3, s=stride, p=1, groups=mid_ch),
            nn.Conv2d(mid_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


def _make_mbv2_layer(in_ch, out_ch, n, stride, expand_ratio=6):
    layers = [InvertedResidual(in_ch, out_ch, stride=stride,
                               expand_ratio=expand_ratio)]
    for _ in range(1, n):
        layers.append(InvertedResidual(out_ch, out_ch, stride=1,
                                       expand_ratio=expand_ratio))
    return nn.Sequential(*layers)


class LandmarkHead2D(nn.Module):
    """
    Regression head -> (N, 106, 2).
    Initialised to predict near the image centre (96, 96).
    """
    def __init__(self, in_features: int, dropout: float = 0.0):
        super().__init__()
        layers: list = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(in_features, NUM_LMK * LMK_DIM))
        self.fc = nn.Sequential(*layers)

        final_linear = self.fc[-1]
        nn.init.normal_(final_linear.weight, std=0.01)
        nn.init.constant_(final_linear.bias, 96.0)   # image centre

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).view(x.size(0), NUM_LMK, LMK_DIM)


# ── Ultra-lite (~0.3 M params) ────────────────────────────────────────────────
class StudentModelUltra(nn.Module):
    """
    ~0.3 M params. Depthwise-separable only.
    Suitable for mobile/edge deployment where teacher (1.2 M) is too heavy.

    Stride schedule: 192 -> 96 -> 48 -> 24 -> 12 -> 6
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnAct(3,  16, s=2),              # 192->96   std conv stem
            DepthwiseSeparable(16,  32, stride=2),  # 96->48
            DepthwiseSeparable(32,  64, stride=2),  # 48->24
            DepthwiseSeparable(64,  96, stride=2),  # 24->12
            DepthwiseSeparable(96, 128, stride=2),  # 12->6
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = LandmarkHead2D(128, dropout=0.1)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


# ── Small (~1.0 M params, same order as teacher) ──────────────────────────────
class StudentModelSmall(nn.Module):
    """
    ~1.0 M params. MobileNetV2-style with reduced channels.
    Approximately teacher-sized but different architecture (no MXNet export quirks).

    Channel schedule mirrors MobileNet-0.5 width multiplier.
    """
    def __init__(self):
        super().__init__()
        # in_ch, out_ch, n_blocks, stride, expand_ratio
        cfg = [
            # stride 2: 192->96
            (3,   16,  1, 2, 1),    # conv stem (expand_ratio=1 = no expand)
            (16,  16,  1, 1, 6),
            (16,  24,  2, 2, 6),    # 96->48
            (24,  32,  3, 2, 6),    # 48->24
            (32,  64,  4, 2, 6),    # 24->12
            (64,  96,  3, 1, 6),
            (96, 128,  3, 2, 6),    # 12->6
            (128,256,  1, 1, 6),
        ]
        layers = []
        in_ch  = 3
        for i, (ic, oc, n, s, er) in enumerate(cfg):
            if i == 0:
                layers.append(ConvBnAct(ic, oc, k=3, s=s, p=1))
            else:
                layers.append(_make_mbv2_layer(ic, oc, n, s, er))
            in_ch = oc

        self.features = nn.Sequential(*layers)
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.head     = LandmarkHead2D(256, dropout=0.2)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


# ── Medium (~3.5 M params, higher accuracy target) ────────────────────────────
class StudentModelMedium(nn.Module):
    """
    ~3.5 M params. MobileNetV2 full-width (width_mult=1.0).
    Expected to surpass teacher accuracy despite different training data.
    """
    def __init__(self):
        super().__init__()
        # Standard MobileNetV2 channel schedule (width_mult=1.0)
        self.features = nn.Sequential(
            ConvBnAct(3,   32, k=3, s=2, p=1),     # 192->96
            _make_mbv2_layer(32,  16, 1, 1, 1),
            _make_mbv2_layer(16,  24, 2, 2, 6),     # 96->48
            _make_mbv2_layer(24,  32, 3, 2, 6),     # 48->24
            _make_mbv2_layer(32,  64, 4, 2, 6),     # 24->12
            _make_mbv2_layer(64,  96, 3, 1, 6),
            _make_mbv2_layer(96, 160, 3, 2, 6),     # 12->6
            _make_mbv2_layer(160,320, 1, 1, 6),
            ConvBnAct(320, 512, k=1, s=1, p=0),    # pointwise expansion
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = LandmarkHead2D(512, dropout=0.3)

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))


# ══════════════════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════════════════

class LandmarkDistillationLoss2D(nn.Module):
    """
    Wing loss + L1 + MSE on xy landmarks (2D only, no z).

    Wing loss ref: Feng et al., CVPR 2018
    """

    def __init__(
        self,
        alpha_wing: float = 1.0,
        alpha_l1:   float = 1.0,
        alpha_mse:  float = 0.1,
        wing_w:     float = 10.0,
        wing_eps:   float = 2.0,
    ):
        super().__init__()
        self.alpha_wing = alpha_wing
        self.alpha_l1   = alpha_l1
        self.alpha_mse  = alpha_mse
        self.wing_w     = wing_w
        self.wing_eps   = wing_eps
        self._wing_C    = wing_w - wing_w * math.log(1.0 + wing_w / wing_eps)

    def _wing(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Wing loss. pred/target: (N, 106, 2)"""
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.wing_w,
            self.wing_w * torch.log(1.0 + diff / self.wing_eps),
            diff - self._wing_C,
        )
        return loss.mean()

    def forward(
        self,
        pred:   torch.Tensor,   # (N, 106, 2)
        target: torch.Tensor,   # (N, 106, 2)
    ) -> tuple:
        loss_wing = self._wing(pred, target)
        loss_l1   = F.l1_loss(pred, target)
        loss_mse  = F.mse_loss(pred, target)

        total = (
            self.alpha_wing * loss_wing
            + self.alpha_l1  * loss_l1
            + self.alpha_mse * loss_mse
        )

        return total, {
            'wing':  loss_wing.item(),
            'l1':    loss_l1.item(),
            'mse':   loss_mse.item(),
            'total': total.item(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Metric: NME% (2D, 106-point scheme)
# ══════════════════════════════════════════════════════════════════════════════

# InsightFace 106-point eye landmark indices
# Left  eye outer corners : 60, 64    (approx; use mean of inner region)
# Right eye outer corners : 68, 72
# For NME we use the eye-centre approximation:
#   left_eye  = mean of pts 60..64  (5 pts)
#   right_eye = mean of pts 68..72  (5 pts)
_LEFT_EYE_SLICE  = slice(60, 65)   # pts 60,61,62,63,64
_RIGHT_EYE_SLICE = slice(68, 73)   # pts 68,69,70,71,72


def compute_nme_2d(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Normalised Mean Error (%) using inter-ocular distance.

    106-point InsightFace scheme eye indices:
      Left  eye region : points 60–64 (5 points)
      Right eye region : points 68–72 (5 points)

    NME = mean(||pred_i - target_i||_2) / IOD * 100%

    Lower is better. Teacher on aligned crops: ~1–2%.
    Good student target: < 3%.
    """
    left_eye  = target[:, _LEFT_EYE_SLICE,  :].mean(dim=1)   # (N, 2)
    right_eye = target[:, _RIGHT_EYE_SLICE, :].mean(dim=1)   # (N, 2)
    iod       = (left_eye - right_eye).norm(dim=1).clamp(min=1e-6)

    diff     = (pred - target).norm(dim=2)   # (N, 106)
    mean_err = diff.mean(dim=1)              # (N,)

    return (mean_err / iod).mean().item() * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# RAM helpers (identical to 1k3d68 pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_dataset_ram_gb(data_dir: str) -> float:
    chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
    if not chunk_files:
        raise ValueError(f"No chunk files found in {data_dir}")
    total = 0
    for f in chunk_files:
        d = np.load(f, mmap_mode='r')
        total += d['images'].nbytes + d['landmarks'].nbytes
        d.close()
    return total / 1e9


def should_stream(data_dir: str, headroom_gb: float = 40.0) -> bool:
    avail_gb   = psutil.virtual_memory().available / 1e9
    dataset_gb = estimate_dataset_ram_gb(data_dir)
    fits       = dataset_gb < (avail_gb - headroom_gb)
    print(f"  Dataset  : {dataset_gb:.1f} GB")
    print(f"  Available: {avail_gb:.1f} GB")
    print(f"  Headroom : {headroom_gb:.1f} GB")
    print(f"  Mode     : {'in-RAM' if fits else 'streaming'}")
    return not fits


# ══════════════════════════════════════════════════════════════════════════════
# Augmentation
# ══════════════════════════════════════════════════════════════════════════════

def _build_transform(augment: bool) -> transforms.Compose:
    """
    Safe colour augmentations only. Spatial transforms excluded because:
      - RandomHorizontalFlip: requires remapping 106-pt landmark pairs
      - RandomRotation       : requires rotating all landmark coordinates
    """
    steps = [transforms.ToPILImage()]
    if augment:
        steps += [
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.3,
                    hue=0.06,
                ),
            ], p=0.6),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ], p=0.25),
            transforms.RandomGrayscale(p=0.05),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(steps)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset — in-memory
# ══════════════════════════════════════════════════════════════════════════════

class DistillationDataset2D(Dataset):
    """
    Loads all chunks into RAM.
    Landmarks shape: (N, 106, 2) float32.
    """

    def __init__(self, data_dir: str, augment: bool = True,
                 val_split: float = 0.01):
        self.transform     = _build_transform(augment)
        self.transform_val = _build_transform(False)

        chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
        if not chunk_files:
            raise ValueError(f"No chunk files found in {data_dir}")

        print(f"Loading {len(chunk_files)} chunks into RAM...")
        all_images    = []
        all_landmarks = []

        for f in tqdm(chunk_files, desc="Loading"):
            d = np.load(f)
            all_images.append(d['images'].copy())
            all_landmarks.append(d['landmarks'].copy())
            d.close()

        images    = np.concatenate(all_images,    axis=0)
        landmarks = np.concatenate(all_landmarks, axis=0)
        del all_images, all_landmarks
        gc.collect()

        # Validate landmark shape
        assert landmarks.shape[1:] == (NUM_LMK, LMK_DIM), (
            f"Expected landmark shape (N, {NUM_LMK}, {LMK_DIM}), "
            f"got {landmarks.shape}"
        )

        n_val              = max(1, int(len(images) * val_split))
        self.val_images    = images[:n_val]
        self.val_landmarks = landmarks[:n_val]
        self.images        = images[n_val:]
        self.landmarks     = landmarks[n_val:]

        print(f"Train: {len(self.images):,}  Val: {len(self.val_images):,}")
        print(f"RAM   : images={images.nbytes/1e9:.1f} GB  "
              f"landmarks={landmarks.nbytes/1e6:.1f} MB  (2D, no z)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.transform(self.images[idx])
        lmk = torch.tensor(self.landmarks[idx], dtype=torch.float32)
        return img, lmk

    def get_val_dataset(self):
        ds            = object.__new__(DistillationDataset2D)
        ds.transform  = self.transform_val
        ds.images     = self.val_images
        ds.landmarks  = self.val_landmarks
        return ds


# ══════════════════════════════════════════════════════════════════════════════
# Dataset — streaming
# ══════════════════════════════════════════════════════════════════════════════

class LargeDistillationDataset2D(IterableDataset):
    """Streams chunks from disk. Holds one chunk in RAM at a time."""

    def __init__(self, data_dir: str, augment: bool = True,
                 shuffle: bool = True):
        self.transform   = _build_transform(augment)
        self.shuffle     = shuffle
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {data_dir}")

        print(f"Indexing {len(self.chunk_files)} chunks...")
        self._total = 0
        for f in tqdm(self.chunk_files, desc="Indexing"):
            d = np.load(f, mmap_mode='r')
            self._total += len(d['landmarks'])
            d.close()
        print(f"Total samples: {self._total:,}")

    def __len__(self):
        return self._total

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        chunk_indices = (
            list(range(len(self.chunk_files)))
            if worker_info is None
            else list(range(len(self.chunk_files)))[
                worker_info.id :: worker_info.num_workers
            ]
        )
        if self.shuffle:
            np.random.shuffle(chunk_indices)

        for ci in chunk_indices:
            d         = np.load(self.chunk_files[ci])
            images    = d['images'].copy()
            landmarks = d['landmarks'].copy()
            d.close()

            indices = list(range(len(images)))
            if self.shuffle:
                np.random.shuffle(indices)

            for i in indices:
                img = self.transform(images[i])
                lmk = torch.tensor(landmarks[i], dtype=torch.float32)
                yield img, lmk

            del images, landmarks
            gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train(config: dict):
    print("=" * 60)
    print("KNOWLEDGE DISTILLATION — 2D 106-POINT LANDMARK (2d106det)")
    print("=" * 60)
    print(json.dumps(config, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name()}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Output dir ────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config['save_dir'], f"run_{ts}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ── Dataset ───────────────────────────────────────────────
    print("\nChecking dataset vs available RAM...")
    stream = config['stream_dataset']
    if stream == 'auto':
        stream = should_stream(config['data_dir'], headroom_gb=40.0)
    else:
        gb = estimate_dataset_ram_gb(config['data_dir'])
        av = psutil.virtual_memory().available / 1e9
        print(f"  Dataset  : {gb:.1f} GB   Available: {av:.1f} GB")
        if not stream and gb > av - 40.0:
            print(f"  WARNING : stream_dataset=False but dataset may exceed RAM.")

    if stream:
        print("\nUsing LargeDistillationDataset2D (streaming)")
        train_ds       = LargeDistillationDataset2D(
            config['data_dir'], augment=True, shuffle=True
        )
        val_ds         = LargeDistillationDataset2D(
            config['data_dir'], augment=False, shuffle=False
        )
        val_ds.chunk_files = val_ds.chunk_files[:1]
        val_ds._total      = min(10_000, val_ds._total)
        shuffle_loader = False
    else:
        print("\nUsing DistillationDataset2D (all in RAM)")
        full_ds        = DistillationDataset2D(
            config['data_dir'], augment=True, val_split=0.01
        )
        train_ds       = full_ds
        val_ds         = full_ds.get_val_dataset()
        shuffle_loader = True

    train_loader = DataLoader(
        train_ds,
        batch_size         = config['batch_size'],
        shuffle            = shuffle_loader,
        num_workers        = config['num_workers'],
        pin_memory         = True,
        pin_memory_device  = 'cuda:0',
        drop_last          = True,
        persistent_workers = True,
        prefetch_factor    = 4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = config['batch_size'],
        shuffle     = False,
        num_workers = min(4, config['num_workers']),
        pin_memory  = True,
        drop_last   = False,
    )

    # ── Model ─────────────────────────────────────────────────
    model_map = {
        'ultra' : StudentModelUltra,
        'small' : StudentModelSmall,
        'medium': StudentModelMedium,
    }
    if config['model_size'] not in model_map:
        raise ValueError(
            f"Unknown model_size '{config['model_size']}'. "
            f"Choose from: {list(model_map.keys())}"
        )
    model = model_map[config['model_size']]().to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel      : StudentModel2D_{config['model_size'].capitalize()}")
    print(f"Parameters : {total_p:,} total   {trainable_p:,} trainable")
    print(f"Teacher    : ~1,200,000 params (MobileNet-0.5)")
    if total_p < 1_200_000:
        print(f"  -> Student is {1_200_000 / total_p:.1f}× smaller than teacher!")

    # ── Loss ──────────────────────────────────────────────────
    criterion = LandmarkDistillationLoss2D(
        alpha_wing = config['alpha_wing'],
        alpha_l1   = config['alpha_l1'],
        alpha_mse  = config['alpha_mse'],
    )

    # ── Optimiser ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config['learning_rate'],
        weight_decay = config['weight_decay'],
    )

    # ── LR schedule: linear warmup + cosine decay ─────────────
    total_steps  = len(train_loader) * config['epochs']
    warmup_steps = len(train_loader) * config.get('warmup_epochs', 3)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Mixed precision ───────────────────────────────────────
    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    # ── Resume ────────────────────────────────────────────────
    start_epoch = 0
    best_nme    = float('inf')

    if config.get('resume_from'):
        ckpt = torch.load(config['resume_from'], map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_nme    = ckpt.get('nme', float('inf'))
        print(f"Resumed from {config['resume_from']} (epoch {start_epoch})")

    print(f"\nStarting training for {config['epochs']} epochs")
    print(f"Total steps  : {total_steps:,}")
    print(f"Warmup steps : {warmup_steps:,}")

    history = []

    for epoch in range(start_epoch, config['epochs']):

        # ── Train ─────────────────────────────────────────────
        model.train()
        ep_losses = {'wing': 0., 'l1': 0., 'mse': 0., 'total': 0.}
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc  = f"Epoch {epoch+1}/{config['epochs']}",
            leave = True,
        )

        for images, teacher_lmk in pbar:
            images      = images.to(device, non_blocking=True)
            teacher_lmk = teacher_lmk.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    student_lmk     = model(images)
                    loss, loss_dict = criterion(student_lmk, teacher_lmk)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                student_lmk     = model(images)
                loss, loss_dict = criterion(student_lmk, teacher_lmk)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            for k, v in loss_dict.items():
                ep_losses[k] += v
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'wing': f"{loss_dict['wing']:.4f}",
                'l1':   f"{loss_dict['l1']:.4f}",
                'lr':   f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg = {k: v / n_batches for k, v in ep_losses.items()}

        # ── Validation NME ────────────────────────────────────
        model.eval()
        nme_list = []
        with torch.no_grad():
            for images, teacher_lmk in val_loader:
                images      = images.to(device)
                teacher_lmk = teacher_lmk.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        student_lmk = model(images)
                else:
                    student_lmk = model(images)
                nme_list.append(compute_nme_2d(student_lmk, teacher_lmk))

        val_nme = float(np.mean(nme_list))

        print(f"\nEpoch {epoch+1:>3} | "
              f"loss={avg['total']:.5f}  "
              f"wing={avg['wing']:.5f}  "
              f"l1={avg['l1']:.5f}  "
              f"mse={avg['mse']:.5f}  "
              f"NME={val_nme:.3f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        history.append({'epoch': epoch + 1, 'nme': val_nme, **avg})

        # ── Checkpointing ─────────────────────────────────────
        payload = {
            'epoch':                epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'nme':                  val_nme,
            'loss':                 avg['total'],
            'config':               config,
        }

        if val_nme < best_nme:
            best_nme = val_nme
            torch.save(payload, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ Best model saved  NME={best_nme:.3f}%")

        if (epoch + 1) % config.get('save_every', 5) == 0:
            torch.save(
                payload,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:04d}.pth'),
            )
            print(f"  ✓ Checkpoint saved")

    # ── Final save ────────────────────────────────────────────
    torch.save(
        {
            'epoch':            config['epochs'],
            'model_state_dict': model.state_dict(),
            'config':           config,
        },
        os.path.join(save_dir, 'final_model.pth'),
    )
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best NME : {best_nme:.3f}%")
    print(f"Saved to : {save_dir}")
    print(f"{'='*60}")

    return save_dir


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    config = {
        # ── Data ──────────────────────────────────────────────
        'data_dir':       'distill_data/ms1m_landmarks_2d106',
        'num_workers':    16,
        'stream_dataset': 'auto',

        # ── Model ─────────────────────────────────────────────
        # 'ultra'  : ~0.3 M params — smallest, for edge/mobile
        # 'small'  : ~1.0 M params — same scale as teacher
        # 'medium' : ~3.5 M params — highest accuracy
        'model_size':     'medium',

        # ── Training ──────────────────────────────────────────
        # 2d106det images are 192×192 — same as 1k3d68.
        # The student is SMALLER so we can afford larger batches.
        # ultra/small: bs=4096 easily fits on 24 GB GPU
        # medium     : bs=2048 recommended
        'batch_size':     4096,
        'epochs':         60,
        # LR: sqrt-scaled from 1e-3 @ bs=256 → 1e-3 * sqrt(4096/256) ≈ 4e-3
        # In practice, 2e-3 to 3e-3 works well for MobileNet-style models.
        'learning_rate':  2e-3,
        'weight_decay':   1e-4,
        'warmup_epochs':  5,

        # ── Loss ──────────────────────────────────────────────
        'alpha_wing':     1.0,   # Wing loss — primary landmark term
        'alpha_l1':       1.0,   # L1  loss  — robust to outliers
        'alpha_mse':      0.1,   # MSE loss  — large-error penalty

        # ── Saving ────────────────────────────────────────────
        'save_dir':       'checkpoints_2d106',
        'save_every':     5,

        # ── Resume ────────────────────────────────────────────
        'resume_from':    None,
    }

    train(config)
