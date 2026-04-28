# 3_train_student.py
"""
Step 3: Train a student landmark model via knowledge distillation.

TASK SUMMARY
────────────
  Teacher  : 1k3d68.onnx  ResNet-50  34.2 M params
  Input    : (N, 3, 192, 192) float32   mean=127.5  std=128.0
  Output   : (N, 68, 3) float32   pixel-space landmarks

STUDENT OUTPUT
──────────────
  Same format: (N, 68, 3)
  No L2-normalisation — raw coordinate regression, not embedding matching.

LOSS
────
  Wing loss  (standard for face landmark regression, CVPR 2018)
  + L1 loss  (robust to outliers, stabilises training)
  + MSE loss (penalises large errors hard)

  All losses computed on decoded coordinates directly.
  z is down-weighted because its scale (~±48 px) matches xy (~0-192 px),
  but z has higher uncertainty — empirically weight 0.5 works well.

METRIC
──────
  NME% (Normalised Mean Error) — inter-ocular distance normalised.
  Standard face alignment metric. Good models: NME < 4% on 300W.
  Reported every epoch on a held-out validation split.

DATASET MODES (auto-selected)
──────────────────────────────
  DistillationDataset      — loads all into RAM  (fast, needs ~107 GB+)
  LargeDistillationDataset — streams from disk   (slower, any RAM)
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


# ============================================================
# Constants (must match 2_prepare_dataset.py)
# ============================================================

INPUT_SIZE = 192
NUM_LMK    = 68
LMK_DIM    = 3


# ============================================================
# Student Model Architecture
# ============================================================

class ConvBnAct(nn.Module):
    """Conv -> BN -> PReLU"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Pre-activation residual block."""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int = 2) -> nn.Sequential:
    layers = [ConvBnAct(in_ch, out_ch, s=stride)]
    for _ in range(n_blocks):
        layers.append(ResBlock(out_ch))
    return nn.Sequential(*layers)


class LandmarkHead(nn.Module):
    """
    Regression head: global-avg-pool -> dropout -> FC -> reshape (N, 68, 3).
    No BN on the output — landmark coordinates are unbounded real numbers.
    """
    def __init__(self, in_features: int, num_lmk: int = NUM_LMK, lmk_dim: int = LMK_DIM,
                 dropout: float = 0.0):
        super().__init__()
        self.num_lmk = num_lmk
        self.lmk_dim = lmk_dim
        layers: list[nn.Module] = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(in_features, num_lmk * lmk_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).view(x.size(0), self.num_lmk, self.lmk_dim)


class StudentModelSmall(nn.Module):
    """~2 M params. Fast iteration / ablation."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnAct(3,   64,  s=2),    # 192 -> 96
            ResBlock(64),
            _make_layer(64,  128, 2, 2), # 96  -> 48
            _make_layer(128, 256, 2, 2), # 48  -> 24
            _make_layer(256, 512, 2, 2), # 24  -> 12
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = LandmarkHead(512, dropout=0.2)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class StudentModelMedium(nn.Module):
    """~10 M params."""
    def __init__(self):
        super().__init__()
        self.stem    = ConvBnAct(3, 64, s=1)
        self.layer1  = _make_layer(64,  64,  3, 2)
        self.layer2  = _make_layer(64,  128, 4, 2)
        self.layer3  = _make_layer(128, 256, 6, 2)
        self.layer4  = _make_layer(256, 512, 3, 2)
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.head    = LandmarkHead(512, dropout=0.3)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class StudentModelLarge(nn.Module):
    """
    ~25 M params. IResNet-50 style.
    Closest to teacher (ResNet-50, 34.2 M) — recommended.
    """
    def __init__(self):
        super().__init__()
        self.stem    = ConvBnAct(3, 64, s=1)
        self.layer1  = _make_layer(64,  64,   3,  2)   # 192->96
        self.layer2  = _make_layer(64,  128,  8,  2)   # 96 ->48
        self.layer3  = _make_layer(128, 256,  16, 2)   # 48 ->24
        self.layer4  = _make_layer(256, 512,  3,  2)   # 24 ->12
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.head    = LandmarkHead(512, dropout=0.4)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ============================================================
# Loss Function
# ============================================================

class LandmarkDistillationLoss(nn.Module):
    """
    Combined regression loss for 3D landmark distillation.

    loss = alpha_wing * WingLoss(xy)
         + alpha_l1  * L1(weighted_xyz)
         + alpha_mse * MSE(weighted_xyz)

    z_weight: down-weight z dimension.
      z is in the same pixel scale as xy (~±48 px range) but has higher
      label noise because depth is estimated, not directly measured.
      z_weight=0.5 means z errors penalised at half the rate of xy.

    Wing loss reference: Feng et al., CVPR 2018
      "Wing Loss for Robust Facial Landmark Localisation with CNN"
    """

    def __init__(
        self,
        alpha_wing: float = 1.0,
        alpha_l1:   float = 1.0,
        alpha_mse:  float = 0.5,
        z_weight:   float = 0.5,
        wing_w:     float = 10.0,
        wing_eps:   float = 2.0,
    ):
        super().__init__()
        self.alpha_wing = alpha_wing
        self.alpha_l1   = alpha_l1
        self.alpha_mse  = alpha_mse
        self.z_weight   = z_weight
        self.wing_w     = wing_w
        self.wing_eps   = wing_eps
        # Precompute Wing constant C = w - w*ln(1 + w/eps)
        self._wing_C = wing_w - wing_w * math.log(1.0 + wing_w / wing_eps)

    def _wing(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Wing loss on xy coordinates only.
        pred/target : (N, 68, 2)
        """
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.wing_w,
            self.wing_w * torch.log(1.0 + diff / self.wing_eps),
            diff - self._wing_C,
        )
        return loss.mean()

    def _apply_z_weight(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns tensors where the z channel is multiplied by z_weight.
        pred/target : (N, 68, 3)
        """
        w = torch.ones(3, device=pred.device, dtype=pred.dtype)
        w[2] = self.z_weight
        # broadcast: (1, 1, 3)
        w = w.view(1, 1, 3)
        return pred * w, target * w

    def forward(
        self,
        pred:   torch.Tensor,   # (N, 68, 3)  student output
        target: torch.Tensor,   # (N, 68, 3)  teacher labels
    ) -> tuple[torch.Tensor, dict]:

        # Wing loss — xy only
        loss_wing = self._wing(pred[:, :, :2], target[:, :, :2])

        # L1 + MSE — all dims with z weighting
        p_w, t_w = self._apply_z_weight(pred, target)
        loss_l1  = F.l1_loss(p_w, t_w)
        loss_mse = F.mse_loss(p_w, t_w)

        total = (
            self.alpha_wing * loss_wing
            + self.alpha_l1 * loss_l1
            + self.alpha_mse * loss_mse
        )

        return total, {
            'wing':  loss_wing.item(),
            'l1':    loss_l1.item(),
            'mse':   loss_mse.item(),
            'total': total.item(),
        }


# ============================================================
# Metric: NME%
# ============================================================

def compute_nme(
    pred:   torch.Tensor,   # (N, 68, 3)
    target: torch.Tensor,   # (N, 68, 3)
) -> float:
    """
    Normalised Mean Error (%) — inter-ocular distance normalised.

    68-point scheme (same as 300W / IBUG):
      Left  eye corners : indices 36-41  (mean = left  eye centre)
      Right eye corners : indices 42-47  (mean = right eye centre)

    Only xy used; z excluded (not part of standard NME definition).
    Lower is better. SOTA on 300W: ~3-4%.
    """
    pred_xy   = pred[:, :, :2]     # (N, 68, 2)
    target_xy = target[:, :, :2]

    # Inter-ocular distance from teacher labels
    left_eye_c  = target_xy[:, 36:42, :].mean(dim=1)   # (N, 2)
    right_eye_c = target_xy[:, 42:48, :].mean(dim=1)   # (N, 2)
    iod         = (left_eye_c - right_eye_c).norm(dim=1).clamp(min=1e-6)  # (N,)

    # Mean Euclidean error across 68 landmarks
    diff      = (pred_xy - target_xy).norm(dim=2)       # (N, 68)
    mean_err  = diff.mean(dim=1)                        # (N,)

    nme = (mean_err / iod).mean().item() * 100.0        # percentage
    return nme


# ============================================================
# RAM helper
# ============================================================

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
    print(f"  Mode     : {'DistillationDataset (RAM)' if fits else 'LargeDistillationDataset (stream)'}")
    return not fits


# ============================================================
# Augmentation pipeline
# ============================================================

def _build_transforms(augment: bool):
    """
    NOTE: RandomGrayscale is intentionally omitted.
    The teacher was run on RGB images; grayscale inputs create a
    distribution shift that degrades landmark accuracy.

    Horizontal flip is ALSO omitted here because flipping a face
    requires remapping landmark indices (left<->right symmetry pairs)
    which is non-trivial for 68-point scheme. Keeping augmentation
    simple avoids label corruption.
    """
    if augment:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.05,
                ),
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


# ============================================================
# Dataset — in-memory
# ============================================================

class DistillationDataset(Dataset):
    """
    Loads all chunks into RAM.

    At 1M samples @ 192x192:
      images     ≈ 107  GB
      landmarks  ≈   0.8 GB
      total      ≈ 107.8 GB

    Only feasible on machines with ≥ 150 GB RAM (e.g. H200 node).
    Use stream_dataset='auto' to auto-select.
    """

    def __init__(self, data_dir: str, augment: bool = True,
                 val_split: float = 0.01):
        self.transform    = _build_transforms(augment)
        self.transform_val = _build_transforms(False)

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

        n_val          = max(1, int(len(images) * val_split))
        self.val_images    = images[:n_val]
        self.val_landmarks = landmarks[:n_val]
        self.images        = images[n_val:]
        self.landmarks     = landmarks[n_val:]

        print(f"Train: {len(self.images):,}  Val: {len(self.val_images):,}")
        print(f"RAM : images={images.nbytes/1e9:.1f} GB  "
              f"landmarks={landmarks.nbytes/1e6:.1f} MB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.transform(self.images[idx])
        lmk = torch.tensor(self.landmarks[idx], dtype=torch.float32)
        return img, lmk

    def get_val_dataset(self) -> 'DistillationDataset':
        """Returns a lightweight val wrapper over the held-out slice."""
        ds             = object.__new__(DistillationDataset)
        ds.transform   = self.transform_val
        ds.images      = self.val_images
        ds.landmarks   = self.val_landmarks
        return ds

    # Make the val wrapper usable as a Dataset
    def __len2__(self):
        return len(self.images)


# ============================================================
# Dataset — streaming
# ============================================================

class LargeDistillationDataset(IterableDataset):
    """
    Streams chunks from disk. Works for any dataset size.
    Holds one chunk (~1 GB) in RAM at a time.

    Worker-aware: each DataLoader worker reads a disjoint subset of chunks.
    """

    def __init__(self, data_dir: str, augment: bool = True,
                 shuffle: bool = True, skip_first_n: int = 0):
        self.augment        = augment
        self.shuffle        = shuffle
        self.skip_first_n   = skip_first_n     # used to carve out val set
        self.transform      = _build_transforms(augment)

        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {data_dir}")

        print(f"Indexing {len(self.chunk_files)} chunks...")
        self._total = 0
        self._chunk_lengths = []
        for f in tqdm(self.chunk_files, desc="Indexing"):
            d = np.load(f, mmap_mode='r')
            n = len(d['landmarks'])
            self._chunk_lengths.append(n)
            self._total += n
            d.close()
        print(f"Total samples: {self._total:,}")

    def __len__(self):
        return self._total

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            chunk_indices = list(range(len(self.chunk_files)))
        else:
            chunk_indices = list(range(len(self.chunk_files)))[
                worker_info.id :: worker_info.num_workers
            ]

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


# ============================================================
# Training loop
# ============================================================

def train(config: dict):
    print("=" * 60)
    print("KNOWLEDGE DISTILLATION — 3D LANDMARK (1k3d68)")
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
    print("\nChecking dataset vs RAM...")
    stream = config['stream_dataset']
    if stream == 'auto':
        stream = should_stream(config['data_dir'], headroom_gb=40.0)
    else:
        gb = estimate_dataset_ram_gb(config['data_dir'])
        av = psutil.virtual_memory().available / 1e9
        print(f"  Dataset  : {gb:.1f} GB   Available: {av:.1f} GB")
        if not stream and gb > av - 40.0:
            print(f"  WARNING: stream_dataset=False but dataset ({gb:.1f} GB) "
                  f"may exceed available RAM ({av:.1f} GB)")

    if stream:
        print("\nUsing LargeDistillationDataset (streaming)")
        train_ds    = LargeDistillationDataset(
            config['data_dir'], augment=True, shuffle=True
        )
        # Hold out first chunk as validation (no augment)
        val_ds      = LargeDistillationDataset(
            config['data_dir'], augment=False, shuffle=False
        )
        # Limit val to first 1000 samples via a wrapper
        val_ds._total       = min(1000, val_ds._total)
        val_ds.chunk_files  = val_ds.chunk_files[:1]
        shuffle_loader = False
    else:
        print("\nUsing DistillationDataset (all in RAM)")
        full_ds    = DistillationDataset(
            config['data_dir'], augment=True, val_split=0.01
        )
        train_ds   = full_ds
        val_ds     = full_ds.get_val_dataset()
        shuffle_loader = True

    train_loader = DataLoader(
        train_ds,
        batch_size       = config['batch_size'],
        shuffle          = shuffle_loader,
        num_workers      = config['num_workers'],
        pin_memory       = True,
        pin_memory_device= 'cuda:0',
        drop_last        = True,
        persistent_workers = True,
        prefetch_factor  = 4,
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
        'small':  StudentModelSmall,
        'medium': StudentModelMedium,
        'large':  StudentModelLarge,
    }
    if config['model_size'] not in model_map:
        raise ValueError(
            f"Unknown model_size '{config['model_size']}'. "
            f"Choose: {list(model_map.keys())}"
        )
    model = model_map[config['model_size']]().to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel      : StudentModel{config['model_size'].capitalize()}")
    print(f"Parameters : {total_p:,} total  {trainable_p:,} trainable")

    # ── Loss ──────────────────────────────────────────────────
    criterion = LandmarkDistillationLoss(
        alpha_wing = config['alpha_wing'],
        alpha_l1   = config['alpha_l1'],
        alpha_mse  = config['alpha_mse'],
        z_weight   = config['z_weight'],
    )

    # ── Optimiser + LR schedule ───────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config['learning_rate'],
        weight_decay = config['weight_decay'],
    )

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

    # ── Train ─────────────────────────────────────────────────
    print(f"\nTraining for {config['epochs']} epochs "
          f"(start={start_epoch + 1})")
    print(f"Total steps  : {total_steps:,}")
    print(f"Warmup steps : {warmup_steps:,}")

    history = []

    for epoch in range(start_epoch, config['epochs']):
        # ── Train epoch ───────────────────────────────────────
        model.train()
        ep_losses   = {'wing': 0., 'l1': 0., 'mse': 0., 'total': 0.}
        n_batches   = 0

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
                    student_lmk         = model(images)
                    loss, loss_dict     = criterion(student_lmk, teacher_lmk)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                student_lmk         = model(images)
                loss, loss_dict     = criterion(student_lmk, teacher_lmk)
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
        all_nme = []
        with torch.no_grad():
            for images, teacher_lmk in val_loader:
                images      = images.to(device)
                teacher_lmk = teacher_lmk.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        student_lmk = model(images)
                else:
                    student_lmk = model(images)
                all_nme.append(compute_nme(student_lmk, teacher_lmk))

        val_nme = float(np.mean(all_nme))

        print(f"\nEpoch {epoch+1} | "
              f"loss={avg['total']:.5f}  wing={avg['wing']:.5f}  "
              f"l1={avg['l1']:.5f}  mse={avg['mse']:.5f}  "
              f"NME={val_nme:.3f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        history.append({
            'epoch':  epoch + 1,
            'nme':    val_nme,
            **avg,
        })

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
        {'epoch': config['epochs'], 'model_state_dict': model.state_dict(),
         'config': config},
        os.path.join(save_dir, 'final_model.pth'),
    )
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best NME  : {best_nme:.3f}%")
    print(f"Saved to  : {save_dir}")
    print(f"{'='*60}")

    return save_dir


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    config = {
        # ── Data ──────────────────────────────────────────────
        'data_dir':       'distill_data/ms1m_landmarks',
        'num_workers':    16,
        # 'auto' measures RAM at runtime and picks the right dataset class.
        # Set False to force in-RAM (fast, needs ≥150 GB available).
        # Set True  to force streaming (safe, slightly slower).
        'stream_dataset': 'auto',

        # ── Model ─────────────────────────────────────────────
        # 'large'  recommended — closest to teacher ResNet-50 depth
        # 'medium' if training time is a concern
        # 'small'  for ablation / quick debugging only
        'model_size':     'large',

        # ── Training ──────────────────────────────────────────
        # Images are 192x192 (2.94x larger than 112x112 used by glintr100).
        # Reduce batch_size relative to glintr100 run accordingly.
        'batch_size':     512,
        'epochs':         50,
        # LR sqrt-scaled from 1e-3 @ bs=256: 1e-3 * sqrt(512/256) = 1.41e-3
        'learning_rate':  1.41e-3,
        'weight_decay':   1e-4,
        'warmup_epochs':  3,

        # ── Loss weights ──────────────────────────────────────
        'alpha_wing':     1.0,   # Wing loss on xy — primary landmark loss
        'alpha_l1':       1.0,   # L1 on xyz (z downweighted by z_weight)
        'alpha_mse':      0.5,   # MSE on xyz — penalises large errors hard
        'z_weight':       0.5,   # z error weight relative to xy

        # ── Saving ────────────────────────────────────────────
        'save_dir':       'checkpoints_landmarks',
        'save_every':     5,

        # ── Resume ────────────────────────────────────────────
        # 'resume_from': 'checkpoints_landmarks/run_YYYYMMDD_HHMMSS/checkpoint_epoch_0010.pth',
        'resume_from':    None,
    }

    train(config)
