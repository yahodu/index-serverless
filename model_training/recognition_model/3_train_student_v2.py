# 3_train_student.py
"""
Step 3: Train YOUR OWN face recognition model via knowledge distillation.

WHAT'S HAPPENING:
  - The teacher model maps face images -> 512-d embeddings
  - We train a new model (student) to produce the SAME embeddings
  - The student's weights are randomly initialized and trained by US
  - After training, the student is entirely OUR creation

DATASET MODES (auto-selected based on RAM fit check):
  stream_dataset = False  -> DistillationDataset      (loads all into RAM, true global shuffle)
  stream_dataset = True   -> LargeDistillationDataset (streams from disk, per-chunk shuffle)

  At runtime the code measures how much RAM loading would require and
  warns you if it exceeds the threshold so you never silently OOM.
"""

import os
import gc
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from tqdm import tqdm
import json
import psutil
from datetime import datetime


# ============================================================
# Student Model Architecture
# ============================================================

class ConvBlock(nn.Module):
    """Basic convolution block: Conv -> BN -> PReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.PReLU(out_ch)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two conv layers"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class StudentModelSmall(nn.Module):
    """Small student model (~2M params). Good for quick testing."""
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64, 3, 2, 1),       # 112 -> 56
            ResidualBlock(64),
            ConvBlock(64, 128, 3, 2, 1),     # 56  -> 28
            ResidualBlock(128),
            ConvBlock(128, 256, 3, 2, 1),    # 28  -> 14
            ResidualBlock(256),
            ResidualBlock(256),
            ConvBlock(256, 512, 3, 2, 1),    # 14  ->  7
            ResidualBlock(512),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(512, embedding_dim)
        self.bn   = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x


class StudentModelMedium(nn.Module):
    """Medium student model (~10M params)."""
    def __init__(self, embedding_dim=512):
        super().__init__()

        def make_layer(in_ch, out_ch, num_blocks, stride=2):
            layers = [ConvBlock(in_ch, out_ch, 3, stride, 1)]
            for _ in range(num_blocks):
                layers.append(ResidualBlock(out_ch))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            ConvBlock(3, 64, 3, 1, 1),
            make_layer(64,  64,  3, 2),
            make_layer(64,  128, 4, 2),
            make_layer(128, 256, 6, 2),
            make_layer(256, 512, 3, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(512, embedding_dim)
        self.bn   = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x


class StudentModelLarge(nn.Module):
    """
    Large student model (~25M params).
    Closest to teacher quality. IResNet-50 style.
    RECOMMENDED for H200 — VRAM is not a constraint.
    """
    def __init__(self, embedding_dim=512):
        super().__init__()

        def make_layer(in_ch, out_ch, num_blocks, stride=2):
            layers = [ConvBlock(in_ch, out_ch, 3, stride, 1)]
            for _ in range(num_blocks):
                layers.append(ResidualBlock(out_ch))
            return nn.Sequential(*layers)

        self.conv1  = ConvBlock(3, 64, 3, 1, 1)
        self.layer1 = make_layer(64,  64,  3,  2)
        self.layer2 = make_layer(64,  128, 8,  2)
        self.layer3 = make_layer(128, 256, 16, 2)
        self.layer4 = make_layer(256, 512, 3,  2)

        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc      = nn.Linear(512, embedding_dim)
        self.bn      = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        return x


# ============================================================
# RAM fit check
# ============================================================

def estimate_dataset_ram_gb(data_dir: str) -> float:
    """
    Estimates how much RAM (in GB) loading all chunks would require.
    Reads only array headers — does not load the actual data.
    """
    chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
    if not chunk_files:
        raise ValueError(f"No chunk files found in {data_dir}")

    total_bytes = 0
    for f in chunk_files:
        data = np.load(f, mmap_mode='r')
        total_bytes += data['images'].nbytes + data['embeddings'].nbytes
        data.close()

    return total_bytes / 1e9


def should_stream(data_dir: str, ram_headroom_gb: float = 40.0) -> bool:
    """
    Returns True if the dataset is too large to load into RAM safely.

    ram_headroom_gb: how much RAM to keep free for OS, model, gradients, etc.
                     40 GB is conservative for a 180 GB system.
    """
    available_gb  = psutil.virtual_memory().available / 1e9
    dataset_gb    = estimate_dataset_ram_gb(data_dir)
    fits_in_ram   = dataset_gb < (available_gb - ram_headroom_gb)

    print(f"Dataset RAM estimate : {dataset_gb:.1f} GB")
    print(f"Available RAM        : {available_gb:.1f} GB")
    print(f"RAM headroom         : {ram_headroom_gb:.1f} GB")
    print(f"Fits in RAM          : {fits_in_ram}  "
          f"-> {'DistillationDataset' if fits_in_ram else 'LargeDistillationDataset'}")

    return not fits_in_ram


# ============================================================
# Dataset — in-memory (1M scale, ~39.7 GB)
# ============================================================

class DistillationDataset(Dataset):
    """
    Loads ALL chunks into RAM upfront.

    Use when: dataset fits in RAM (available RAM - 40 GB headroom).

    Advantages over streaming:
      - True random global shuffle across all samples every epoch
      - Zero disk I/O during training (pure GPU-bound after load)
      - No worker chunk-splitting complexity

    1M samples:
      images     : 1M x 112 x 112 x 3 = 37.6 GB
      embeddings : 1M x 512 x 4       =  2.1 GB
      total                            = 39.7 GB  << 180 GB RAM
    """

    def __init__(self, data_dir: str, augment: bool = True):
        self.data_dir = data_dir
        self.augment  = augment

        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {data_dir}")

        print(f"Loading {len(self.chunk_files)} chunks into RAM...")
        all_images     = []
        all_embeddings = []

        for f in tqdm(self.chunk_files, desc="Loading chunks"):
            data = np.load(f)
            all_images.append(data['images'].copy())
            all_embeddings.append(data['embeddings'].copy())
            data.close()

        self.images     = np.concatenate(all_images,     axis=0)
        self.embeddings = np.concatenate(all_embeddings, axis=0)

        del all_images, all_embeddings
        gc.collect()

        img_gb = self.images.nbytes     / 1e9
        emb_gb = self.embeddings.nbytes / 1e9
        print(f"Loaded {len(self.images):,} samples  "
              f"(images: {img_gb:.2f} GB  "
              f"embeddings: {emb_gb:.2f} GB  "
              f"total: {img_gb + emb_gb:.2f} GB)")

        self._build_transforms()

    def _build_transforms(self):
        # NOTE: RandomGrayscale intentionally omitted — teacher embeddings
        # were computed on RGB; grayscale inputs cause a distribution mismatch.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.transform_no_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image     = self.images[idx]
        embedding = self.embeddings[idx]
        image     = self.transform(image) if self.augment else self.transform_no_aug(image)
        return image, torch.tensor(embedding, dtype=torch.float32)


# ============================================================
# Dataset — streaming (5M scale, ~198 GB)
# ============================================================

class LargeDistillationDataset(IterableDataset):
    """
    Streams chunks from disk one at a time.

    Use when: dataset does not fit in available RAM.

    5M samples:
      images     : 5M x 112 x 112 x 3 = 188.2 GB
      embeddings : 5M x 512 x 4       =  10.2 GB
      total                            = 198.4 GB  > 180 GB RAM

    Worker-aware chunk splitting ensures each DataLoader worker reads a
    disjoint subset of chunks — no sample is seen twice per epoch.

    Each chunk (~378 MB at 489 chunks / 185 GB) is read sequentially and
    freed immediately after yielding all its samples.
    """

    def __init__(self, data_dir: str, augment: bool = True, shuffle: bool = True):
        self.data_dir    = data_dir
        self.augment     = augment
        self.shuffle     = shuffle

        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {data_dir}")

        # Count total samples by reading headers only (no data loaded)
        print(f"Indexing {len(self.chunk_files)} chunks...")
        self._total_len = 0
        for f in tqdm(self.chunk_files, desc="Indexing"):
            data = np.load(f, mmap_mode='r')
            self._total_len += len(data['embeddings'])
            data.close()
        print(f"Total samples: {self._total_len:,}")

        self._build_transforms()

    def _build_transforms(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        self.transform_no_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return self._total_len

    def __iter__(self):
        # Split chunks evenly across DataLoader workers.
        # Worker 0 gets chunks [0, num_workers, 2*num_workers, ...]
        # Worker 1 gets chunks [1, num_workers+1, ...]
        # etc. — ensures no duplicates within a batch.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            chunk_indices = list(range(len(self.chunk_files)))
        else:
            chunk_indices = list(range(len(self.chunk_files)))[
                worker_info.id :: worker_info.num_workers
            ]

        if self.shuffle:
            np.random.shuffle(chunk_indices)

        for chunk_idx in chunk_indices:
            data       = np.load(self.chunk_files[chunk_idx])
            images     = data['images'].copy()
            embeddings = data['embeddings'].copy()
            data.close()   # release file handle immediately

            indices = list(range(len(images)))
            if self.shuffle:
                np.random.shuffle(indices)

            for i in indices:
                img = images[i]
                emb = embeddings[i]
                img = self.transform(img) if self.augment else self.transform_no_aug(img)
                yield img, torch.tensor(emb, dtype=torch.float32)

            del images, embeddings
            gc.collect()


# ============================================================
# Loss Function
# ============================================================

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation of face embeddings.

    loss = alpha_mse    * MSE(student_norm, teacher_norm)
         + alpha_cosine * mean(1 - cosine_similarity(student_norm, teacher_norm))

    Both terms operate on L2-normalised embeddings.
    MSE on unit vectors:    ||u-v||^2 = 2 - 2*cos(u,v)  -> encodes direction + magnitude
    Cosine similarity loss: 1 - cos(u,v)                -> encodes direction only
    Together they reinforce both magnitude alignment and angular alignment.
    """

    def __init__(self, alpha_mse: float = 1.0, alpha_cosine: float = 1.0):
        super().__init__()
        self.alpha_mse    = alpha_mse
        self.alpha_cosine = alpha_cosine
        self.mse          = nn.MSELoss()

    def forward(self, student_emb: torch.Tensor, teacher_emb: torch.Tensor):
        student_norm = F.normalize(student_emb, p=2, dim=1)
        teacher_norm = F.normalize(teacher_emb, p=2, dim=1)

        loss_mse    = self.mse(student_norm, teacher_norm)
        loss_cosine = (1.0 - F.cosine_similarity(student_norm, teacher_norm, dim=1)).mean()
        total_loss  = self.alpha_mse * loss_mse + self.alpha_cosine * loss_cosine

        return total_loss, {
            'mse':    loss_mse.item(),
            'cosine': loss_cosine.item(),
            'total':  total_loss.item(),
        }


# ============================================================
# Training Loop
# ============================================================

def train(config: dict):

    print("=" * 60)
    print("KNOWLEDGE DISTILLATION TRAINING")
    print("=" * 60)
    print(json.dumps(config, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU : {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Output directory ──────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = os.path.join(config['save_dir'], f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # ── Dataset — auto-select mode ────────────────────────────
    print("\nChecking dataset size vs available RAM...")

    # Override stream_dataset with auto-detection if set to 'auto'
    stream = config['stream_dataset']
    if stream == 'auto':
        stream = should_stream(config['data_dir'], ram_headroom_gb=40.0)
        print(f"Auto-selected stream_dataset={stream}")
    else:
        # Still print the numbers for visibility even if user forced a mode
        dataset_gb = estimate_dataset_ram_gb(config['data_dir'])
        available_gb = psutil.virtual_memory().available / 1e9
        print(f"Dataset RAM estimate : {dataset_gb:.1f} GB")
        print(f"Available RAM        : {available_gb:.1f} GB")
        if not stream and dataset_gb > available_gb - 40.0:
            print(
                f"\nWARNING: stream_dataset=False but dataset ({dataset_gb:.1f} GB) "
                f"may exceed available RAM ({available_gb:.1f} GB). "
                f"Consider setting stream_dataset='auto' or True."
            )

    if stream:
        print("\nUsing LargeDistillationDataset (streaming from disk)")
        train_dataset  = LargeDistillationDataset(
            config['data_dir'], augment=True, shuffle=True
        )
        shuffle_loader = False   # shuffling done inside the dataset
    else:
        print("\nUsing DistillationDataset (loading all into RAM)")
        train_dataset  = DistillationDataset(config['data_dir'], augment=True)
        shuffle_loader = True    # DataLoader handles global shuffle

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle_loader,
        num_workers=config['num_workers'],
        pin_memory=True,
        pin_memory_device='cuda:0',
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # ── Model ─────────────────────────────────────────────────
    print(f"\nCreating student model: {config['model_size']}")
    model_map = {
        'small':  StudentModelSmall,
        'medium': StudentModelMedium,
        'large':  StudentModelLarge,
    }
    if config['model_size'] not in model_map:
        raise ValueError(
            f"Unknown model_size '{config['model_size']}'. "
            f"Choose from: {list(model_map.keys())}"
        )
    model = model_map[config['model_size']](embedding_dim=config['embedding_dim'])
    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ── Loss ──────────────────────────────────────────────────
    criterion = DistillationLoss(
        alpha_mse=config['alpha_mse'],
        alpha_cosine=config['alpha_cosine'],
    )

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    # ── LR Scheduler: linear warmup + cosine decay ────────────
    total_steps  = len(train_loader) * config['epochs']
    warmup_steps = len(train_loader) * config.get('warmup_epochs', 2)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Mixed precision ───────────────────────────────────────
    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    # ── Resume from checkpoint ────────────────────────────────
    start_epoch = 0
    best_loss   = float('inf')

    if config.get('resume_from'):
        ckpt = torch.load(config['resume_from'], map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_loss   = ckpt.get('loss', float('inf'))
        print(f"Resumed from {config['resume_from']} at epoch {start_epoch}")

    # ── Training loop ─────────────────────────────────────────
    print(f"\nStarting training for {config['epochs']} epochs "
          f"(starting at epoch {start_epoch + 1})...")
    print(f"Total steps  : {total_steps:,}")
    print(f"Warmup steps : {warmup_steps:,}")

    history = []

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        epoch_losses = {'mse': 0.0, 'cosine': 0.0, 'total': 0.0}
        num_batches  = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config['epochs']}",
            leave=True,
        )

        for images, teacher_embeddings in pbar:
            images             = images.to(device, non_blocking=True)
            teacher_embeddings = teacher_embeddings.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    student_embeddings = model(images)
                    loss, loss_dict    = criterion(student_embeddings, teacher_embeddings)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                student_embeddings = model(images)
                loss, loss_dict    = criterion(student_embeddings, teacher_embeddings)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            for k, v in loss_dict.items():
                epoch_losses[k] += v
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'mse':  f"{loss_dict['mse']:.4f}",
                'cos':  f"{loss_dict['cosine']:.4f}",
                'lr':   f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # ── Epoch summary ─────────────────────────────────────
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        print(f"\nEpoch {epoch+1} summary:")
        print(f"  Avg loss : {avg_losses['total']:.6f}")
        print(f"  MSE      : {avg_losses['mse']:.6f}")
        print(f"  Cosine   : {avg_losses['cosine']:.6f}")
        print(f"  LR       : {scheduler.get_last_lr()[0]:.2e}")
        history.append({'epoch': epoch + 1, **avg_losses})

        # ── Checkpointing ─────────────────────────────────────
        ckpt_payload = {
            'epoch':                epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':                 avg_losses['total'],
            'config':               config,
        }

        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            torch.save(ckpt_payload, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ New best model saved (loss: {best_loss:.6f})")

        if (epoch + 1) % config.get('save_every', 5) == 0:
            torch.save(
                ckpt_payload,
                os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:04d}.pth'),
            )
            print(f"  ✓ Periodic checkpoint saved")

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
    print(f"Best loss : {best_loss:.6f}")
    print(f"Saved to  : {save_dir}")
    print(f"{'='*60}")

    return save_dir


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    config = {
        # ── Data ──────────────────────────────────────────────
        'data_dir':       'distill_data/ms1m',
        'num_workers':    16,
        #   'auto' -> measures dataset size vs available RAM at runtime
        #             and picks DistillationDataset or LargeDistillationDataset
        #   False  -> always load into RAM  (fast, needs RAM >= dataset size + 40 GB)
        #   True   -> always stream         (slower, works for any dataset size)
        'stream_dataset': 'auto',

        # ── Model ─────────────────────────────────────────────
        'model_size':    'large',    # 'small' | 'medium' | 'large'
        'embedding_dim':  512,       # must match teacher output

        # ── Training ──────────────────────────────────────────
        'batch_size':     2048,
        'epochs':         50,
        'learning_rate':  2.8e-3,    # sqrt-scaled from 1e-3 @ bs=256: 1e-3 * sqrt(2048/256)
        'weight_decay':   1e-4,
        'warmup_epochs':  5,

        # ── Loss ──────────────────────────────────────────────
        'alpha_mse':      1.0,
        'alpha_cosine':   1.0,

        # ── Saving ────────────────────────────────────────────
        'save_dir':       'checkpoints',
        'save_every':     5,

        # ── Resume (optional) ─────────────────────────────────
        # 'resume_from': 'checkpoints/run_20260423_120000/checkpoint_epoch_0010.pth',
        'resume_from':    None,
    }

    train(config)
