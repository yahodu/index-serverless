# 3_train_student.py
"""
Step 3: Train YOUR OWN face recognition model via knowledge distillation.

This is the "drawing the painting yourself" step.

WHAT'S HAPPENING:
  - The teacher model maps face images → 512-d embeddings
  - We train a new model (student) to produce the SAME embeddings
  - The student's weights are randomly initialized and trained by US
  - After training, the student is entirely OUR creation

STUDENT ARCHITECTURE CHOICES:
  We use a smaller ResNet variant for efficiency, but you can use anything.
  The key is: YOUR architecture + YOUR random init + YOUR training = YOUR model.
"""

import os
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
from datetime import datetime


# ============================================================
# Student Model Architecture
# ============================================================
# You can choose ANY architecture. Here are several options:

class ConvBlock(nn.Module):
    """Basic convolution block: Conv -> BN -> PReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

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
    """
    Small student model (~2M params).
    Good for quick training and testing.
    """
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            # 112x112 -> 56x56
            ConvBlock(3, 64, 3, 2, 1),
            ResidualBlock(64),

            # 56x56 -> 28x28
            ConvBlock(64, 128, 3, 2, 1),
            ResidualBlock(128),

            # 28x28 -> 14x14
            ConvBlock(128, 256, 3, 2, 1),
            ResidualBlock(256),
            ResidualBlock(256),

            # 14x14 -> 7x7
            ConvBlock(256, 512, 3, 2, 1),
            ResidualBlock(512),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x


class StudentModelMedium(nn.Module):
    """
    Medium student model (~10M params).
    Better quality, reasonable training time.
    RECOMMENDED for production use.
    """
    def __init__(self, embedding_dim=512):
        super().__init__()

        def make_layer(in_ch, out_ch, num_blocks, stride=2):
            layers = [ConvBlock(in_ch, out_ch, 3, stride, 1)]
            for _ in range(num_blocks):
                layers.append(ResidualBlock(out_ch))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            ConvBlock(3, 64, 3, 1, 1),     # 112x112

            make_layer(64, 64, 3, 2),       # -> 56x56
            make_layer(64, 128, 4, 2),      # -> 28x28
            make_layer(128, 256, 6, 2),     # -> 14x14
            make_layer(256, 512, 3, 2),     # -> 7x7
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

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
    Closest to teacher quality.
    Uses IResNet-50 style architecture.
    """
    def __init__(self, embedding_dim=512):
        super().__init__()

        def make_layer(in_ch, out_ch, num_blocks, stride=2):
            layers = [ConvBlock(in_ch, out_ch, 3, stride, 1)]
            for _ in range(num_blocks):
                layers.append(ResidualBlock(out_ch))
            return nn.Sequential(*layers)

        self.conv1 = ConvBlock(3, 64, 3, 1, 1)  # 112x112

        self.layer1 = make_layer(64, 64, 3, 2)    # -> 56x56
        self.layer2 = make_layer(64, 128, 8, 2)   # -> 28x28
        self.layer3 = make_layer(128, 256, 16, 2)  # -> 14x14
        self.layer4 = make_layer(256, 512, 3, 2)   # -> 7x7

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

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
# Dataset
# ============================================================

class DistillationDataset(Dataset):
    """
    Loads pre-computed (image, teacher_embedding) pairs from .npz chunks.
    Applies data augmentation to images (the teacher embedding stays fixed).

    Data augmentation is CRITICAL — it's like studying the painting from
    different angles, lighting, etc. to truly understand it.
    """

    def __init__(self, data_dir: str, augment: bool = True):
        self.data_dir = data_dir
        self.augment = augment

        # Find all chunk files
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {data_dir}")

        # Load all chunks into memory (if they fit)
        # For very large datasets, use memory mapping or IterableDataset
        print(f"Loading {len(self.chunk_files)} chunks from {data_dir}...")
        all_images = []
        all_embeddings = []

        for f in tqdm(self.chunk_files, desc="Loading chunks"):
            data = np.load(f)
            all_images.append(data['images'])
            all_embeddings.append(data['embeddings'])

        self.images = np.concatenate(all_images, axis=0)      # (N, 112, 112, 3) uint8
        self.embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 512) float32

        print(f"Loaded {len(self.images)} samples")

        # Augmentation transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                ),
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ], p=0.2),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
        ])

        self.transform_no_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (112, 112, 3) uint8 RGB
        embedding = self.embeddings[idx]  # (512,) float32

        if self.augment:
            image = self.transform(image)
        else:
            image = self.transform_no_aug(image)

        embedding = torch.tensor(embedding, dtype=torch.float32)

        return image, embedding


class LargeDistillationDataset(IterableDataset):
    """
    Memory-efficient version for very large datasets.
    Streams chunks from disk instead of loading all into RAM.
    """

    def __init__(self, data_dir: str, augment: bool = True, shuffle: bool = True):
        self.data_dir = data_dir
        self.augment = augment
        self.shuffle = shuffle
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        self.transform_no_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        # Calculate total length
        self._total_len = 0
        for f in self.chunk_files:
            data = np.load(f)
            self._total_len += len(data['embeddings'])
            data.close()

    def __iter__(self):
        chunk_order = list(range(len(self.chunk_files)))
        if self.shuffle:
            np.random.shuffle(chunk_order)

        for chunk_idx in chunk_order:
            data = np.load(self.chunk_files[chunk_idx])
            images = data['images']
            embeddings = data['embeddings']

            indices = list(range(len(images)))
            if self.shuffle:
                np.random.shuffle(indices)

            for i in indices:
                img = images[i]
                emb = embeddings[i]

                if self.augment:
                    img = self.transform(img)
                else:
                    img = self.transform_no_aug(img)

                yield img, torch.tensor(emb, dtype=torch.float32)

            data.close()


# ============================================================
# Loss Functions
# ============================================================

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation of face embeddings.

    We want the student to produce embeddings that:
      1. Are close to the teacher's embeddings (MSE)
      2. Have the same directional similarity (Cosine)
      3. Preserve relative distances between samples (Optional: Relational)
    """

    def __init__(self, alpha_mse=1.0, alpha_cosine=1.0):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_cosine = alpha_cosine
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineEmbeddingLoss()

    def forward(self, student_emb, teacher_emb):
        # Normalize embeddings (face recognition models typically use normalized embeddings)
        student_norm = F.normalize(student_emb, p=2, dim=1)
        teacher_norm = F.normalize(teacher_emb, p=2, dim=1)

        # MSE Loss: match the raw embedding values
        loss_mse = self.mse(student_norm, teacher_norm)

        # Cosine Loss: match the direction of embeddings
        target = torch.ones(student_emb.size(0), device=student_emb.device)
        loss_cosine = self.cosine(student_norm, teacher_norm, target)

        total_loss = self.alpha_mse * loss_mse + self.alpha_cosine * loss_cosine

        return total_loss, {
            'mse': loss_mse.item(),
            'cosine': loss_cosine.item(),
            'total': total_loss.item(),
        }


# ============================================================
# Training Loop
# ============================================================

def train(config: dict):
    """Main training function."""

    print("=" * 60)
    print("KNOWLEDGE DISTILLATION TRAINING")
    print("=" * 60)
    print(json.dumps(config, indent=2))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config['save_dir'], f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Dataset
    print("\nLoading dataset...")
    train_dataset = DistillationDataset(config['data_dir'], augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    # Model
    print(f"\nCreating student model: {config['model_size']}")
    if config['model_size'] == 'small':
        model = StudentModelSmall(embedding_dim=config['embedding_dim'])
    elif config['model_size'] == 'medium':
        model = StudentModelMedium(embedding_dim=config['embedding_dim'])
    elif config['model_size'] == 'large':
        model = StudentModelLarge(embedding_dim=config['embedding_dim'])
    else:
        raise ValueError(f"Unknown model size: {config['model_size']}")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss
    criterion = DistillationLoss(
        alpha_mse=config['alpha_mse'],
        alpha_cosine=config['alpha_cosine'],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    # Learning rate scheduler
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = len(train_loader) * config.get('warmup_epochs', 2)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    best_loss = float('inf')
    history = []

    for epoch in range(config['epochs']):
        model.train()
        epoch_losses = {'mse': 0, 'cosine': 0, 'total': 0}
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config['epochs']}",
            leave=True,
        )

        for batch_idx, (images, teacher_embeddings) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            teacher_embeddings = teacher_embeddings.to(device, non_blocking=True)

            # Forward pass with mixed precision
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    student_embeddings = model(images)
                    loss, loss_dict = criterion(student_embeddings, teacher_embeddings)

                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                student_embeddings = model(images)
                loss, loss_dict = criterion(student_embeddings, teacher_embeddings)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # Accumulate losses
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'mse': f"{loss_dict['mse']:.4f}",
                'cos': f"{loss_dict['cosine']:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}",
            })

        # Epoch summary
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Loss: {avg_losses['total']:.6f}")
        print(f"  MSE: {avg_losses['mse']:.6f}")
        print(f"  Cosine: {avg_losses['cosine']:.6f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.8f}")

        history.append(avg_losses)

        # Save best model
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (loss: {best_loss:.6f})")

        # Save periodic checkpoint
        if (epoch + 1) % config.get('save_every', 10) == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_losses['total'],
                'config': config,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Save final model
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'config': config,
    }, os.path.join(save_dir, 'final_model.pth'))

    # Save training history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*60}")

    return save_dir


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    config = {
        # Data
        'data_dir': 'distill_data/ms1m',    # Directory with .npz chunks
        'num_workers': 4,                    # DataLoader workers

        # Model
        'model_size': 'medium',              # 'small', 'medium', or 'large'
        'embedding_dim': 512,                # Must match teacher (512)

        # Training
        'batch_size': 256,                   # Adjust based on GPU VRAM
        'epochs': 50,                        # 30-100 depending on dataset size
        'learning_rate': 1e-3,               # Peak learning rate
        'weight_decay': 1e-4,                # L2 regularization
        'warmup_epochs': 2,                  # LR warmup

        # Loss weights
        'alpha_mse': 1.0,                    # MSE loss weight
        'alpha_cosine': 1.0,                 # Cosine loss weight

        # Saving
        'save_dir': 'checkpoints',
        'save_every': 10,                    # Save checkpoint every N epochs
    }

    # Adjust batch size based on available VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        if vram_gb < 8:
            config['batch_size'] = 64
        elif vram_gb < 16:
            config['batch_size'] = 128
        elif vram_gb < 24:
            config['batch_size'] = 256
        else:
            config['batch_size'] = 512

        print(f"Auto-set batch_size={config['batch_size']} for {vram_gb:.1f}GB VRAM")

    save_dir = train(config)
