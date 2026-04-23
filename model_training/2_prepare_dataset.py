# 2_prepare_dataset.py
"""
Step 2: Prepare training data.

STRATEGY: We use publicly available face datasets, pass every face image
through the teacher model, and store (image, teacher_embedding) pairs.

This is like carefully studying every detail of the painting from
multiple angles and lighting conditions.

DATASETS (choose one or combine):
  Option A: Download a subset of MS1M/Glint360K (recommended)
  Option B: Use any large face dataset (LFW, CelebA, VGGFace2, etc.)
  Option C: Generate synthetic faces + augmentations
"""

import os
import sys
import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm
from pathlib import Path
import pickle
import mxnet as mx
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# OPTION A: Load from MXNet RecordIO format (MS1M / Glint360K)
# ============================================================
# Most InsightFace datasets come in .rec/.idx format.
# Download from: https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_
#
# For example, download "ms1m-retinaface-t1" or "glint360k":
#   https://drive.google.com/file/d/1JgmzL9OLTqDAZE86pBgETtSQL4USKTFy
#
# After download, you'll have:
#   faces_ms1m_112x112/
#     ├── train.rec
#     ├── train.idx
#     └── property

class MXRecordDatasetReader:
    """
    Reads face images from MXNet RecordIO (.rec) files.
    These are already aligned and cropped to 112x112.
    """

    def __init__(self, rec_path: str, idx_path: str):
        self.record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
        # Read all keys
        self.keys = list(self.record.keys)
        print(f"Dataset contains {len(self.keys)} images")

    def __len__(self):
        return len(self.keys)

    def read_image(self, index: int) -> np.ndarray:
        """Read a single image, returns (112, 112, 3) BGR uint8"""
        header, raw = mx.recordio.unpack(self.record.read_idx(self.keys[index]))
        img = mx.image.imdecode(raw).asnumpy()  # RGB, (112, 112, 3)
        return img

    def read_batch(self, indices: list) -> np.ndarray:
        """Read multiple images"""
        images = []
        for idx in indices:
            try:
                img = self.read_image(idx)
                images.append(img)
            except Exception:
                continue
        return np.stack(images, axis=0) if images else np.array([])


# ============================================================
# OPTION B: Load from a folder of images
# ============================================================

class FolderDatasetReader:
    """
    Reads face images from a folder structure:
      dataset_root/
        person_001/
          img_001.jpg
          img_002.jpg
        person_002/
          ...

    Images should ideally be pre-aligned to 112x112.
    If not, we'll resize them.
    """

    def __init__(self, root_dir: str, target_size: int = 112):
        self.root_dir = root_dir
        self.target_size = target_size
        self.image_paths = []

        # Collect all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(
                list(Path(root_dir).rglob(ext))
            )

        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, index: int) -> np.ndarray:
        """Read and resize a single image to 112x112 RGB"""
        img = cv2.imread(str(self.image_paths[index]))
        if img is None:
            raise ValueError(f"Cannot read {self.image_paths[index]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.target_size, self.target_size):
            img = cv2.resize(img, (self.target_size, self.target_size))
        return img


# ============================================================
# Teacher Embedding Generator
# ============================================================

class TeacherEmbeddingGenerator:
    """
    Feeds images through the teacher ONNX model and collects embeddings.
    This is the "studying the painting" phase.
    """

    def __init__(self, onnx_path: str, batch_size: int = 64):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.batch_size = batch_size

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess images for the teacher model.
        AntelopeV2/InsightFace expects:
          - Shape: (N, 3, 112, 112)
          - Range: approximately [-1, 1] or [0, 255] with specific normalization
          - The InsightFace models typically use: (img - 127.5) / 127.5
          - Channel order: RGB
        """
        # images: (N, 112, 112, 3) uint8 RGB
        imgs = images.astype(np.float32)

        # Normalize: map [0, 255] -> [-1, 1]
        imgs = (imgs - 127.5) / 127.5

        # Transpose: (N, H, W, C) -> (N, C, H, W)
        imgs = imgs.transpose(0, 3, 1, 2)

        return imgs

    def get_embeddings(self, images: np.ndarray) -> np.ndarray:
        """
        Get teacher embeddings for a batch of images.
        images: (N, 112, 112, 3) uint8 RGB
        returns: (N, 512) float32 embeddings
        """
        preprocessed = self.preprocess(images)
        embeddings = self.session.run(None, {self.input_name: preprocessed})[0]
        return embeddings

    def process_dataset_rec(
        self,
        rec_path: str,
        idx_path: str,
        output_dir: str,
        max_samples: int = None,
        chunk_size: int = 100000,
    ):
        """
        Process an entire MXNet RecordIO dataset and save embeddings in chunks.

        We save in chunks to avoid running out of RAM. Each chunk file contains:
          - 'images': (chunk_size, 112, 112, 3) uint8
          - 'embeddings': (chunk_size, 512) float32
        """
        os.makedirs(output_dir, exist_ok=True)
        reader = MXRecordDatasetReader(rec_path, idx_path)

        total = min(len(reader), max_samples) if max_samples else len(reader)
        print(f"Processing {total} images...")

        chunk_images = []
        chunk_embeddings = []
        chunk_idx = 0
        processed = 0

        pbar = tqdm(total=total, desc="Generating teacher embeddings")

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            indices = list(range(start, end))

            # Read batch
            batch_imgs = reader.read_batch(indices)
            if len(batch_imgs) == 0:
                continue

            # Get teacher embeddings
            embeddings = self.get_embeddings(batch_imgs)

            chunk_images.append(batch_imgs)
            chunk_embeddings.append(embeddings)
            processed += len(batch_imgs)
            pbar.update(len(batch_imgs))

            # Save chunk
            if processed >= chunk_size or end >= total:
                all_imgs = np.concatenate(chunk_images, axis=0)
                all_embs = np.concatenate(chunk_embeddings, axis=0)

                chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
                np.savez_compressed(
                    chunk_file,
                    images=all_imgs,
                    embeddings=all_embs,
                )
                print(f"\nSaved {chunk_file}: {all_imgs.shape[0]} samples")

                chunk_images = []
                chunk_embeddings = []
                chunk_idx += 1
                processed = 0

        pbar.close()
        print(f"\nDone! Saved {chunk_idx} chunks to {output_dir}")

    def process_dataset_folder(
        self,
        folder_path: str,
        output_dir: str,
        max_samples: int = None,
        chunk_size: int = 100000,
    ):
        """Process a folder-based dataset."""
        os.makedirs(output_dir, exist_ok=True)
        reader = FolderDatasetReader(folder_path)

        total = min(len(reader), max_samples) if max_samples else len(reader)
        print(f"Processing {total} images...")

        chunk_images = []
        chunk_embeddings = []
        chunk_idx = 0
        processed = 0

        pbar = tqdm(total=total, desc="Generating teacher embeddings")

        batch_imgs_list = []
        for i in range(total):
            try:
                img = reader.read_image(i)
                batch_imgs_list.append(img)
            except Exception:
                continue

            if len(batch_imgs_list) == self.batch_size or i == total - 1:
                batch_imgs = np.stack(batch_imgs_list, axis=0)
                embeddings = self.get_embeddings(batch_imgs)

                chunk_images.append(batch_imgs)
                chunk_embeddings.append(embeddings)
                processed += len(batch_imgs)
                pbar.update(len(batch_imgs))
                batch_imgs_list = []

                if processed >= chunk_size or i == total - 1:
                    all_imgs = np.concatenate(chunk_images, axis=0)
                    all_embs = np.concatenate(chunk_embeddings, axis=0)

                    chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
                    np.savez_compressed(chunk_file, images=all_imgs, embeddings=all_embs)
                    print(f"\nSaved {chunk_file}: {all_imgs.shape[0]} samples")

                    chunk_images = []
                    chunk_embeddings = []
                    chunk_idx += 1
                    processed = 0

        pbar.close()


if __name__ == "__main__":
    teacher = TeacherEmbeddingGenerator(
        onnx_path="teacher_model/glintr100.onnx",
        batch_size=128,
    )

    # =============================================
    # OPTION A: MXNet RecordIO dataset (Recommended)
    # =============================================
    # Download MS1M-RetinaFace or Glint360K from InsightFace repo:
    #   https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_
    #
    # Uncomment below:

    teacher.process_dataset_rec(
        rec_path="datasets/ms1m_retinaface/train.rec",
        idx_path="datasets/ms1m_retinaface/train.idx",
        output_dir="distill_data/ms1m",
        max_samples=1000000,  # Use 1M samples for good quality
    )

    # =============================================
    # OPTION B: Folder of face images
    # =============================================
    # If you have face images in a folder structure:

    # teacher.process_dataset_folder(
    #     folder_path="datasets/my_faces/",
    #     output_dir="distill_data/my_faces",
    #     max_samples=500000,
    # )

    # =============================================
    # QUICK TEST: Small synthetic dataset (for testing the pipeline)
    # =============================================
    print("\n=== Quick test with synthetic data ===")
    os.makedirs("distill_data/test", exist_ok=True)

    # Generate random 112x112 "face" images (just for testing)
    n_test = 1024
    test_images = np.random.randint(0, 255, (n_test, 112, 112, 3), dtype=np.uint8)
    test_embeddings = teacher.get_embeddings(test_images)

    np.savez_compressed(
        "distill_data/test/chunk_0000.npz",
        images=test_images,
        embeddings=test_embeddings,
    )
    print(f"Saved test data: images={test_images.shape}, embeddings={test_embeddings.shape}")
    print("Pipeline test successful! Now use real face data for actual training.")
