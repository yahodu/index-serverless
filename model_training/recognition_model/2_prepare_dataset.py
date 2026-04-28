# 2_prepare_dataset.py
"""
Step 2: Prepare training data.

STRATEGY: We use publicly available face datasets, pass every face image
through the teacher model, and store (image, teacher_embedding) pairs.

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
import struct


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
    Pure-Python MXNet RecordIO reader — no MXNet dependency.
    Reads the same .rec/.idx files InsightFace datasets use.

    Binary layout per record:
      [0:4]    magic       uint32 LE  must == 0xced7230a
      [4:8]    encoded_len uint32 LE  upper 3 bits = cflag, lower 29 bits = data_len/4
      [8:16]   index       uint64 LE  (ignored)
      [16:24]  id          uint64 LE  (ignored)
      [24:]    payload

    InsightFace MS1M/Glint360K payload layout (confirmed by probe):
      [0:16]   label header  (16 bytes: two float64s or mixed — does NOT contain image)
      [16:]    JPEG bytes     (starts with ff d8 ff)

    Key=0 is always a dataset header record (no image) — skipped automatically.
    """

    _MAGIC = 0xced7230a
    _JPEG_SKIP = 16   # confirmed by probe_rec.py: SOI marker at payload byte 16

    def __init__(self, rec_path: str, idx_path: str):
        self.rec_path = rec_path

        # Index file: "key\toffset\n" (tab-separated)
        # Offsets are sometimes stored as floats (e.g. "1234.0"), so cast via float() first
        self.offsets = {}
        with open(idx_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    self.offsets[int(parts[0])] = int(float(parts[1]))

        all_keys = sorted(self.offsets.keys())

        # Key=0 is a metadata/header record (no JPEG) — always skip it
        self.keys = [k for k in all_keys if k != 0]
        print(f"Dataset contains {len(self.keys)} images "
              f"(skipped 1 header record, {len(all_keys)} total keys)")

        # Keep file handle open for fast seeking
        self._rec_file = open(rec_path, 'rb')

    def __len__(self):
        return len(self.keys)

    def __del__(self):
        if hasattr(self, '_rec_file') and self._rec_file:
            self._rec_file.close()

    def _read_record_at(self, offset: int) -> bytes:
        """
        Read one RecordIO record at a given byte offset.
        Returns raw JPEG bytes (everything after the 16-byte label header).
        """
        self._rec_file.seek(offset)

        # 4-byte magic
        raw = self._rec_file.read(4)
        if len(raw) < 4:
            raise ValueError(f"EOF at offset {offset}")
        magic = struct.unpack_from('<I', raw)[0]
        if magic != self._MAGIC:
            raise ValueError(
                f"Bad magic 0x{magic:08x} at offset {offset} "
                f"(expected 0x{self._MAGIC:08x})"
            )

        # 4-byte encoded length: lower 29 bits * 4 = total data length
        raw = self._rec_file.read(4)
        encoded  = struct.unpack_from('<I', raw)[0]
        data_len = (encoded & 0x1FFFFFFF) * 4

        # 16-byte header (index + id) — skip
        self._rec_file.read(16)

        # payload = data_len - 16 bytes
        payload_len = data_len - 16
        if payload_len <= self._JPEG_SKIP:
            raise ValueError(
                f"Payload too small ({payload_len} bytes) at offset {offset} — "
                f"likely a header/metadata record"
            )

        payload = self._rec_file.read(payload_len)

        # Skip the 16-byte label header to reach the JPEG SOI marker
        jpeg_bytes = payload[self._JPEG_SKIP:]
        return jpeg_bytes

    def read_image(self, index: int) -> np.ndarray:
        """
        Read image at position `index`.
        Returns (112, 112, 3) RGB uint8.
        """
        key    = self.keys[index]
        offset = self.offsets[key]
        raw    = self._read_record_at(offset)

        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError(
                f"imdecode failed at key={key}, offset={offset}, jpeg_len={len(raw)}"
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # -> RGB
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        return img

    def read_batch(self, indices: list) -> np.ndarray:
        """Read multiple images, skipping any that fail."""
        images = []
        for idx in indices:
            try:
                images.append(self.read_image(idx))
            except Exception as e:
                print(f"  [warn] skipping index {idx}: {e}")
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
    If not, they will be resized.
    """

    def __init__(self, root_dir: str, target_size: int = 112):
        self.root_dir    = root_dir
        self.target_size = target_size
        self.image_paths = []

        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_paths.extend(list(Path(root_dir).rglob(ext)))

        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, index: int) -> np.ndarray:
        """Read and resize a single image to target_size RGB."""
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

    Recommended batch sizes for glintr100 by GPU VRAM:
      4  GB  ->  8
      6  GB  -> 16
      8  GB  -> 24
      16 GB  -> 48
      24 GB+ -> 64-128
    """

    def __init__(self, onnx_path: str, batch_size: int = 16, gpu_mem_limit_gb: float = 4.0):
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern  = True
        sess_options.enable_cpu_mem_arena = True

        cuda_options = {
            "device_id": 0,
            # Don't grab all available VRAM upfront
            "arena_extend_strategy": "kSameAsRequested",
            # Hard cap on GPU memory usage
            "gpu_mem_limit": int(gpu_mem_limit_gb * 1024 * 1024 * 1024),
            "cudnn_conv_algo_search": "HEURISTIC",
        }

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=[
                ("CUDAExecutionProvider", cuda_options),
                "CPUExecutionProvider",
            ],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.batch_size = batch_size

        print(f"Teacher model loaded: {onnx_path}")
        print(f"  Batch size : {batch_size}")
        print(f"  GPU mem cap: {gpu_mem_limit_gb} GB")
        print(f"  Providers  : {self.session.get_providers()}")

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess images for InsightFace / AntelopeV2 models.
          Input : (N, 112, 112, 3) uint8 RGB
          Output: (N, 3, 112, 112) float32, range [-1, 1]
        Normalization: (pixel - 127.5) / 127.5
        """
        imgs = images.astype(np.float32)
        imgs = (imgs - 127.5) / 127.5      # [0, 255] -> [-1, 1]
        imgs = imgs.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        return imgs

    def get_embeddings(self, images: np.ndarray) -> np.ndarray:
        """
        Get teacher embeddings for a batch of images.
          Input : (N, 112, 112, 3) uint8 RGB
          Output: (N, 512) float32 embeddings
        """
        preprocessed = self.preprocess(images)
        embeddings   = self.session.run(None, {self.input_name: preprocessed})[0]
        return embeddings

    def process_dataset_rec(
        self,
        rec_path: str,
        idx_path: str,
        output_dir: str,
        max_samples: int = None,
        chunk_size:  int = 100_000,
    ):
        """
        Process an entire MXNet RecordIO dataset and save embeddings in chunks.

        Each saved .npz chunk contains:
          'images'     : (N, 112, 112, 3) uint8
          'embeddings' : (N, 512) float32
        """
        os.makedirs(output_dir, exist_ok=True)
        reader = MXRecordDatasetReader(rec_path, idx_path)

        total = min(len(reader), max_samples) if max_samples else len(reader)
        print(f"Processing {total} images...")

        chunk_images     = []
        chunk_embeddings = []
        chunk_idx        = 0
        processed        = 0

        pbar = tqdm(total=total, desc="Generating teacher embeddings")

        for start in range(0, total, self.batch_size):
            end     = min(start + self.batch_size, total)
            indices = list(range(start, end))

            batch_imgs = reader.read_batch(indices)
            if len(batch_imgs) == 0:
                continue

            embeddings = self.get_embeddings(batch_imgs)

            chunk_images.append(batch_imgs)
            chunk_embeddings.append(embeddings)
            processed += len(batch_imgs)
            pbar.update(len(batch_imgs))

            # Flush chunk to disk when it reaches chunk_size or we're at the end
            if processed >= chunk_size or end >= total:
                all_imgs = np.concatenate(chunk_images, axis=0)
                all_embs = np.concatenate(chunk_embeddings, axis=0)

                chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
                np.savez_compressed(chunk_file, images=all_imgs, embeddings=all_embs)
                print(f"\nSaved {chunk_file}: {all_imgs.shape[0]} samples")

                chunk_images     = []
                chunk_embeddings = []
                chunk_idx        += 1
                processed        = 0

        pbar.close()
        print(f"\nDone! Saved {chunk_idx} chunks to {output_dir}")

    def process_dataset_folder(
        self,
        folder_path: str,
        output_dir:  str,
        max_samples: int = None,
        chunk_size:  int = 100_000,
    ):
        """Process a folder-based dataset and save embeddings in chunks."""
        os.makedirs(output_dir, exist_ok=True)
        reader = FolderDatasetReader(folder_path)

        total = min(len(reader), max_samples) if max_samples else len(reader)
        print(f"Processing {total} images...")

        chunk_images     = []
        chunk_embeddings = []
        chunk_idx        = 0
        processed        = 0

        pbar           = tqdm(total=total, desc="Generating teacher embeddings")
        batch_imgs_list = []

        for i in range(total):
            try:
                img = reader.read_image(i)
                batch_imgs_list.append(img)
            except Exception as e:
                print(f"  [warn] skipping image {i}: {e}")
                continue

            if len(batch_imgs_list) == self.batch_size or i == total - 1:
                if not batch_imgs_list:
                    continue

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

                    chunk_images     = []
                    chunk_embeddings = []
                    chunk_idx        += 1
                    processed        = 0

        pbar.close()


# ============================================================
# Sanity checks — always run before the full pipeline
# ============================================================

def sanity_check_reader(rec_path: str, idx_path: str, n_samples: int = 5) -> bool:
    """Verify the RecordIO reader can decode real images."""
    print("\n=== Sanity check: RecordIO reader ===")
    reader = MXRecordDatasetReader(rec_path, idx_path)
    ok = 0
    for i in range(min(n_samples, len(reader))):
        try:
            img = reader.read_image(i)
            assert img.shape == (112, 112, 3), f"Unexpected shape: {img.shape}"
            assert img.dtype == np.uint8
            print(f"  [{i}] shape={img.shape} dtype={img.dtype} "
                  f"min={img.min()} max={img.max()}  OK")
            ok += 1
        except Exception as e:
            print(f"  [{i}] FAILED: {e}")
    print(f"Reader check: {ok}/{n_samples} passed")
    return ok == n_samples


def sanity_check_model(teacher: TeacherEmbeddingGenerator, n_images: int = 2) -> bool:
    """Verify the ONNX model runs and produces expected output shape."""
    print("\n=== Sanity check: ONNX model ===")
    dummy = np.random.randint(0, 255, (n_images, 112, 112, 3), dtype=np.uint8)
    try:
        embs = teacher.get_embeddings(dummy)
        assert embs.shape == (n_images, 512), f"Unexpected embedding shape: {embs.shape}"
        print(f"  Output shape : {embs.shape}  OK")
        print(f"  Output dtype : {embs.dtype}")
        print(f"  Value range  : [{embs.min():.4f}, {embs.max():.4f}]")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    REC_PATH = "datasets/ms1m_retinaface/train.rec"
    IDX_PATH = "datasets/ms1m_retinaface/train.idx"

    # Adjust both values to your GPU — see class docstring for reference table
    teacher = TeacherEmbeddingGenerator(
        onnx_path="teacher_model/glintr100.onnx",
        batch_size=24,
        gpu_mem_limit_gb=12.0,
    )

    # --- Step 1: Sanity-check the reader ---
    reader_ok = sanity_check_reader(REC_PATH, IDX_PATH)
    if not reader_ok:
        print("\nERROR: RecordIO reader failed — check _JPEG_SKIP value.")
        sys.exit(1)

    # --- Step 2: Sanity-check the ONNX model ---
    model_ok = sanity_check_model(teacher)
    if not model_ok:
        print("\nERROR: ONNX model failed — check batch_size / gpu_mem_limit_gb.")
        sys.exit(1)

    # =============================================
    # OPTION A: MXNet RecordIO dataset (Recommended)
    # =============================================
    teacher.process_dataset_rec(
        rec_path=REC_PATH,
        idx_path=IDX_PATH,
        output_dir="distill_data/ms1m",
        max_samples=1_000_000,  # 1M samples for good distillation quality
    )

    # =============================================
    # OPTION B: Folder of face images
    # =============================================
    # teacher.process_dataset_folder(
    #     folder_path="datasets/my_faces/",
    #     output_dir="distill_data/my_faces",
    #     max_samples=500_000,
    # )