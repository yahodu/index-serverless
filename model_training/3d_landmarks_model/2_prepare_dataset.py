# 2_prepare_dataset.py
"""
Step 2: Generate teacher landmark labels from MS1MV3.

KEY FACTS for 1k3d68.onnx
──────────────────────────
  Input resolution : 192 × 192  (MS1MV3 images are 112×112 — resized here)
  Normalisation    : (RGB - 127.5) / 128.0
  Raw output       : (N, 3309)  flat float32
  Decode           : reshape (N, 68, 3)
                     xy  = (xy  + 1) * 96    pixel-space
                     z   =  z        * 96    relative depth

STORAGE per chunk of 10 000 samples
─────────────────────────────────────
  images     : 10 000 × 192 × 192 × 3  uint8  ≈ 1.06 GB
  landmarks  : 10 000 × 68  × 3        float32 ≈  7.8 MB
  total                                         ≈ 1.07 GB / chunk

At 1 000 000 samples (100 chunks) → ≈ 107 GB images + 0.78 GB landmarks
"""

import os
import sys
import struct
import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm
from pathlib import Path


# ============================================================
# Constants
# ============================================================

INPUT_SIZE   = 192          # 1k3d68 input resolution
INPUT_MEAN   = 127.5
INPUT_STD    = 128.0
OUTPUT_FLAT  = 3309         # 68 * 3 * ~  (confirmed by inspect)
NUM_LMK      = 68
LMK_DIM      = 3


# ============================================================
# Landmark decoding (mirrors insightface/model_zoo/landmark.py)
# ============================================================

def decode_landmarks(raw: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
    """
    raw  : (N, 3309) float32
    out  : (N, 68, 3) float32  — xy in [0, input_size], z same scale
    """
    half = input_size // 2
    lmks = raw.reshape(raw.shape[0], NUM_LMK, LMK_DIM).copy()
    lmks[:, :, 0:2] += 1.0
    lmks[:, :, 0:2] *= half
    lmks[:, :, 2]   *= half
    return lmks


# ============================================================
# MXNet RecordIO reader  (same as glintr100 version, resize added)
# ============================================================

class MXRecordDatasetReader:
    """
    Pure-Python MXNet RecordIO reader — no MXNet dependency.
    MS1MV3 images are 112×112 JPEG; we resize to 192×192 on read.

    Binary layout:
      [0:4]  magic      uint32 LE  0xced7230a
      [4:8]  encoded    uint32 LE  upper 3 bits = cflag, lower 29 bits = data_len/4
      [8:24] index+id   16 bytes   skipped
      [24:]  payload

    InsightFace payload layout:
      [0:16]  label header   (16 bytes)
      [16:]   JPEG bytes     (starts ff d8 ff)

    Key=0 is the dataset header record — always skipped.
    """

    _MAGIC     = 0xced7230a
    _JPEG_SKIP = 16

    def __init__(self, rec_path: str, idx_path: str,
                 target_size: int = INPUT_SIZE):
        self.target_size = target_size

        self.offsets: dict[int, int] = {}
        with open(idx_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 2:
                    self.offsets[int(parts[0])] = int(float(parts[1]))

        all_keys    = sorted(self.offsets.keys())
        self.keys   = [k for k in all_keys if k != 0]
        print(f"Dataset: {len(self.keys)} images "
              f"(skipped 1 header, {len(all_keys)} total keys)")

        self._rec_file = open(rec_path, 'rb')

    def __len__(self):
        return len(self.keys)

    def __del__(self):
        if hasattr(self, '_rec_file') and self._rec_file:
            self._rec_file.close()

    def _read_record_at(self, offset: int) -> bytes:
        self._rec_file.seek(offset)

        raw = self._rec_file.read(4)
        if len(raw) < 4:
            raise ValueError(f"EOF at offset {offset}")
        magic = struct.unpack_from('<I', raw)[0]
        if magic != self._MAGIC:
            raise ValueError(f"Bad magic 0x{magic:08x} at offset {offset}")

        raw      = self._rec_file.read(4)
        encoded  = struct.unpack_from('<I', raw)[0]
        data_len = (encoded & 0x1FFFFFFF) * 4

        self._rec_file.read(16)   # skip index + id

        payload_len = data_len - 16
        if payload_len <= self._JPEG_SKIP:
            raise ValueError(f"Payload too small ({payload_len} B) at {offset}")

        payload    = self._rec_file.read(payload_len)
        jpeg_bytes = payload[self._JPEG_SKIP:]
        return jpeg_bytes

    def read_image(self, index: int) -> np.ndarray:
        """
        Returns (192, 192, 3) uint8 RGB.
        MS1MV3 source images are 112×112 — resized to 192×192 here.
        Bilinear resize is sufficient; the teacher never saw sub-pixel
        detail beyond what 112px contains anyway.
        """
        key    = self.keys[index]
        offset = self.offsets[key]
        raw    = self._read_record_at(offset)

        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(
                f"imdecode failed key={key} offset={offset} jpeg_len={len(raw)}"
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.target_size, self.target_size):
            img = cv2.resize(
                img, (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR,
            )
        return img

    def read_batch(self, indices: list) -> np.ndarray:
        images = []
        for idx in indices:
            try:
                images.append(self.read_image(idx))
            except Exception as e:
                print(f"  [warn] skipping index {idx}: {e}")
        return np.stack(images, axis=0) if images else np.array([])


# ============================================================
# Teacher Landmark Generator
# ============================================================

class TeacherLandmarkGenerator:
    """
    Runs 1k3d68.onnx on batches of face images and stores decoded
    landmark coordinates.

    Preprocessing mirrors insightface/model_zoo/landmark.py exactly:
      blob = cv2.dnn.blobFromImage(img, 1/128.0, (192,192), (127.5,)*3, swapRB=True)
    which is equivalent to:
      (RGB_float - 127.5) / 128.0,  transposed to NCHW

    Recommended batch sizes by GPU VRAM:
       4 GB  ->  8
       8 GB  -> 16
      16 GB  -> 32
      24 GB+ -> 64
    """

    def __init__(
        self,
        onnx_path:        str,
        batch_size:       int   = 16,
        gpu_mem_limit_gb: float = 4.0,
    ):
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern   = True
        sess_options.enable_cpu_mem_arena = True
        sess_options.log_severity_level   = 3

        cuda_options = {
            "device_id":                0,
            "arena_extend_strategy":    "kSameAsRequested",
            "gpu_mem_limit":            int(gpu_mem_limit_gb * 1024**3),
            "cudnn_conv_algo_search":   "HEURISTIC",
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

        # Detect normalisation from graph
        import onnx as _onnx
        _model     = _onnx.load(onnx_path)
        find_sub   = find_mul = False
        for nid, node in enumerate(_model.graph.node[:8]):
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
            if nid < 3 and node.name == 'bn_data':
                find_sub = find_mul = True
        self.input_mean = 0.0   if (find_sub and find_mul) else INPUT_MEAN
        self.input_std  = 1.0   if (find_sub and find_mul) else INPUT_STD

        print(f"Teacher model loaded : {onnx_path}")
        print(f"  Batch size         : {batch_size}")
        print(f"  GPU mem cap        : {gpu_mem_limit_gb} GB")
        print(f"  Normalisation      : mean={self.input_mean}, std={self.input_std}")
        print(f"  Providers          : {self.session.get_providers()}")

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """
        Input  : (N, 192, 192, 3) uint8 RGB
        Output : (N, 3,   192, 192) float32

        Mirrors cv2.dnn.blobFromImage with swapRB=True:
          output = (input_RGB - mean) / std  then NHWC -> NCHW
        """
        imgs = images.astype(np.float32)
        imgs = (imgs - self.input_mean) / self.input_std
        imgs = imgs.transpose(0, 3, 1, 2)   # NHWC -> NCHW
        return imgs

    def get_landmarks(self, images: np.ndarray) -> np.ndarray:
        """
        Input  : (N, 192, 192, 3) uint8 RGB
        Output : (N, 68, 3)       float32  decoded pixel-space landmarks
        """
        preprocessed = self.preprocess(images)
        raw          = self.session.run(None, {self.input_name: preprocessed})[0]
        return decode_landmarks(raw, input_size=INPUT_SIZE)

    @staticmethod
    def _save_chunk(path: str, images: np.ndarray, landmarks: np.ndarray):
        """
        Save uncompressed .npz chunk.

        Keys:
          'images'    : (N, 192, 192, 3) uint8
          'landmarks' : (N, 68, 3)       float32
        """
        np.savez(path, images=images, landmarks=landmarks)

    def process_dataset(
        self,
        rec_path:    str,
        idx_path:    str,
        output_dir:  str,
        max_samples: int  = None,
        chunk_size:  int  = 10_000,
    ):
        """
        Full pipeline: read MS1MV3 .rec -> run teacher -> save .npz chunks.

        chunk_size=10_000 keeps each chunk ≈ 1.07 GB (images dominate).
        Reduce to 5_000 if RAM is tight during concatenation.
        """
        os.makedirs(output_dir, exist_ok=True)
        reader = MXRecordDatasetReader(rec_path, idx_path, target_size=INPUT_SIZE)

        total = min(len(reader), max_samples) if max_samples else len(reader)
        print(f"\nProcessing {total:,} images  |  chunk_size={chunk_size}")
        print(f"Output -> {output_dir}")

        chunk_images    = []
        chunk_landmarks = []
        chunk_idx       = 0
        chunk_count     = 0
        pbar            = tqdm(total=total, desc="Teacher labels")

        for start in range(0, total, self.batch_size):
            end     = min(start + self.batch_size, total)
            indices = list(range(start, end))

            batch_imgs = reader.read_batch(indices)
            if len(batch_imgs) == 0:
                continue

            lmks = self.get_landmarks(batch_imgs)

            chunk_images.append(batch_imgs)
            chunk_landmarks.append(lmks)
            chunk_count += len(batch_imgs)
            pbar.update(len(batch_imgs))

            flush = (chunk_count >= chunk_size) or (end >= total)
            if flush and chunk_images:
                all_imgs = np.concatenate(chunk_images,    axis=0)
                all_lmks = np.concatenate(chunk_landmarks, axis=0)

                chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
                self._save_chunk(chunk_file, all_imgs, all_lmks)

                img_mb = all_imgs.nbytes / 1e6
                lmk_mb = all_lmks.nbytes / 1e6
                print(
                    f"\n  Saved {chunk_file}  "
                    f"{all_imgs.shape[0]:,} samples  "
                    f"images={img_mb:.0f} MB  landmarks={lmk_mb:.1f} MB"
                )

                chunk_images    = []
                chunk_landmarks = []
                chunk_idx      += 1
                chunk_count     = 0

        pbar.close()
        print(f"\nDone — {chunk_idx} chunk(s) saved to {output_dir}")


# ============================================================
# Sanity checks
# ============================================================

def sanity_check_reader(rec_path: str, idx_path: str, n: int = 5) -> bool:
    print("\n=== Sanity check: RecordIO reader ===")
    reader = MXRecordDatasetReader(rec_path, idx_path)
    ok = 0
    for i in range(min(n, len(reader))):
        try:
            img = reader.read_image(i)
            assert img.shape  == (INPUT_SIZE, INPUT_SIZE, 3), img.shape
            assert img.dtype  == np.uint8
            print(f"  [{i}] shape={img.shape} dtype={img.dtype} "
                  f"min={img.min()} max={img.max()}  OK")
            ok += 1
        except Exception as e:
            print(f"  [{i}] FAILED: {e}")
    print(f"Reader: {ok}/{n} passed")
    return ok == n


def sanity_check_model(gen: TeacherLandmarkGenerator, n: int = 2) -> bool:
    print("\n=== Sanity check: ONNX model ===")
    dummy = np.random.randint(0, 255, (n, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    try:
        lmks = gen.get_landmarks(dummy)
        assert lmks.shape == (n, NUM_LMK, LMK_DIM), f"Bad shape: {lmks.shape}"
        print(f"  Output shape : {lmks.shape}  OK   (N, 68, 3)")
        print(f"  x range      : [{lmks[:,:,0].min():.2f}, {lmks[:,:,0].max():.2f}]")
        print(f"  y range      : [{lmks[:,:,1].min():.2f}, {lmks[:,:,1].max():.2f}]")
        print(f"  z range      : [{lmks[:,:,2].min():.4f}, {lmks[:,:,2].max():.4f}]")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    REC_PATH = "datasets/ms1m-retinaface-t1/train.rec"
    IDX_PATH = "datasets/ms1m-retinaface-t1/train.idx"

    gen = TeacherLandmarkGenerator(
        onnx_path        = "teacher_model/1k3d68.onnx",
        batch_size       = 24,      # lower if VRAM < 8 GB
        gpu_mem_limit_gb = 12.0,
    )

    if not sanity_check_reader(REC_PATH, IDX_PATH):
        print("ERROR: reader failed"); sys.exit(1)

    if not sanity_check_model(gen):
        print("ERROR: model failed"); sys.exit(1)

    gen.process_dataset(
        rec_path    = REC_PATH,
        idx_path    = IDX_PATH,
        output_dir  = "distill_data/ms1m_landmarks",
        max_samples = 1_000_000,
        chunk_size  = 10_000,
    )
