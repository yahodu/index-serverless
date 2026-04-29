# 2_prepare_dataset.py
"""
Step 2: Generate teacher landmark labels from MS1MV3 for 2d106det distillation.

KEY FACTS for 2d106det.onnx
────────────────────────────
  Input resolution : 192 × 192  (MS1MV3 images are 112×112 — resized here)
  Preprocessing    : (RGB - 127.5) / 128.0, NHWC -> NCHW
  Raw output       : (N, 212) flat float32
  Decode           :
    reshape(N, 106, 2)
    xy = (xy + 1) * 96    pixel-space [0, 192]
    (no z — this is purely 2D)

WHY MS1MV3?
───────────
  The original 2d106det was trained on internal/undisclosed data.
  For knowledge distillation, we don't need ground-truth labels —
  the teacher *generates* pseudo-labels at inference time.
  MS1MV3 provides 5.8M diverse, well-cropped face images at 112×112
  which we resize to 192×192 to match teacher input. This is the
  same source dataset used for 1k3d68 distillation, so you can
  reuse the same .rec/.idx files.

STORAGE per chunk of 10 000 samples
─────────────────────────────────────
  images     : 10 000 × 192 × 192 × 3  uint8    ≈ 1 059 MB
  landmarks  : 10 000 × 106 × 2        float32  ≈     8.1 MB (vs 7.8 MB for 68×3)
  total                                          ≈ 1 067 MB / chunk

At 1 000 000 samples (100 chunks) -> ≈ 106 GB images + 0.81 GB landmarks
"""

import os
import sys
import struct
import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_SIZE  = 192
INPUT_MEAN  = 127.5
INPUT_STD   = 128.0
OUTPUT_FLAT = 212       # 106 * 2
NUM_LMK     = 106
LMK_DIM     = 2         # 2D only


# ── Landmark decoding (mirrors landmark.py — 2D branch) ───────────────────────

def decode_landmarks(raw: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
    """
    raw  : (N, 212) float32   raw session output
    out  : (N, 106, 2) float32  decoded pixel-space 2D landmarks

    Decoding from landmark.py (the else branch, output_shape[1] != 3309):
      pred = raw[i]                      # (212,)
      pred = pred.reshape((-1, 2))       # (106, 2)
      pred[:, 0:2] += 1
      pred[:, 0:2] *= 96                 # pixel space [0, 192]
    """
    half  = input_size // 2   # 96
    N     = raw.shape[0]
    lmks  = np.empty((N, NUM_LMK, LMK_DIM), dtype=np.float32)

    for i in range(N):
        pred = raw[i].reshape(-1, LMK_DIM).copy()   # (106, 2)
        pred[:, 0:2] += 1.0
        pred[:, 0:2] *= half
        lmks[i] = pred

    return lmks


# ── MXNet RecordIO reader ──────────────────────────────────────────────────────

class MXRecordDatasetReader:
    """
    Pure-Python MXNet RecordIO reader — no MXNet dependency.
    Identical to the 1k3d68 pipeline; reuse the same .rec/.idx files.

    Binary layout per record:
      [0:4]   magic      uint32 LE   must == 0xced7230a
      [4:8]   encoded    uint32 LE   lower 29 bits = data_len/4
      [8:24]  index+id   16 bytes    skipped
      [24:]   payload

    InsightFace payload layout:
      [0:16]  label header  (16 bytes, skipped)
      [16:]   JPEG bytes    (starts ff d8 ff)
    """

    _MAGIC     = 0xced7230a
    _JPEG_SKIP = 16

    def __init__(self, rec_path: str, idx_path: str,
                 target_size: int = INPUT_SIZE):
        self.target_size = target_size
        self.offsets: dict = {}

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
        print(f"Dataset: {len(self.keys):,} images "
              f"(skipped 1 header, {len(all_keys):,} total keys)")

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
        return payload[self._JPEG_SKIP:]

    def read_image(self, index: int) -> np.ndarray:
        """Returns (192, 192, 3) uint8 RGB."""
        key    = self.keys[index]
        offset = self.offsets[key]
        raw    = self._read_record_at(offset)

        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"imdecode failed key={key}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.target_size, self.target_size):
            img = cv2.resize(img, (self.target_size, self.target_size),
                             interpolation=cv2.INTER_LINEAR)
        return img

    def read_batch(self, indices: list) -> np.ndarray:
        images = []
        for idx in indices:
            try:
                images.append(self.read_image(idx))
            except Exception as e:
                print(f"  [warn] skipping index {idx}: {e}")
        return np.stack(images, axis=0) if images else np.array([])


# ── Teacher Landmark Generator ─────────────────────────────────────────────────

class TeacherLandmarkGenerator:
    """
    Runs 2d106det.onnx on batches of face images and stores decoded
    2D landmark coordinates.

    Preprocessing: (RGB - 127.5) / 128.0  NHWC -> NCHW
    Output: (N, 106, 2) float32 in pixel space [0, 192]
    """

    def __init__(
        self,
        onnx_path:        str,
        batch_size:       int   = 32,
        gpu_mem_limit_gb: float = 4.0,
    ):
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern   = True
        sess_options.enable_cpu_mem_arena = True
        sess_options.log_severity_level   = 3

        cuda_options = {
            "device_id":              0,
            "arena_extend_strategy":  "kSameAsRequested",
            "gpu_mem_limit":          int(gpu_mem_limit_gb * 1024 ** 3),
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

        out_size = self.session.get_outputs()[0].shape[1]
        assert out_size == OUTPUT_FLAT, (
            f"Expected output size {OUTPUT_FLAT} (for 2d106det), got {out_size}.\n"
            f"If you got 3309, you accidentally loaded 1k3d68.onnx."
        )

        print(f"Teacher model loaded : {onnx_path}")
        print(f"  Batch size         : {batch_size}")
        print(f"  GPU mem cap        : {gpu_mem_limit_gb} GB")
        print(f"  Preprocessing      : (RGB - {INPUT_MEAN}) / {INPUT_STD}")
        print(f"  Output decode      : reshape(106,2) -> (xy+1)*96")
        print(f"  Providers          : {self.session.get_providers()}")

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """
        Input  : (N, 192, 192, 3) uint8 RGB
        Output : (N, 3, 192, 192) float32
        """
        imgs = images.astype(np.float32)
        imgs = (imgs - INPUT_MEAN) / INPUT_STD
        imgs = imgs.transpose(0, 3, 1, 2)   # NHWC -> NCHW
        return np.ascontiguousarray(imgs)

    def get_landmarks(self, images: np.ndarray) -> np.ndarray:
        """
        Input  : (N, 192, 192, 3) uint8 RGB
        Output : (N, 106, 2)      float32 decoded pixel-space landmarks
        """
        blob = self.preprocess(images)
        raw  = self.session.run(None, {self.input_name: blob})[0]   # (N, 212)
        return decode_landmarks(raw)

    @staticmethod
    def _save_chunk(path: str, images: np.ndarray, landmarks: np.ndarray):
        """
        Save uncompressed .npz chunk.
        Keys: 'images' (N,192,192,3) uint8, 'landmarks' (N,106,2) float32
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
        """
        os.makedirs(output_dir, exist_ok=True)
        reader = MXRecordDatasetReader(rec_path, idx_path, target_size=INPUT_SIZE)

        total = min(len(reader), max_samples) if max_samples else len(reader)
        print(f"\nProcessing {total:,} images  |  chunk_size={chunk_size:,}")
        print(f"Storage estimate : {total * INPUT_SIZE * INPUT_SIZE * 3 / 1e9:.1f} GB images  "
              f"+ {total * NUM_LMK * LMK_DIM * 4 / 1e6:.1f} MB landmarks")
        print(f"Output -> {output_dir}\n")

        chunk_images    = []
        chunk_landmarks = []
        chunk_idx       = 0
        chunk_count     = 0
        pbar            = tqdm(total=total, desc="Teacher 2D labels")

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

                print(f"\n  Saved {chunk_file}  "
                      f"{all_imgs.shape[0]:,} samples  "
                      f"images={all_imgs.nbytes / 1e6:.0f} MB  "
                      f"landmarks={all_lmks.nbytes / 1e6:.2f} MB")

                chunk_images    = []
                chunk_landmarks = []
                chunk_idx      += 1
                chunk_count     = 0

        pbar.close()
        print(f"\nDone — {chunk_idx} chunk(s) written to {output_dir}")


# ── Sanity checks ──────────────────────────────────────────────────────────────

def sanity_check_reader(rec_path: str, idx_path: str, n: int = 5) -> bool:
    print("\n=== Sanity check: RecordIO reader ===")
    reader = MXRecordDatasetReader(rec_path, idx_path)
    ok = 0
    for i in range(min(n, len(reader))):
        try:
            img = reader.read_image(i)
            assert img.shape == (INPUT_SIZE, INPUT_SIZE, 3), img.shape
            assert img.dtype == np.uint8
            print(f"  [{i}] shape={img.shape} dtype={img.dtype} "
                  f"min={img.min()} max={img.max()}  OK")
            ok += 1
        except Exception as e:
            print(f"  [{i}] FAILED: {e}")
    print(f"Reader: {ok}/{n} passed")
    return ok == n


def sanity_check_model(gen: TeacherLandmarkGenerator, n: int = 2) -> bool:
    print("\n=== Sanity check: ONNX model + 2D decode ===")
    dummy = np.random.randint(0, 256, (n, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    try:
        lmks = gen.get_landmarks(dummy)
        assert lmks.shape == (n, NUM_LMK, LMK_DIM), f"Bad shape: {lmks.shape}"
        print(f"  Output shape : {lmks.shape}  OK   (N, 106, 2)")
        print(f"  x range      : [{lmks[:,:,0].min():.2f}, {lmks[:,:,0].max():.2f}]  (expect ~0–{INPUT_SIZE})")
        print(f"  y range      : [{lmks[:,:,1].min():.2f}, {lmks[:,:,1].max():.2f}]  (expect ~0–{INPUT_SIZE})")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    REC_PATH = "datasets/ms1m-retinaface-t1/train.rec"
    IDX_PATH = "datasets/ms1m-retinaface-t1/train.idx"

    gen = TeacherLandmarkGenerator(
        onnx_path        = "teacher_model/2d106det.onnx",
        batch_size       = 64,      # 2d106det is tiny — bigger batches are fine
        gpu_mem_limit_gb = 4.0,
    )

    if not sanity_check_reader(REC_PATH, IDX_PATH):
        print("ERROR: reader failed"); sys.exit(1)

    if not sanity_check_model(gen):
        print("ERROR: model failed"); sys.exit(1)

    gen.process_dataset(
        rec_path    = REC_PATH,
        idx_path    = IDX_PATH,
        output_dir  = "distill_data/ms1m_landmarks_2d106",
        max_samples = 1_000_000,
        chunk_size  = 10_000,
    )
