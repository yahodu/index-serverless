# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2_prepare_dataset_scrfd.py                                                  ║
# ║                                                                              ║
# ║  Download Open Images V7 (CC-BY images only, "Human face" class),           ║
# ║  run scrfd_10g_bnkps.onnx over every image, and store the raw 9-tensor      ║
# ║  teacher outputs as pseudo-labels in .npz chunks.                            ║
# ║                                                                              ║
# ║  WHY PURE PSEUDO-LABELS (no OI GT boxes used for supervision):               ║
# ║    The teacher outputs RAW distance predictions (pre-stride).                ║
# ║    Open Images GT boxes are in a different format (normalised xmin/xmax).   ║
# ║    Matching GT to 16,800 anchors per image adds complexity with no           ║
# ║    benefit when the teacher already provides dense, calibrated targets.      ║
# ║    This exactly mirrors the 2d106det / 1k3d68 pipeline.                     ║
# ║                                                                              ║
# ║  STORAGE per chunk of 1 000 images:                                          ║
# ║    images    : 1000 × 640 × 640 × 3  uint8    ≈ 1 229 MB                    ║
# ║    score_*   : 3 × 1000 × {12800,3200,800} × 1  float32  ≈ 65 MB            ║
# ║    bbox_*    : 3 × 1000 × {12800,3200,800} × 4  float32  ≈ 259 MB           ║
# ║    kps_*     : 3 × 1000 × {12800,3200,800} × 10 float32  ≈ 648 MB           ║
# ║    total                                         ≈  2.2 GB / chunk           ║
# ║  Recommendation: use chunk_size=500 or enable stream mode in training.       ║
# ║                                                                              ║
# ║  Open Images V7 "Human face" label MID: /m/0dzct                            ║
# ║  CC-BY-safe image filter applied via metadata CSV (License column).          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import sys
import csv
import json
import glob
import shutil
import argparse
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm

# ── Optional fiftyone import (download step only) ─────────────────────────────
try:
    import fiftyone as fo
    import fiftyone.zoo as foz
    HAS_FIFTYONE = True
except ImportError:
    HAS_FIFTYONE = False

# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_SIZE  = 640
INPUT_MEAN  = 127.5
INPUT_STD   = 128.0
FMC         = 3
STRIDES     = [8, 16, 32]
NUM_ANCHORS = 2

# Anchor counts at 640×640
ANCHOR_COUNTS = {
    8:  (INPUT_SIZE // 8)  * (INPUT_SIZE // 8)  * NUM_ANCHORS,   # 12800
    16: (INPUT_SIZE // 16) * (INPUT_SIZE // 16) * NUM_ANCHORS,   #  3200
    32: (INPUT_SIZE // 32) * (INPUT_SIZE // 32) * NUM_ANCHORS,   #   800
}

# CC-BY-compatible Open Images license IDs (commercial use allowed)
# From Open Images metadata CSV: 1=CC-BY (Attribution), 2=CC-BY-SA,
# 4=CC-BY-NC (skip), 5=CC-BY-NC-SA (skip), 6=CC-BY-NC-ND (skip)
# Only IDs present in OI train that allow commercial use:
CC_BY_LICENSE_IDS = {"1", "2", "3"}  # Attribution, Attribution-ShareAlike only

OUTPUT_NAMES = [
    "score_8", "score_16", "score_32",
    "bbox_8",  "bbox_16",  "bbox_32",
    "kps_8",   "kps_16",   "kps_32",
]


# ══════════════════════════════════════════════════════════════════════════════
# Step 0 — Download Open Images V7 "Human face" subset via FiftyOne
# ══════════════════════════════════════════════════════════════════════════════

def download_open_images(
    output_dir:  str,
    max_samples: int = 200_000,
    splits:      list = ("train",),
) -> str:
    """
    Downloads the 'Human face' detection subset of Open Images V7 using
    FiftyOne. Images are saved under output_dir/images/.
    Returns the path to the metadata CSV we write for license filtering.

    Requires: pip install fiftyone
    """
    if not HAS_FIFTYONE:
        raise ImportError(
            "fiftyone is required for the download step.\n"
            "  pip install fiftyone\n"
            "If you already have images elsewhere, skip to step 1 and set\n"
            "IMAGE_DIR in the config dict."
        )

    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    all_paths = []

    for split in splits:
        print(f"\nDownloading Open Images V7 '{split}' split — "
              f"class: 'Human face'  max_samples={max_samples:,}")

        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split          = split,
            label_types    = ["detections"],
            classes        = ["Human face"],
            max_samples    = max_samples,
            shuffle        = True,
            seed           = 42,
            dataset_name   = f"oi7_face_{split}_{max_samples}",
        )

        print(f"  Downloaded {len(dataset)} samples")

        # Export image paths + open images IDs to a simple CSV
        meta_csv = os.path.join(output_dir, f"meta_{split}.csv")
        with open(meta_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["oi_id", "filepath"])
            for sample in tqdm(dataset, desc=f"Collecting {split} paths"):
                oi_id = sample.get_field("open_images_id") or ""
                writer.writerow([oi_id, sample.filepath])
                all_paths.append(sample.filepath)

        print(f"  Metadata written to {meta_csv}")

    # Write combined list
    combined = os.path.join(output_dir, "all_image_paths.txt")
    with open(combined, "w") as f:
        for p in all_paths:
            f.write(p + "\n")

    print(f"\nTotal images collected: {len(all_paths):,}")
    print(f"Image list: {combined}")
    return combined


def filter_cc_by_images(
    image_list_txt:  str,
    oi_metadata_csv: str,
    output_txt:      str,
) -> int:
    """
    Filter image paths to only CC-BY-compatible images using the Open Images
    image metadata CSV (download from Open Images website):
      https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv

    The CSV has columns: ImageID, Subset, OriginalURL, OriginalLandingURL,
    License, AuthorProfileURL, Author, Title, OriginalSize, OriginalMD5,
    Thumbnail300KURL, Rotation

    License column values (URL fragments):
      licenses/by/2.0/      -> CC BY 2.0   (commercial ✓)
      licenses/by-sa/2.0/   -> CC BY-SA 2.0 (commercial ✓)
      licenses/by-nd/2.0/   -> CC BY-ND 2.0 (commercial ✓ no-deriv, use as-is)
      licenses/by-nc/2.0/   -> CC BY-NC 2.0 (commercial ✗ SKIP)
      licenses/by-nc-sa/2.0 -> CC BY-NC-SA  (commercial ✗ SKIP)
      licenses/by-nc-nd/2.0 -> CC BY-NC-ND  (commercial ✗ SKIP)
    """
    print(f"Loading OI image metadata from {oi_metadata_csv}...")
    allowed_ids = set()

    with open(oi_metadata_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lic = row.get("License", "")
            # Keep CC-BY variants that allow commercial use (no -nc)
            if "nc" not in lic.lower():
                # Extract image ID (last part of URL or direct ID)
                image_id = row.get("ImageID", "").strip()
                if image_id:
                    allowed_ids.add(image_id)

    print(f"  CC-BY-safe image IDs: {len(allowed_ids):,}")

    with open(image_list_txt) as f:
        all_paths = [l.strip() for l in f if l.strip()]

    kept = []
    skipped = 0
    for p in all_paths:
        # Extract OI image ID from filename (e.g. abc123def456.jpg -> abc123def456)
        stem = Path(p).stem
        if stem in allowed_ids:
            kept.append(p)
        else:
            skipped += 1

    with open(output_txt, "w") as f:
        for p in kept:
            f.write(p + "\n")

    print(f"  Kept: {len(kept):,}  Skipped (NC): {skipped:,}")
    print(f"  CC-BY filtered list: {output_txt}")
    return len(kept)


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def letterbox_bgr(img_bgr: np.ndarray, size: int = INPUT_SIZE) -> tuple:
    """
    Aspect-ratio preserving resize + bottom/right zero-pad.
    Mirrors SCRFD.detect() preprocessing exactly.

    Returns:
        canvas (size, size, 3) uint8 BGR
        scale  float — original_pixel = canvas_pixel / scale
    """
    h, w  = img_bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, scale


def preprocess_batch(images_bgr: list) -> np.ndarray:
    """
    Input : list of (H, W, 3) BGR uint8 images (already letterboxed to 640×640)
    Output: (N, 3, 640, 640) float32
    Matches: cv2.dnn.blobFromImage(..., swapRB=True)
    """
    blobs = []
    for img in images_bgr:
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb - INPUT_MEAN) / INPUT_STD
        blobs.append(norm.transpose(2, 0, 1))
    return np.stack(blobs, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Teacher runner
# ══════════════════════════════════════════════════════════════════════════════

class SCRFDTeacher:
    """
    Runs scrfd_10g_bnkps.onnx on batches and returns the raw 9 output tensors.

    IMPORTANT: We store RAW outputs (pre-stride multiplication).
    InsightFace's scrfd.py multiplies bbox_* and kps_* by the stride at
    inference time. The student must learn to reproduce these same raw values.
    """

    def __init__(
        self,
        onnx_path:        str,
        batch_size:       int   = 4,
        gpu_mem_limit_gb: float = 6.0,
    ):
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3

        cuda_opts = {
            "device_id":              0,
            "arena_extend_strategy":  "kSameAsRequested",
            "gpu_mem_limit":          int(gpu_mem_limit_gb * 1024 ** 3),
            "cudnn_conv_algo_search": "HEURISTIC",
        }

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options = sess_opts,
            providers    = [
                ("CUDAExecutionProvider", cuda_opts),
                "CPUExecutionProvider",
            ],
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.batch_size  = batch_size

        assert len(self.output_names) == 9, (
            f"Expected 9 outputs, got {len(self.output_names)}. "
            f"Wrong model? (needs scrfd_*_bnkps.onnx)"
        )
        print(f"Teacher loaded        : {onnx_path}")
        print(f"  Input name          : {self.input_name}")
        print(f"  Output names        : {self.output_names}")
        print(f"  Providers           : {self.session.get_providers()}")
        print(f"  Batch size          : {batch_size}")

    def run_batch(self, blob: np.ndarray) -> list:
        """
        blob : (N, 3, 640, 640) float32
        Returns list of 9 arrays, each (N, anchors, channels).
        Note: scrfd_10g_bnkps in non-batched mode returns (anchors, channels).
        We check and handle both batched (shape[0]==3) and non-batched outputs.
        """
        outs = self.session.run(self.output_names, {self.input_name: blob})
        # outs[i] shape is either (N, A, C) [batched] or (A, C) [non-batched]
        # The publicly released scrfd_10g_bnkps.onnx is non-batched (no batch dim).
        # We run N=1 at a time for simplicity and correctness.
        return outs

    def process_single(self, blob_1: np.ndarray) -> list:
        """
        blob_1: (1, 3, 640, 640) float32
        Returns list of 9 numpy arrays with batch dim removed:
          score_8  (12800, 1), score_16 (3200, 1), score_32 (800, 1)
          bbox_8   (12800, 4), ...
          kps_8    (12800,10), ...
        """
        outs = self.session.run(self.output_names, {self.input_name: blob_1})
        # Handle both batched and non-batched ONNX exports
        result = []
        for arr in outs:
            if arr.ndim == 3:    # (1, A, C) batched
                result.append(arr[0])
            else:                # (A, C) non-batched
                result.append(arr)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Chunk I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_chunk(path: str, images: np.ndarray, labels: dict):
    """
    Save one chunk.
    images : (N, 640, 640, 3) uint8
    labels : dict with keys matching OUTPUT_NAMES, each (N, A, C) float32
    """
    np.savez_compressed(path, images=images, **labels)


def load_chunk(path: str) -> tuple:
    """Returns (images, labels_dict)."""
    d      = np.load(path)
    images = d["images"]
    labels = {k: d[k] for k in OUTPUT_NAMES}
    d.close()
    return images, labels


# ══════════════════════════════════════════════════════════════════════════════
# Main processing pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_dataset(
    image_list_txt:  str,
    onnx_path:       str,
    output_dir:      str,
    max_samples:     int  = None,
    chunk_size:      int  = 500,
    gpu_mem_limit_gb: float = 6.0,
    skip_existing:   bool = True,
):
    """
    Reads image paths from image_list_txt, letterboxes each to 640×640,
    runs teacher inference, saves .npz chunks to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(image_list_txt) as f:
        image_paths = [l.strip() for l in f if l.strip()]

    if max_samples:
        image_paths = image_paths[:max_samples]

    total = len(image_paths)
    print(f"\nProcessing {total:,} images  chunk_size={chunk_size}")

    # Storage estimate
    img_gb  = total * 640 * 640 * 3 / 1e9
    lbl_gb  = total * (12800 + 3200 + 800) * (1 + 4 + 10) * 4 / 1e9
    print(f"Storage estimate: {img_gb:.1f} GB images  {lbl_gb:.1f} GB labels  "
          f"({img_gb + lbl_gb:.1f} GB total)")
    print(f"Output -> {output_dir}\n")

    teacher = SCRFDTeacher(onnx_path, batch_size=1,
                           gpu_mem_limit_gb=gpu_mem_limit_gb)

    chunk_imgs   = []
    chunk_labels = {k: [] for k in OUTPUT_NAMES}
    chunk_idx    = 0
    skipped      = 0

    # Find existing chunks to resume
    existing = set(
        int(Path(p).stem.split("_")[1])
        for p in glob.glob(os.path.join(output_dir, "chunk_*.npz"))
    ) if skip_existing else set()
    if existing:
        completed = max(existing) + 1
        start_img = completed * chunk_size
        chunk_idx = completed
        print(f"Resuming from chunk {chunk_idx} (image {start_img:,})")
        image_paths = image_paths[start_img:]

    pbar = tqdm(image_paths, desc="Teacher labels", unit="img")

    for img_path in pbar:
        # ── Load & letterbox ──────────────────────────────────
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            skipped += 1
            continue

        canvas, _ = letterbox_bgr(img_bgr)
        blob      = preprocess_batch([canvas])   # (1, 3, 640, 640)

        # ── Teacher inference ─────────────────────────────────
        try:
            outs = teacher.process_single(blob)
        except Exception as e:
            print(f"  [warn] inference failed for {img_path}: {e}")
            skipped += 1
            continue

        # ── Accumulate ────────────────────────────────────────
        chunk_imgs.append(canvas)
        for name, arr in zip(OUTPUT_NAMES, outs):
            chunk_labels[name].append(arr)

        # ── Flush chunk ───────────────────────────────────────
        if len(chunk_imgs) >= chunk_size:
            _flush_chunk(output_dir, chunk_idx,
                         chunk_imgs, chunk_labels)
            chunk_imgs   = []
            chunk_labels = {k: [] for k in OUTPUT_NAMES}
            chunk_idx   += 1

        pbar.set_postfix(chunks=chunk_idx, skipped=skipped)

    # Final partial chunk
    if chunk_imgs:
        _flush_chunk(output_dir, chunk_idx, chunk_imgs, chunk_labels)
        chunk_idx += 1

    pbar.close()
    print(f"\nDone — {chunk_idx} chunks written to {output_dir}")
    print(f"Skipped (bad images): {skipped}")


def _flush_chunk(output_dir, idx, imgs, labels):
    all_imgs = np.stack(imgs, axis=0)
    all_lbl  = {k: np.stack(v, axis=0) for k, v in labels.items()}
    path     = os.path.join(output_dir, f"chunk_{idx:04d}.npz")
    save_chunk(path, all_imgs, all_lbl)

    total_mb = (all_imgs.nbytes + sum(v.nbytes for v in all_lbl.values())) / 1e6
    print(f"\n  Saved {path}  N={len(all_imgs)}  {total_mb:.0f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# Sanity checks
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check(onnx_path: str, image_path: str = None):
    print("\n=== Sanity check ===")
    teacher = SCRFDTeacher(onnx_path, batch_size=1)

    if image_path and Path(image_path).exists():
        img = cv2.imread(image_path)
    else:
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        print("  Using random dummy image")

    canvas, scale = letterbox_bgr(img)
    blob = preprocess_batch([canvas])
    outs = teacher.process_single(blob)

    expected_shapes = [
        (12800, 1), (3200, 1), (800, 1),
        (12800, 4), (3200, 4), (800, 4),
        (12800,10), (3200,10), (800,10),
    ]
    all_ok = True
    for name, arr, expected in zip(OUTPUT_NAMES, outs, expected_shapes):
        ok = arr.shape == expected
        print(f"  {'✓' if ok else '✗'} {name:10s}  {arr.shape}  "
              f"range=[{arr.min():.3f}, {arr.max():.3f}]")
        if not ok:
            all_ok = False

    # Count detections at threshold 0.5
    score_concat = np.concatenate([outs[0], outs[1], outs[2]], axis=0)
    n_det = int((score_concat >= 0.5).sum())
    print(f"\n  Detections (score≥0.5): {n_det} anchors out of "
          f"{len(score_concat)} total")
    print(f"  Sanity: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download",    action="store_true",
                        help="Run FiftyOne download step first")
    parser.add_argument("--filter_cc",  action="store_true",
                        help="Filter to CC-BY images using OI metadata CSV")
    parser.add_argument("--oi_meta_csv", default=None,
                        help="Path to Open Images image metadata CSV "
                             "(for --filter_cc)")
    parser.add_argument("--image_list",  default=None,
                        help="Path to text file of image paths (skip download)")
    parser.add_argument("--max_download", type=int, default=200_000)
    parser.add_argument("--max_process",  type=int, default=None)
    parser.add_argument("--chunk_size",   type=int, default=500)
    parser.add_argument("--sanity_image", default=None)
    args = parser.parse_args()

    ONNX_PATH  = "teacher_model/scrfd_10g_bnkps.onnx"
    DOWNLOAD_DIR = "datasets/open_images_face"
    OUTPUT_DIR   = "distill_data/scrfd_pseudo_labels"

    # ── Step 0: Download (optional) ───────────────────────────
    if args.download:
        image_list = download_open_images(
            DOWNLOAD_DIR, max_samples=args.max_download
        )
    else:
        image_list = args.image_list or os.path.join(
            DOWNLOAD_DIR, "all_image_paths.txt"
        )

    # ── Step 0b: CC-BY filter (optional) ─────────────────────
    if args.filter_cc:
        if not args.oi_meta_csv:
            print("ERROR: --oi_meta_csv required with --filter_cc")
            print("Download from: "
                  "https://storage.googleapis.com/openimages/2018_04/"
                  "image_ids_and_rotation.csv")
            sys.exit(1)
        filtered_list = image_list.replace(".txt", "_ccby.txt")
        filter_cc_by_images(image_list, args.oi_meta_csv, filtered_list)
        image_list = filtered_list

    # ── Sanity check ──────────────────────────────────────────
    if not sanity_check(ONNX_PATH, args.sanity_image):
        print("ERROR: sanity check failed"); sys.exit(1)

    # ── Process ───────────────────────────────────────────────
    process_dataset(
        image_list_txt   = image_list,
        onnx_path        = ONNX_PATH,
        output_dir       = OUTPUT_DIR,
        max_samples      = args.max_process,
        chunk_size       = args.chunk_size,
        gpu_mem_limit_gb = 6.0,
    )
