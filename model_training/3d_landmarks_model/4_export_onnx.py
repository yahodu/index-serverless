# 4_export_onnx.py
"""
Step 4: Export trained 1k3d68 student model to ONNX.

Imports model definitions directly from 3_train_student.py.

Teacher output decode (official InsightFace landmark.py):
  raw shape : (1, 3309)
  reshape   : (1103, 3)
  slice     : [-68:, :]  →  (68, 3)   ← last 68 rows only
  xy decode : (xy + 1) * 96
  z  decode : z * 96

Usage:
    python 4_export_onnx.py
    python 4_export_onnx.py --checkpoint checkpoints_landmarks/run_.../best_model.pth
    python 4_export_onnx.py --teacher_onnx teacher_model/1k3d68.onnx --test_image face.jpg
    python 4_export_onnx.py --no_simplify
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort
import cv2
from pathlib import Path

# ── Import model definitions directly from training script ────────────────────
from train_student import (
    StudentModelSmall,
    StudentModelMedium,
    StudentModelLarge,
    INPUT_SIZE,   # 192
    NUM_LMK,      # 68
    LMK_DIM,      # 3
)

# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_MEAN  = 127.5
INPUT_STD   = 128.0

MODEL_MAP = {
    'small' : StudentModelSmall,
    'medium': StudentModelMedium,
    'large' : StudentModelLarge,
}

# Teacher raw output size — confirmed from InsightFace landmark.py
TEACHER_RAW_SIZE = 3309          # (1103, 3) → last 68 rows = landmarks


# ══════════════════════════════════════════════════════════════════════════════
# Teacher output decoder  (official InsightFace logic)
# ══════════════════════════════════════════════════════════════════════════════

def decode_teacher_output(raw: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
    """
    Decode raw 1k3d68.onnx output to pixel-space landmarks.

    Official logic from insightface/model_zoo/landmark.py:

        pred = session.run(...)[0][0]          # shape (3309,)
        pred = pred.reshape((-1, 3))           # (1103, 3)
        pred = pred[-68:, :]                   # (68,   3)  ← LAST 68 rows
        pred[:, 0:2] += 1                      # shift xy
        pred[:, 0:2] *= (input_size // 2)      # scale xy → pixels
        pred[:, 2]   *= (input_size // 2)      # scale z  → pixels

    Args:
        raw       : (3309,) flat float32 array — single sample, batch dim already removed
        input_size: teacher input resolution (192)

    Returns:
        (68, 3) float32 landmarks in pixel coordinates [0, ~192]
    """
    assert raw.shape == (TEACHER_RAW_SIZE,), (
        f"Expected teacher output shape ({TEACHER_RAW_SIZE},), got {raw.shape}.\n"
        f"Pass the [0][0] slice: output[0][0], not output[0] or output."
    )
    pred = raw.reshape(-1, 3)           # (1103, 3)
    pred = pred[-NUM_LMK:, :].copy()   # (68,   3)  ← critical: last 68 only
    pred[:, 0:2] += 1
    pred[:, 0:2] *= (input_size // 2)
    pred[:, 2]   *= (input_size // 2)
    return pred.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint loader
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load checkpoint and return (model, metadata_dict).
    Handles checkpoint key 'model_state_dict' from 3_train_student.py.
    """
    print(f"Loading checkpoint : {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ── State dict key ────────────────────────────────────────────────────
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        raise KeyError(
            f"Cannot find model weights in checkpoint.\n"
            f"Keys present: {list(ckpt.keys())}"
        )

    # ── Model size ────────────────────────────────────────────────────────
    config     = ckpt.get('config', {})
    model_size = config.get('model_size', None)

    if model_size is None:
        # Infer from architecture: check for stem layer and layer3 depth
        has_stem    = any(k.startswith('stem.') for k in state_dict)
        layer3_keys = [k for k in state_dict if k.startswith('layer3.')]
        n_blocks    = len(set(k.split('.')[1] for k in layer3_keys)) - 1

        if not has_stem:
            model_size = 'small'
        elif n_blocks <= 4:
            model_size = 'medium'
        else:
            model_size = 'large'
        print(f"  model_size        : '{model_size}'  (inferred from weights)")
    else:
        print(f"  model_size        : '{model_size}'  (from config)")

    if model_size not in MODEL_MAP:
        raise ValueError(f"Unknown model_size='{model_size}'. Choose from {list(MODEL_MAP.keys())}")

    model = MODEL_MAP[model_size]()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    meta = {
        'model_size': model_size,
        'epoch'     : ckpt.get('epoch', '?'),
        'nme'       : ckpt.get('nme',   None),
        'loss'      : ckpt.get('loss',  None),
    }

    print(f"  Epoch             : {meta['epoch']}")
    if meta['nme'] is not None:
        print(f"  Best NME          : {meta['nme']:.3f}%")
    print(f"  Parameters        : {sum(p.numel() for p in model.parameters()):,}")

    return model, meta


# ══════════════════════════════════════════════════════════════════════════════
# ONNX export + validation
# ══════════════════════════════════════════════════════════════════════════════

def export_to_onnx(
    model   : torch.nn.Module,
    out_path: str,
    opset   : int  = 17,
    simplify: bool = True,
) -> None:

    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    print(f"\nExporting to : {out_path}  (opset {opset})")
    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version       = opset,
        do_constant_folding = True,
        input_names         = ['input'],
        output_names        = ['landmarks'],
        dynamic_axes        = {
            'input'    : {0: 'batch_size'},
            'landmarks': {0: 'batch_size'},
        },
    )
    print("  ✓ Export complete")

    # ── Structural check ──────────────────────────────────────────────────
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model is structurally valid")

    # ── Optional simplification ───────────────────────────────────────────
    if simplify:
        try:
            import onnxsim
            print("\nSimplifying...")
            onnx_model, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(onnx_model, out_path)
                print("  ✓ Simplified and saved")
            else:
                print("  ⚠ Simplification could not be validated — keeping original")
        except ImportError:
            print("  ⚠ onnxsim not installed — skipping  (pip install onnx-simplifier)")

    # ── Output shape check ────────────────────────────────────────────────
    sess       = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    test_input = np.random.randn(2, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    ort_out    = sess.run(None, {'input': test_input})[0]

    expected = (2, NUM_LMK, LMK_DIM)
    assert ort_out.shape == expected, (
        f"Shape mismatch: got {ort_out.shape}, expected {expected}"
    )
    print(f"  ✓ Output shape    : {ort_out.shape}  (batch=2, lmk={NUM_LMK}, dim={LMK_DIM})")

    # ── PyTorch vs ONNX numerical agreement ───────────────────────────────
    with torch.no_grad():
        pt_out = model(torch.tensor(test_input)).numpy()

    max_diff = float(np.abs(ort_out - pt_out).max())
    print(f"  Max diff PT↔ONNX  : {max_diff:.2e}")
    assert max_diff < 1e-4, f"Output mismatch! max_diff={max_diff:.2e}"
    print("  ✓ PyTorch and ONNX outputs match")

    size_mb = os.path.getsize(out_path) / 1024 ** 2
    print(f"  File size         : {size_mb:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# Teacher vs Student comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_with_teacher(
    student_onnx : str,
    teacher_onnx : str,
    image_path   : str = None,
) -> None:
    print(f"\n{'─' * 60}")
    print("Teacher vs Student comparison")
    print(f"{'─' * 60}")

    # ── Load or create test image ──────────────────────────────────────────
    if image_path and Path(image_path).exists():
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ⚠ Could not read '{image_path}' — using grey placeholder")
            img = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
        else:
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"  Image  : {image_path}")
    else:
        img = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
        print("  Image  : grey placeholder (pass --test_image <face.jpg> for real results)")

    # Preprocess — matches both teacher and student preprocessing
    blob = img.astype(np.float32)
    blob = (blob - INPUT_MEAN) / INPUT_STD
    blob = blob.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 192, 192)

    # ── Teacher ───────────────────────────────────────────────────────────
    t_sess  = ort.InferenceSession(teacher_onnx, providers=['CPUExecutionProvider'])
    t_iname = t_sess.get_inputs()[0].name
    t_raw   = t_sess.run(None, {t_iname: blob})[0][0]   # (3309,)

    print(f"\n  Teacher raw output : shape={t_raw.shape}  "
          f"min={t_raw.min():.3f}  max={t_raw.max():.3f}")

    # Decode using official InsightFace logic
    t_lmk = decode_teacher_output(t_raw)             # (68, 3) pixel coords

    # ── Student ───────────────────────────────────────────────────────────
    s_sess = ort.InferenceSession(student_onnx, providers=['CPUExecutionProvider'])
    s_lmk  = s_sess.run(None, {'input': blob})[0][0]  # (68, 3) pixel coords

    print(f"  Teacher decoded    : shape={t_lmk.shape}  "
          f"xy_range=[{t_lmk[:,:2].min():.1f}, {t_lmk[:,:2].max():.1f}]")
    print(f"  Student output     : shape={s_lmk.shape}  "
          f"xy_range=[{s_lmk[:,:2].min():.1f}, {s_lmk[:,:2].max():.1f}]")

    # ── Metrics ───────────────────────────────────────────────────────────
    diff   = np.abs(t_lmk - s_lmk)
    diff_2d = np.linalg.norm(t_lmk[:, :2] - s_lmk[:, :2], axis=1)  # per-landmark 2D error

    # Inter-ocular distance on teacher landmarks (normaliser for NME)
    left_eye  = t_lmk[36:42, :2].mean(axis=0)
    right_eye = t_lmk[42:48, :2].mean(axis=0)
    iod       = np.linalg.norm(left_eye - right_eye)

    nme_vs_teacher = (diff_2d.mean() / max(iod, 1e-6)) * 100.0

    print(f"\n  Mean abs error     : {diff.mean():.3f} px  (all xyz)")
    print(f"  Mean 2D error      : {diff_2d.mean():.3f} px  (xy only)")
    print(f"  Max  2D error      : {diff_2d.max():.3f} px")
    print(f"  x-axis mean error  : {diff[:, 0].mean():.3f} px")
    print(f"  y-axis mean error  : {diff[:, 1].mean():.3f} px")
    print(f"  z-axis mean error  : {diff[:, 2].mean():.3f} px")
    print(f"  Inter-ocular dist  : {iod:.1f} px")
    print(f"  NME vs teacher     : {nme_vs_teacher:.3f}%")

    if diff_2d.mean() < 3.0:
        verdict = "✓ Excellent — student closely matches teacher"
    elif diff_2d.mean() < 8.0:
        verdict = "✓ Good — minor deviation from teacher"
    elif diff_2d.mean() < 15.0:
        verdict = "⚠ Moderate deviation — consider more training epochs"
    else:
        verdict = "✗ Large deviation — check training or preprocessing"
    print(f"\n  {verdict}")


# ══════════════════════════════════════════════════════════════════════════════
# Auto-find latest checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_best_model(checkpoints_root: str = 'checkpoints_landmarks') -> str:
    run_dirs = sorted(glob.glob(os.path.join(checkpoints_root, 'run_*')))
    if not run_dirs:
        raise FileNotFoundError(
            f"No run_* directories found under '{checkpoints_root}'.\n"
            f"Run 3_train_student.py first."
        )
    for run_dir in reversed(run_dirs):
        candidate = os.path.join(run_dir, 'best_model.pth')
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"No best_model.pth found in any run_* dir under '{checkpoints_root}'."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Export 1k3d68 student to ONNX')
    parser.add_argument('--checkpoint',       default=None,
                        help='Path to .pth checkpoint. Auto-finds latest if omitted.')
    parser.add_argument('--checkpoints_root', default='checkpoints_landmarks')
    parser.add_argument('--out',              default='my_1k3d68_student.onnx')
    parser.add_argument('--opset',            type=int, default=17)
    parser.add_argument('--no_simplify',      action='store_true')
    parser.add_argument('--teacher_onnx',     default='teacher_model/1k3d68.onnx')
    parser.add_argument('--test_image',       default=None)
    args = parser.parse_args()

    device = torch.device('cpu')   # always export on CPU for portability

    # ── Resolve checkpoint ────────────────────────────────────────────────
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        print(f"No --checkpoint given. Searching {args.checkpoints_root}...")
        checkpoint_path = find_latest_best_model(args.checkpoints_root)
        print(f"  Found: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("EXPORT — 1k3d68 Student → ONNX")
    print("=" * 60)

    model, meta = load_checkpoint(checkpoint_path, device)

    export_to_onnx(
        model    = model,
        out_path = args.out,
        opset    = args.opset,
        simplify = not args.no_simplify,
    )

    if Path(args.teacher_onnx).exists():
        compare_with_teacher(args.out, args.teacher_onnx, args.test_image)
    else:
        print(f"\n⚠  Teacher not found at '{args.teacher_onnx}' — skipping comparison.")

    print(f"\n{'=' * 60}")
    print(f"Export complete!")
    print(f"  Model  : StudentModel{meta['model_size'].capitalize()}")
    print(f"  Epoch  : {meta['epoch']}")
    if meta['nme'] is not None:
        print(f"  NME    : {meta['nme']:.3f}%")
    print(f"  Output : {args.out}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
