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
    # Standalone export (output shape: (batch, 68, 3) pixel coords)
    python 4_export_onnx.py

    # InsightFace drop-in replacement (output shape: (batch, 3309) raw logits)
    python 4_export_onnx.py --teacher_compatible --out my_1k3d68_student_insightface.onnx

    # With specific checkpoint
    python 4_export_onnx.py --teacher_compatible --checkpoint checkpoints_landmarks/run_.../best_model.pth

    # With teacher comparison using a real face image
    python 4_export_onnx.py --teacher_compatible --teacher_onnx teacher_model/1k3d68.onnx --test_image face.jpg

    # Skip onnxsim simplification
    python 4_export_onnx.py --teacher_compatible --no_simplify
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
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
INPUT_MEAN    = 127.5
INPUT_STD     = 128.0
MINIMUM_OPSET = 18        # PyTorch >= 2.4 generates opset 18 natively
TEACHER_RAW_SIZE = 3309   # (1103, 3) → last 68 rows = landmarks

MODEL_MAP = {
    'small' : StudentModelSmall,
    'medium': StudentModelMedium,
    'large' : StudentModelLarge,
}


# ══════════════════════════════════════════════════════════════════════════════
# Teacher-compatible wrapper
# ══════════════════════════════════════════════════════════════════════════════

class TeacherCompatibleWrapper(nn.Module):
    """
    Wraps the student model to produce output in the same format as the
    teacher 1k3d68.onnx so it is a drop-in replacement inside InsightFace.

    Teacher output  : (batch, 3309)  raw logits
    InsightFace decode (landmark.py):
        pred = out[0].reshape(-1, 3)      # (1103, 3)
        pred = pred[-68:]                 # (68, 3)   ← last 68 rows
        pred[:, :2] = (pred[:, :2] + 1) * 96   # xy → pixels
        pred[:,  2] =  pred[:,  2]       * 96   # z  → pixels

    This wrapper:
      1. Runs the student forward pass  →  (B, 68, 3) pixel coords
      2. Inverts the InsightFace decode →  (B, 68, 3) raw logits
             raw_xy = pixel_xy / 96 - 1
             raw_z  = pixel_z  / 96
      3. Flattens to (B, 204)
      4. Pads the front with zeros to (B, 3309)
         InsightFace discards the first 3105 values (pred[:-68]) so they
         can safely be zero.

    Output node name is 'fc1' — the name InsightFace looks for.
    """

    def __init__(self, student: nn.Module, input_size: int = INPUT_SIZE):
        super().__init__()
        self.student   = student
        self.half_size = input_size // 2   # 96

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Student forward  →  pixel coords ──────────────────────────────
        lmk = self.student(x)              # (B, 68, 3)

        # ── Invert InsightFace decode  →  raw logits ───────────────────────
        raw = lmk.clone()
        raw[:, :, :2] = lmk[:, :, :2] / self.half_size - 1.0   # xy
        raw[:, :,  2] = lmk[:, :,  2] / self.half_size          # z

        # ── Flatten landmarks  →  (B, 204) ────────────────────────────────
        B   = raw.shape[0]
        raw = raw.reshape(B, -1)           # (B, 204)

        # ── Pad front to match teacher's (B, 3309) ────────────────────────
        pad = torch.zeros(B, TEACHER_RAW_SIZE - raw.shape[1],
                          dtype=raw.dtype, device=raw.device)
        return torch.cat([pad, raw], dim=1)   # (B, 3309)


# ══════════════════════════════════════════════════════════════════════════════
# Teacher output decoder  (official InsightFace logic)
# ══════════════════════════════════════════════════════════════════════════════

def decode_teacher_output(raw: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
    """
    Decode raw 1k3d68.onnx output to pixel-space landmarks.

    Official logic from insightface/model_zoo/landmark.py:
        pred = session.run(...)[0][0]          # shape (3309,)
        pred = pred.reshape((-1, 3))           # (1103, 3)
        pred = pred[-68:, :]                   # (68, 3)  ← LAST 68 rows
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (input_size // 2)
        pred[:, 2]   *= (input_size // 2)

    Args:
        raw       : (3309,) flat float32 — single sample, batch dim removed
        input_size: teacher input resolution (192)

    Returns:
        (68, 3) float32 landmarks in pixel coordinates
    """
    assert raw.shape == (TEACHER_RAW_SIZE,), (
        f"Expected shape ({TEACHER_RAW_SIZE},), got {raw.shape}.\n"
        f"Pass output[0][0], not output[0] or output."
    )
    pred = raw.reshape(-1, 3)           # (1103, 3)
    pred = pred[-NUM_LMK:, :].copy()   # (68,   3)
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
    Handles 'model_state_dict' key written by 3_train_student.py.
    """
    print(f"Loading checkpoint : {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ── Resolve state dict ────────────────────────────────────────────────
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        raise KeyError(
            f"Cannot find model weights in checkpoint.\n"
            f"Keys present: {list(ckpt.keys())}"
        )

    # ── Resolve model size ────────────────────────────────────────────────
    config     = ckpt.get('config', {})
    model_size = config.get('model_size', None)

    if model_size is None:
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
        raise ValueError(
            f"Unknown model_size='{model_size}'. Choose from {list(MODEL_MAP.keys())}"
        )

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
    model              : nn.Module,
    out_path           : str,
    opset              : int  = 18,
    simplify           : bool = True,
    teacher_compatible : bool = False,
) -> None:

    # ── Opset guard ───────────────────────────────────────────────────────
    if opset < MINIMUM_OPSET:
        raise ValueError(
            f"opset={opset} is below the minimum supported value ({MINIMUM_OPSET}).\n"
            f"Your installed PyTorch generates opset {MINIMUM_OPSET} natively.\n"
            f"Requesting a lower opset triggers a known conversion failure:\n"
            f"  'Assertion node->hasAttribute(kaxes) failed'\n"
            f"Fix: pass --opset {MINIMUM_OPSET} (or remove --opset entirely)."
        )

    # ── Choose export model and metadata ──────────────────────────────────
    if teacher_compatible:
        export_model       = TeacherCompatibleWrapper(model)
        expected_shape     = (2, TEACHER_RAW_SIZE)
        output_node_name   = 'fc1'        # name InsightFace landmark.py expects
        print("  Mode              : teacher-compatible")
        print(f"  Output node name  : '{output_node_name}'")
        print(f"  Output shape      : (batch, {TEACHER_RAW_SIZE})  raw logits")
    else:
        export_model       = model
        expected_shape     = (2, NUM_LMK, LMK_DIM)
        output_node_name   = 'landmarks'
        print("  Mode              : standalone")
        print(f"  Output node name  : '{output_node_name}'")
        print(f"  Output shape      : (batch, {NUM_LMK}, {LMK_DIM})  pixel coords")

    export_model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    print(f"\nExporting to : {out_path}  (opset {opset})")
    torch.onnx.export(
        export_model,
        dummy,
        out_path,
        opset_version       = opset,
        do_constant_folding = True,
        input_names         = ['input'],
        output_names        = [output_node_name],
        dynamic_axes        = {
            'input'          : {0: 'batch_size'},
            output_node_name : {0: 'batch_size'},
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
    ort_out    = sess.run(None, {sess.get_inputs()[0].name: test_input})[0]

    assert ort_out.shape == expected_shape, (
        f"Shape mismatch: got {ort_out.shape}, expected {expected_shape}"
    )
    print(f"  ✓ Output shape    : {ort_out.shape}")

    # ── PyTorch vs ONNX numerical agreement ───────────────────────────────
    with torch.no_grad():
        pt_out = export_model(torch.tensor(test_input)).numpy()

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
    student_onnx       : str,
    teacher_onnx       : str,
    image_path         : str  = None,
    teacher_compatible : bool = False,
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
            print(f"  Image             : {image_path}")
    else:
        img = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
        print("  Image             : grey placeholder "
              "(pass --test_image <face.jpg> for real results)")

    # Preprocess
    blob = img.astype(np.float32)
    blob = (blob - INPUT_MEAN) / INPUT_STD
    blob = blob.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 192, 192)

    # ── Run teacher ───────────────────────────────────────────────────────
    t_sess  = ort.InferenceSession(teacher_onnx, providers=['CPUExecutionProvider'])
    t_iname = t_sess.get_inputs()[0].name
    t_raw   = t_sess.run(None, {t_iname: blob})[0][0]   # (3309,)

    print(f"\n  Teacher raw output : shape={t_raw.shape}  "
          f"min={t_raw.min():.3f}  max={t_raw.max():.3f}")

    t_lmk = decode_teacher_output(t_raw)   # (68, 3) pixel coords

    # ── Run student ───────────────────────────────────────────────────────
    s_sess  = ort.InferenceSession(student_onnx, providers=['CPUExecutionProvider'])
    s_iname = s_sess.get_inputs()[0].name
    s_raw   = s_sess.run(None, {s_iname: blob})[0][0]   # (3309,) or (68, 3)

    if teacher_compatible:
        # Student outputs raw logits — decode exactly as InsightFace does
        print(f"\n  Student raw output : shape={s_raw.shape}  "
              f"min={s_raw.min():.3f}  max={s_raw.max():.3f}")
        s_lmk = decode_teacher_output(s_raw)   # (68, 3) pixel coords
    else:
        # Student outputs pixel coords directly
        s_lmk = s_raw   # (68, 3)

    print(f"  Teacher decoded    : shape={t_lmk.shape}  "
          f"xy_range=[{t_lmk[:, :2].min():.1f}, {t_lmk[:, :2].max():.1f}]")
    print(f"  Student decoded    : shape={s_lmk.shape}  "
          f"xy_range=[{s_lmk[:, :2].min():.1f}, {s_lmk[:, :2].max():.1f}]")

    # ── Metrics ───────────────────────────────────────────────────────────
    diff    = np.abs(t_lmk - s_lmk)
    diff_2d = np.linalg.norm(t_lmk[:, :2] - s_lmk[:, :2], axis=1)

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

    # ── Round-trip check (teacher_compatible mode only) ───────────────────
    if teacher_compatible:
        print(f"\n{'─' * 60}")
        print("Round-trip encode→decode consistency check")
        print(f"{'─' * 60}")
        # Re-decode student raw output and compare to student pixel output
        # Max error should be near float32 epsilon (~1e-6 px) if inversion is correct
        s_raw_full  = s_sess.run(None, {s_iname: blob})[0][0]   # (3309,)
        s_lmk_rt    = decode_teacher_output(s_raw_full)          # (68, 3)
        rt_max_diff = float(np.abs(s_lmk - s_lmk_rt).max())
        print(f"  Encode→decode max error : {rt_max_diff:.2e} px")
        if rt_max_diff < 1e-3:
            print("  ✓ Round-trip is numerically consistent")
        else:
            print("  ⚠ Round-trip error is large — check TeacherCompatibleWrapper")


# ══════════════════════════════════════════════════════════════════════════════
# InsightFace usage instructions
# ══════════════════════════════════════════════════════════════════════════════

def print_insightface_usage(out_path: str) -> None:
    print(f"""
{'═' * 60}
How to use in InsightFace
{'═' * 60}

  Option 1 — Automatic model-zoo swap (recommended):

      import insightface
      from insightface.app import FaceAnalysis

      app = FaceAnalysis()
      app.prepare(ctx_id=0)   # or ctx_id=-1 for CPU

      # Replace the landmark model with your student
      app.models['landmark_3d_68'] = insightface.model_zoo.get_model(
          '{out_path}'
      )

      img    = cv2.imread('photo.jpg')
      faces  = app.get(img)
      for face in faces:
          lmk3d = face.landmark_3d_68   # (68, 3) as usual

  Option 2 — Direct copy (simplest):

      cp {out_path} ~/.insightface/models/buffalo_l/1k3d68.onnx

      # InsightFace will load your student automatically next time.
      # Make a backup of the original first:
      #   cp ~/.insightface/models/buffalo_l/1k3d68.onnx \\
      #      ~/.insightface/models/buffalo_l/1k3d68.onnx.bak
{'═' * 60}
""")


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
    parser = argparse.ArgumentParser(
        description='Export 1k3d68 student to ONNX',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Path to .pth checkpoint.\nAuto-finds latest best_model.pth if omitted.',
    )
    parser.add_argument(
        '--checkpoints_root',
        default='checkpoints_landmarks',
        help='Root directory to search when --checkpoint is omitted.',
    )
    parser.add_argument(
        '--out',
        default=None,
        help=(
            'Output .onnx file path.\n'
            'Defaults to my_1k3d68_student_insightface.onnx (--teacher_compatible)\n'
            '         or my_1k3d68_student.onnx             (standalone).'
        ),
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=18,
        help=f'ONNX opset version. Must be >= {MINIMUM_OPSET} with this PyTorch version.',
    )
    parser.add_argument(
        '--no_simplify',
        action='store_true',
        help='Skip onnx-simplifier step.',
    )
    parser.add_argument(
        '--teacher_compatible',
        action='store_true',
        help=(
            'Export in teacher format:\n'
            f"  output node name : 'fc1'\n"
            f'  output shape     : (batch, {TEACHER_RAW_SIZE})  raw logits\n'
            'Drop-in replacement for 1k3d68.onnx inside InsightFace.'
        ),
    )
    parser.add_argument(
        '--teacher_onnx',
        default='teacher_model/1k3d68.onnx',
        help='Path to teacher .onnx for output comparison.',
    )
    parser.add_argument(
        '--test_image',
        default=None,
        help='Path to a face image for teacher vs student comparison.',
    )
    args = parser.parse_args()

    # ── Resolve output path ───────────────────────────────────────────────
    if args.out is None:
        args.out = (
            'my_1k3d68_student_insightface.onnx'
            if args.teacher_compatible
            else 'my_1k3d68_student.onnx'
        )

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
        model              = model,
        out_path           = args.out,
        opset              = args.opset,
        simplify           = not args.no_simplify,
        teacher_compatible = args.teacher_compatible,
    )

    if Path(args.teacher_onnx).exists():
        compare_with_teacher(
            student_onnx       = args.out,
            teacher_onnx       = args.teacher_onnx,
            image_path         = args.test_image,
            teacher_compatible = args.teacher_compatible,
        )
    else:
        print(f"\n⚠  Teacher not found at '{args.teacher_onnx}' — skipping comparison.")
        print(f"   Pass --teacher_onnx <path> to enable it.")

    print(f"\n{'=' * 60}")
    print(f"Export complete!")
    print(f"  Model  : StudentModel{meta['model_size'].capitalize()}")
    print(f"  Mode   : {'teacher-compatible (InsightFace drop-in)' if args.teacher_compatible else 'standalone'}")
    print(f"  Epoch  : {meta['epoch']}")
    if meta['nme'] is not None:
        print(f"  NME    : {meta['nme']:.3f}%")
    print(f"  Output : {args.out}")
    print(f"{'=' * 60}")

    if args.teacher_compatible:
        print_insightface_usage(args.out)


if __name__ == '__main__':
    main()
