# 4_export_onnx.py
"""
Step 4: Export trained 2d106det student model to ONNX.

Teacher output decode (official InsightFace landmark.py — 2D branch):
  raw shape : (1, 212)
  reshape   : (106, 2)
  xy decode : (xy + 1) * 96

Usage:
    # Standalone export (output: (batch, 106, 2) pixel coords)
    python 4_export_onnx_2d106.py

    # InsightFace drop-in replacement (output: (batch, 212) raw logits)
    python 4_export_onnx_2d106.py --teacher_compatible --out my_2d106det_student_insightface.onnx

    # With specific checkpoint
    python 4_export_onnx_2d106.py --teacher_compatible \\
        --checkpoint checkpoints_2d106/run_.../best_model.pth

    # With teacher comparison using a real face image
    python 4_export_onnx_2d106.py --teacher_compatible \\
        --teacher_onnx teacher_model/2d106det.onnx --test_image face.jpg

    # Skip onnxsim simplification
    python 4_export_onnx_2d106.py --teacher_compatible --no_simplify
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

from train_student import (
    StudentModelUltra,
    StudentModelSmall,
    StudentModelMedium,
    INPUT_SIZE,
    NUM_LMK,
    LMK_DIM,
)

# ── Constants ─────────────────────────────────────────────────────────────────
INPUT_MEAN       = 127.5
INPUT_STD        = 128.0
MINIMUM_OPSET    = 18
TEACHER_RAW_SIZE = 212    # 106 * 2

MODEL_MAP = {
    'ultra' : StudentModelUltra,
    'small' : StudentModelSmall,
    'medium': StudentModelMedium,
}


# ══════════════════════════════════════════════════════════════════════════════
# Teacher-compatible wrapper
# ══════════════════════════════════════════════════════════════════════════════

class TeacherCompatibleWrapper2D(nn.Module):
    """
    Wraps the 2D student to produce output matching 2d106det.onnx format.

    Teacher output  : (batch, 212)  raw logits
    InsightFace decode (landmark.py — 2D branch):
        pred = out[0].reshape(-1, 2)          # (106, 2)
        pred[:, :2] = (pred[:, :2] + 1) * 96  # pixel coords

    This wrapper:
      1. Runs student forward pass  →  (B, 106, 2) pixel coords
      2. Inverts the InsightFace decode  →  (B, 106, 2) raw logits
             raw_xy = pixel_xy / 96 - 1
      3. Flattens to (B, 212) — exactly teacher format

    Output node name is set to match what InsightFace looks for.
    """

    def __init__(self, student: nn.Module, input_size: int = INPUT_SIZE):
        super().__init__()
        self.student   = student
        self.half_size = input_size // 2   # 96

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lmk = self.student(x)             # (B, 106, 2) pixel coords
        raw = lmk / self.half_size - 1.0  # invert: (xy+1)*96 -> xy
        return raw.reshape(x.shape[0], -1)  # (B, 212)


# ══════════════════════════════════════════════════════════════════════════════
# Teacher output decoder (official InsightFace logic — 2D branch)
# ══════════════════════════════════════════════════════════════════════════════

def decode_teacher_output(raw: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
    """
    Decode raw 2d106det.onnx output to pixel-space landmarks.

    landmark.py (2D else branch):
        pred = session.run(...)[0][0]      # (212,)
        pred = pred.reshape((-1, 2))       # (106, 2)
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (input_size // 2)  # * 96

    Args:
        raw       : (212,) float32 — single sample
        input_size: 192

    Returns:
        (106, 2) float32 in pixel coordinates
    """
    assert raw.shape == (TEACHER_RAW_SIZE,), (
        f"Expected shape ({TEACHER_RAW_SIZE},), got {raw.shape}.\n"
        f"Pass output[0][0], not output[0] or output."
    )
    pred = raw.reshape(-1, 2).copy()   # (106, 2)
    pred[:, 0:2] += 1
    pred[:, 0:2] *= (input_size // 2)
    return pred.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint loader
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(checkpoint_path: str, device: torch.device):
    print(f"Loading checkpoint : {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        raise KeyError(f"Cannot find model weights. Keys: {list(ckpt.keys())}")

    config     = ckpt.get('config', {})
    model_size = config.get('model_size', None)

    if model_size is None:
        # Infer from weight keys
        has_features = any(k.startswith('features.') for k in state_dict)
        n_params     = sum(v.numel() for v in state_dict.values())
        if n_params < 500_000:
            model_size = 'ultra'
        elif n_params < 2_000_000:
            model_size = 'small'
        else:
            model_size = 'medium'
        print(f"  model_size : '{model_size}'  (inferred, {n_params:,} params)")
    else:
        print(f"  model_size : '{model_size}'  (from config)")

    if model_size not in MODEL_MAP:
        raise ValueError(f"Unknown model_size='{model_size}'.")

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

    print(f"  Epoch      : {meta['epoch']}")
    if meta['nme'] is not None:
        print(f"  Best NME   : {meta['nme']:.3f}%")
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")

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

    if opset < MINIMUM_OPSET:
        raise ValueError(
            f"opset={opset} below minimum {MINIMUM_OPSET}. Use --opset {MINIMUM_OPSET}."
        )

    if teacher_compatible:
        export_model       = TeacherCompatibleWrapper2D(model)
        expected_shape     = (2, TEACHER_RAW_SIZE)
        output_node_name   = 'fc1'
        print("  Mode              : teacher-compatible (InsightFace drop-in)")
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

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model structurally valid")

    if simplify:
        try:
            import onnxsim
            print("\nSimplifying...")
            onnx_model, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(onnx_model, out_path)
                print("  ✓ Simplified and saved")
            else:
                print("  ⚠ Simplification failed — keeping original")
        except ImportError:
            print("  ⚠ onnxsim not installed (pip install onnx-simplifier)")

    # ── Shape check ───────────────────────────────────────────
    sess       = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    test_input = np.random.randn(2, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    ort_out    = sess.run(None, {sess.get_inputs()[0].name: test_input})[0]

    assert ort_out.shape == expected_shape, (
        f"Shape mismatch: got {ort_out.shape}, expected {expected_shape}"
    )
    print(f"  ✓ Output shape    : {ort_out.shape}")

    # ── PT vs ONNX numerical check ────────────────────────────
    with torch.no_grad():
        pt_out = export_model(torch.tensor(test_input)).numpy()
    max_diff = float(np.abs(ort_out - pt_out).max())
    print(f"  Max diff PT↔ONNX  : {max_diff:.2e}")
    assert max_diff < 1e-4, f"Output mismatch! max_diff={max_diff:.2e}"
    print("  ✓ PyTorch and ONNX outputs match")

    size_mb = os.path.getsize(out_path) / 1024 ** 2
    print(f"  File size         : {size_mb:.1f} MB  "
          f"(teacher is ~5 MB; student ultra ≈ 1–2 MB, small ≈ 4 MB)")


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
    print("Teacher vs Student comparison (2D 106-point)")
    print(f"{'─' * 60}")

    if image_path and Path(image_path).exists():
        img = cv2.imread(image_path)
        if img is None:
            img = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
            print("  Image : grey placeholder (could not read)")
        else:
            img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"  Image : {image_path}")
    else:
        img = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
        print("  Image : grey placeholder (pass --test_image <face.jpg>)")

    blob = img.astype(np.float32)
    blob = (blob - INPUT_MEAN) / INPUT_STD
    blob = blob.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 192, 192)

    # ── Teacher ───────────────────────────────────────────────
    t_sess  = ort.InferenceSession(teacher_onnx, providers=['CPUExecutionProvider'])
    t_iname = t_sess.get_inputs()[0].name
    t_raw   = t_sess.run(None, {t_iname: blob})[0][0]   # (212,)
    print(f"\n  Teacher raw output : shape={t_raw.shape}  "
          f"min={t_raw.min():.3f}  max={t_raw.max():.3f}")
    t_lmk = decode_teacher_output(t_raw)   # (106, 2)

    # ── Student ───────────────────────────────────────────────
    s_sess  = ort.InferenceSession(student_onnx, providers=['CPUExecutionProvider'])
    s_iname = s_sess.get_inputs()[0].name
    s_raw   = s_sess.run(None, {s_iname: blob})[0][0]

    if teacher_compatible:
        print(f"\n  Student raw output : shape={s_raw.shape}  "
              f"min={s_raw.min():.3f}  max={s_raw.max():.3f}")
        s_lmk = decode_teacher_output(s_raw)   # (106, 2)
    else:
        s_lmk = s_raw   # (106, 2) already decoded

    print(f"  Teacher decoded    : shape={t_lmk.shape}  "
          f"xy_range=[{t_lmk.min():.1f}, {t_lmk.max():.1f}]")
    print(f"  Student decoded    : shape={s_lmk.shape}  "
          f"xy_range=[{s_lmk.min():.1f}, {s_lmk.max():.1f}]")

    # ── Metrics ───────────────────────────────────────────────
    diff    = np.abs(t_lmk - s_lmk)
    diff_2d = np.linalg.norm(t_lmk - s_lmk, axis=1)

    # IOD using eye centres (106-pt scheme)
    left_eye  = t_lmk[60:65, :].mean(axis=0)
    right_eye = t_lmk[68:73, :].mean(axis=0)
    iod       = np.linalg.norm(left_eye - right_eye)

    nme_vs_teacher = (diff_2d.mean() / max(iod, 1e-6)) * 100.0

    print(f"\n  Mean abs error     : {diff.mean():.3f} px  (all xy)")
    print(f"  Mean 2D error      : {diff_2d.mean():.3f} px")
    print(f"  Max  2D error      : {diff_2d.max():.3f} px")
    print(f"  x-axis mean error  : {diff[:, 0].mean():.3f} px")
    print(f"  y-axis mean error  : {diff[:, 1].mean():.3f} px")
    print(f"  Inter-ocular dist  : {iod:.1f} px")
    print(f"  NME vs teacher     : {nme_vs_teacher:.3f}%")

    if diff_2d.mean() < 2.0:
        verdict = "✓ Excellent — student closely matches teacher"
    elif diff_2d.mean() < 5.0:
        verdict = "✓ Good — minor deviation from teacher"
    elif diff_2d.mean() < 10.0:
        verdict = "⚠ Moderate — consider more training epochs"
    else:
        verdict = "✗ Large deviation — check training or preprocessing"
    print(f"\n  {verdict}")

    if teacher_compatible:
        print(f"\n{'─' * 60}")
        print("Round-trip encode→decode consistency")
        s_raw_rt    = s_sess.run(None, {s_iname: blob})[0][0]
        s_lmk_rt    = decode_teacher_output(s_raw_rt)
        rt_max_diff = float(np.abs(s_lmk - s_lmk_rt).max())
        print(f"  Encode→decode max error : {rt_max_diff:.2e} px")
        if rt_max_diff < 1e-3:
            print("  ✓ Round-trip numerically consistent")
        else:
            print("  ⚠ Round-trip error large — check TeacherCompatibleWrapper2D")


# ══════════════════════════════════════════════════════════════════════════════
# InsightFace usage
# ══════════════════════════════════════════════════════════════════════════════

def print_insightface_usage(out_path: str) -> None:
    print(f"""
{'═' * 60}
How to use in InsightFace (2d106det drop-in)
{'═' * 60}

  Option 1 — Automatic model-zoo swap (recommended):

      import insightface
      from insightface.app import FaceAnalysis

      app = FaceAnalysis()
      app.prepare(ctx_id=0)

      # Replace the 2D landmark model with your student
      app.models['landmark_2d_106'] = insightface.model_zoo.get_model(
          '{out_path}'
      )

      img   = cv2.imread('photo.jpg')
      faces = app.get(img)
      for face in faces:
          lmk2d = face.landmark_2d_106   # (106, 2) as usual

  Option 2 — Direct copy (simplest):

      cp {out_path} ~/.insightface/models/buffalo_l/2d106det.onnx

      # Backup original first:
      #   cp ~/.insightface/models/buffalo_l/2d106det.onnx \\
      #      ~/.insightface/models/buffalo_l/2d106det.onnx.bak
{'═' * 60}
""")


# ══════════════════════════════════════════════════════════════════════════════
# Auto-find latest checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_best_model(checkpoints_root: str = 'checkpoints_2d106') -> str:
    run_dirs = sorted(glob.glob(os.path.join(checkpoints_root, 'run_*')))
    if not run_dirs:
        raise FileNotFoundError(
            f"No run_* directories found under '{checkpoints_root}'.\n"
            f"Run 3_train_student_2d106.py first."
        )
    for run_dir in reversed(run_dirs):
        candidate = os.path.join(run_dir, 'best_model.pth')
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"No best_model.pth found under '{checkpoints_root}'."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Export 2d106det student to ONNX',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--checkpoint',       default=None)
    parser.add_argument('--checkpoints_root', default='checkpoints_2d106')
    parser.add_argument('--out',              default=None)
    parser.add_argument('--opset',            type=int, default=18)
    parser.add_argument('--no_simplify',      action='store_true')
    parser.add_argument(
        '--teacher_compatible', action='store_true',
        help=(
            'Export in teacher format:\n'
            f"  output node name : 'fc1'\n"
            f'  output shape     : (batch, {TEACHER_RAW_SIZE})  raw logits\n'
            'Drop-in replacement for 2d106det.onnx inside InsightFace.'
        ),
    )
    parser.add_argument('--teacher_onnx', default='teacher_model/2d106det.onnx')
    parser.add_argument('--test_image',   default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = (
            'my_2d106det_student_insightface.onnx'
            if args.teacher_compatible
            else 'my_2d106det_student.onnx'
        )

    device = torch.device('cpu')

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
    print("EXPORT — 2d106det Student → ONNX")
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

    print(f"\n{'=' * 60}")
    print(f"Export complete!")
    print(f"  Model  : StudentModel2D_{meta['model_size'].capitalize()}")
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
