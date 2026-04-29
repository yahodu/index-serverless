# 1_inspect_teacher.py
"""
Step 1: Understand what 2d106det.onnx expects and produces.

CONFIRMED specs for 2d106det.onnx (from insightface/model_zoo/landmark.py):
  Input  : (N, 3, 192, 192)  float32
  Output : (N, 212)          flat float32

  Decode (from landmark.py — the else branch when output_shape[1] != 3309):
    pred = raw.reshape((-1, 2))       -> (106, 2)   landmarks
    pred[:, 0:2] += 1
    pred[:, 0:2] *= 96               -> xy in pixel space [0, 192]

  The model does NOT have internal MXNet-style normalisation (no bn_data/Sub/Mul
  in first 8 nodes for this PyTorch/ONNX export). Therefore:
    input_mean = 127.5
    input_std  = 128.0
    blob = (RGB - 127.5) / 128.0   NHWC -> NCHW

  This is identical to 1k3d68 preprocessing — same pipeline, different output.

  Model size : ~5 MB  (MobileNet-0.5, ~1.2 M params)
  Task       : 2D 106-point face alignment (static image)
               Covers eyes, brows, nose, lips, jaw contour — dense 2D coverage
"""

import os
import numpy as np
import onnx
import onnxruntime as ort


# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_SIZE  = 192
INPUT_MEAN  = 127.5
INPUT_STD   = 128.0
OUTPUT_FLAT = 212       # 106 * 2
NUM_LMK     = 106
LMK_DIM     = 2         # 2D landmarks (x, y) only — no z


def decode_landmarks(raw: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
    """
    Decode raw 2d106det output to pixel-space landmark coordinates.

    Mirrors insightface/model_zoo/landmark.py exactly for the 2D branch:

        pred = session.run(...)[0][0]      # (212,)
        pred = pred.reshape((-1, 2))       # (106, 2)
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (input_size // 2)  # * 96 => pixel space [0, 192]

    Input  : raw  (N, 212)   float32  — batch output from session.run
    Output : lmks (N, 106, 2) float32 — pixel-space landmarks
    """
    half  = input_size // 2    # 96
    batch = raw.shape[0]
    lmks  = np.empty((batch, NUM_LMK, LMK_DIM), dtype=np.float32)

    for i in range(batch):
        pred = raw[i]                          # (212,)
        pred = pred.reshape(-1, LMK_DIM)       # (106, 2)
        pred = pred.copy()
        pred[:, 0:2] += 1.0
        pred[:, 0:2] *= half
        lmks[i] = pred

    return lmks


def inspect_teacher(model_path: str):
    print("=" * 60)
    print(f"Inspecting: {model_path}")
    print("=" * 60)

    # ── ONNX graph metadata ───────────────────────────────────
    model = onnx.load(model_path)
    print(f"\nModel IR version : {model.ir_version}")
    print(f"Opset version    : {model.opset_import[0].version}")
    print(f"Producer         : {model.producer_name!r}")

    # Detect normalisation style — mirrors landmark.py logic exactly
    find_sub = find_mul = False
    for nid, node in enumerate(model.graph.node[:8]):
        if node.name.startswith('Sub') or node.name.startswith('_minus'):
            find_sub = True
        if node.name.startswith('Mul') or node.name.startswith('_mul'):
            find_mul = True
        if nid < 3 and node.name == 'bn_data':
            find_sub = find_mul = True

    if find_sub and find_mul:
        input_mean, input_std = 0.0, 1.0
        norm_note = "model has internal normalisation (MXNet export)"
    else:
        input_mean, input_std = INPUT_MEAN, INPUT_STD
        norm_note = "external normalisation required (standard PyTorch/ONNX export)"

    print(f"\nDetected normalisation : mean={input_mean}, std={input_std}")
    print(f"Note                   : {norm_note}")
    print(f"Preprocessing formula  : blob = (RGB - {INPUT_MEAN}) / {INPUT_STD}")
    print(f"  pixel   0  -> {(0   - INPUT_MEAN) / INPUT_STD:+.4f}")
    print(f"  pixel 127  -> {(127 - INPUT_MEAN) / INPUT_STD:+.4f}")
    print(f"  pixel 255  -> {(255 - INPUT_MEAN) / INPUT_STD:+.4f}")

    # ── Inference session ─────────────────────────────────────
    session = ort.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    print("\n--- INPUTS ---")
    for inp in session.get_inputs():
        print(f"  Name  : {inp.name}")
        print(f"  Shape : {inp.shape}")
        print(f"  Type  : {inp.type}")

    print("\n--- OUTPUTS ---")
    for out in session.get_outputs():
        print(f"  Name  : {out.name}")
        print(f"  Shape : {out.shape}")
        print(f"  Type  : {out.type}")

    out_shape = session.get_outputs()[0].shape
    out_size  = out_shape[1]

    assert out_size == OUTPUT_FLAT, (
        f"Expected output size {OUTPUT_FLAT}, got {out_size}. "
        f"Wrong model? (3309 → you have 1k3d68, not 2d106det)"
    )
    print(f"\n  Output breakdown: {out_size} = {NUM_LMK} landmarks × {LMK_DIM} dims (xy)")
    print(f"  Landmark scheme : 106 points covering eyes, brows, nose, lips, jaw")
    print(f"  Note            : ALL {NUM_LMK} rows are used (no intermediate rows)")

    # ── Test run ──────────────────────────────────────────────
    input_name = session.get_inputs()[0].name
    dummy_img  = np.random.randint(0, 256, (1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    dummy_blob = (dummy_img.astype(np.float32) - INPUT_MEAN) / INPUT_STD
    dummy_blob = dummy_blob.transpose(0, 3, 1, 2)   # NHWC -> NCHW

    raw  = session.run(None, {input_name: dummy_blob})[0]   # (1, 212)
    lmks = decode_landmarks(raw.copy())                      # (1, 106, 2)

    print(f"\n--- TEST RUN ---")
    print(f"  Input  shape   : {dummy_blob.shape}      <- (N, 3, 192, 192)")
    print(f"  Raw output     : {raw.shape}        <- (N, 212) flat")
    print(f"  Decoded shape  : {lmks.shape}   <- (N, 106, 2)")
    print(f"  x range        : [{lmks[0,:,0].min():.2f}, {lmks[0,:,0].max():.2f}]  (expect ~0–{INPUT_SIZE})")
    print(f"  y range        : [{lmks[0,:,1].min():.2f}, {lmks[0,:,1].max():.2f}]  (expect ~0–{INPUT_SIZE})")

    # ── Model size ────────────────────────────────────────────
    file_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n--- MODEL SIZE ---")
    print(f"  File size        : {file_mb:.1f} MB  (teacher is already small!)")
    print(f"  Backbone         : MobileNet-0.5  (~1.2 M params)")
    print(f"  Distillation goal: Compress to a tinier backbone OR")
    print(f"                     match accuracy with a different architecture")

    return session


if __name__ == "__main__":
    inspect_teacher("teacher_model/2d106det.onnx")
