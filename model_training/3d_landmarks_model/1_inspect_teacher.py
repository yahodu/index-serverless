# 1_inspect_teacher.py
"""
Step 1: Understand what the teacher model expects and produces.

Confirmed specs for 1k3d68.onnx (InsightFace antelopev2 / buffalo_l):
  Input  : (N, 3, 192, 192)  float32  RGB  normalised by mean=127.5, std=128.0
  Output : (N, 3309)         float32  flat vector
  Decode : reshape -> (68, 3), then:
             xy  = (xy  + 1) * 96   [pixel coords in 192x192 space]
             z   = z        * 96   [relative depth, same scale]
  Source : insightface/python-package/insightface/model_zoo/landmark.py
"""

import os
import numpy as np
import onnx
import onnxruntime as ort


def decode_landmarks(raw: np.ndarray, input_size: int = 192) -> np.ndarray:
    """
    Decode raw model output to landmark coordinates.

    Input  : raw  (N, 3309)  float32
    Output : lmks (N, 68, 3) float32  — pixel-space xy + scaled z
    """
    half = input_size // 2          # 96
    lmks = raw.reshape(-1, 68, 3)   # (N, 68, 3)
    lmks[:, :, 0:2] += 1.0
    lmks[:, :, 0:2] *= half         # xy -> [0, 192]
    lmks[:, :, 2]   *= half         # z  -> same scale
    return lmks


def inspect_teacher(model_path: str):
    print("=" * 60)
    print(f"Inspecting: {model_path}")
    print("=" * 60)

    # ── ONNX graph metadata ───────────────────────────────────
    model = onnx.load(model_path)
    print(f"\nModel IR version : {model.ir_version}")
    print(f"Opset version    : {model.opset_import[0].version}")
    print(f"Producer         : {model.producer_name}")

    # Detect normalisation style from graph (mirrors landmark.py logic)
    find_sub = find_mul = False
    for nid, node in enumerate(model.graph.node[:8]):
        if node.name.startswith('Sub') or node.name.startswith('_minus'):
            find_sub = True
        if node.name.startswith('Mul') or node.name.startswith('_mul'):
            find_mul = True
        if nid < 3 and node.name == 'bn_data':
            find_sub = find_mul = True

    input_mean = 0.0   if (find_sub and find_mul) else 127.5
    input_std  = 1.0   if (find_sub and find_mul) else 128.0
    print(f"\nDetected normalisation: mean={input_mean}, std={input_std}")

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

    # ── Test run ──────────────────────────────────────────────
    input_name  = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape          # [N, 3, 192, 192]
    H = input_shape[2] if isinstance(input_shape[2], int) else 192
    W = input_shape[3] if isinstance(input_shape[3], int) else 192

    dummy = np.random.randn(1, 3, H, W).astype(np.float32)
    raw   = session.run(None, {input_name: dummy})[0]    # (1, 3309)

    print(f"\n--- TEST RUN (dummy input) ---")
    print(f"  Input  shape : {dummy.shape}")
    print(f"  Output shape : {raw.shape}   <- raw flat vector")

    assert raw.shape[1] == 3309, (
        f"Expected output size 3309 (68*3*~), got {raw.shape[1]}. "
        f"Inspect the model manually."
    )

    lmks = decode_landmarks(raw.copy(), input_size=H)
    print(f"  Decoded shape: {lmks.shape}  <- (N, 68, 3)")
    print(f"  x range      : [{lmks[0,:,0].min():.2f}, {lmks[0,:,0].max():.2f}]  (expect ~0–{H})")
    print(f"  y range      : [{lmks[0,:,1].min():.2f}, {lmks[0,:,1].max():.2f}]  (expect ~0–{H})")
    print(f"  z range      : [{lmks[0,:,2].min():.4f}, {lmks[0,:,2].max():.4f}]")

    # ── Model size ────────────────────────────────────────────
    file_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n--- MODEL SIZE ---")
    print(f"  File size          : {file_mb:.1f} MB")
    print(f"  Backbone           : ResNet-50")
    print(f"  Published params   : 34.2 M")

    print(f"\n--- NORMALISATION SUMMARY ---")
    print(f"  blob = (RGB_image - {input_mean}) / {input_std}")
    print(f"  i.e. pixel 0   -> {(0   - input_mean) / input_std:.4f}")
    print(f"       pixel 127 -> {(127 - input_mean) / input_std:.4f}")
    print(f"       pixel 255 -> {(255 - input_mean) / input_std:.4f}")

    return session


if __name__ == "__main__":
    inspect_teacher("teacher_model/1k3d68.onnx")
