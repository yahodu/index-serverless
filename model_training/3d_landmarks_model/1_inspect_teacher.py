# 1_inspect_teacher.py
"""
Step 1: Understand what the teacher model expects and produces.

CONFIRMED specs for 1k3d68.onnx (from insightface/model_zoo/landmark.py):
  Input  : (N, 3, 192, 192)  float32
  Output : (N, 3309)         flat float32

  Decode:
    raw.reshape(-1, 3)         -> (1103, 3)   intermediate outputs
    take last 68 rows          -> (68,   3)   actual landmarks
    pred[:, 0:2] += 1
    pred[:, 0:2] *= 96         -> xy in pixel space [0, 192]
    pred[:, 2]   *= 96         -> z at same scale

  Normalisation detected at runtime from ONNX graph:
    This model has bn_data / Sub+Mul in first 8 nodes -> mean=0.0, std=1.0
    Which means the model does its own normalisation internally.
    Input should still be passed as (RGB - 127.5) / 128.0 via
    cv2.dnn.blobFromImage — confirmed by landmark.py line:
      blob = cv2.dnn.blobFromImage(aimg, 1.0/self.input_std, ...)
    When input_std=1.0 that becomes scale=1.0, mean=(0,0,0) — raw float.
    BUT the model was exported expecting (RGB-127.5)/128 pre-applied externally.
    We mirror exactly what landmark.py does.
"""

import os
import numpy as np
import onnx
import onnxruntime as ort


# ── Confirmed constants ────────────────────────────────────────────────────────
INPUT_SIZE  = 192
INPUT_MEAN  = 127.5
INPUT_STD   = 128.0
OUTPUT_FLAT = 3309      # 1103 * 3  (last 68 rows are landmarks)
NUM_LMK     = 68
LMK_DIM     = 3


def decode_landmarks(raw: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
    """
    Decode raw model output to landmark coordinates.

    Mirrors insightface/model_zoo/landmark.py exactly:

        pred = session.run(...)[0][0]      # (3309,)
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))   # (1103, 3)
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num * -1:, :]  # last 68 rows -> (68, 3)
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.input_size[0] // 2)   # * 96
        pred[:, 2]   *= (self.input_size[0] // 2)   # * 96

    Input  : raw  (N, 3309)  float32  — batch output from session.run
    Output : lmks (N, 68, 3) float32  — pixel-space landmarks
    """
    half  = input_size // 2   # 96
    batch = raw.shape[0]
    lmks  = np.empty((batch, NUM_LMK, LMK_DIM), dtype=np.float32)

    for i in range(batch):
        pred = raw[i]                          # (3309,)
        pred = pred.reshape(-1, LMK_DIM)       # (1103, 3)
        pred = pred[NUM_LMK * -1:, :].copy()  # last 68 rows -> (68, 3)
        pred[:, 0:2] += 1.0
        pred[:, 0:2] *= half
        pred[:, 2]   *= half
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
        norm_note = "external normalisation required"

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

    # Validate output size
    out_size = session.get_outputs()[0].shape[1]
    assert out_size == OUTPUT_FLAT, (
        f"Expected output size {OUTPUT_FLAT}, got {out_size}. "
        f"Wrong model?"
    )
    print(f"\n  Output breakdown: {out_size} = {out_size // LMK_DIM} rows × {LMK_DIM} dims")
    print(f"  Landmark rows used: last {NUM_LMK} of {out_size // LMK_DIM} "
          f"({out_size // LMK_DIM - NUM_LMK} intermediate rows discarded)")

    # ── Test run ──────────────────────────────────────────────
    input_name = session.get_inputs()[0].name
    # Simulate real preprocessing: (random uint8 image - 127.5) / 128.0
    dummy_img  = np.random.randint(0, 256, (1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    dummy_blob = (dummy_img.astype(np.float32) - INPUT_MEAN) / INPUT_STD
    dummy_blob = dummy_blob.transpose(0, 3, 1, 2)   # NHWC -> NCHW

    raw  = session.run(None, {input_name: dummy_blob})[0]   # (1, 3309)
    lmks = decode_landmarks(raw.copy())                      # (1, 68, 3)

    print(f"\n--- TEST RUN ---")
    print(f"  Input  shape   : {dummy_blob.shape}")
    print(f"  Raw output     : {raw.shape}   <- (N, 3309) flat")
    print(f"  Decoded shape  : {lmks.shape}  <- (N, 68, 3)")
    print(f"  x range        : [{lmks[0,:,0].min():.2f}, {lmks[0,:,0].max():.2f}]  (expect ~0–{INPUT_SIZE})")
    print(f"  y range        : [{lmks[0,:,1].min():.2f}, {lmks[0,:,1].max():.2f}]  (expect ~0–{INPUT_SIZE})")
    print(f"  z range        : [{lmks[0,:,2].min():.4f}, {lmks[0,:,2].max():.4f}]")

    # ── Model size ────────────────────────────────────────────
    file_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n--- MODEL SIZE ---")
    print(f"  File size        : {file_mb:.1f} MB")
    print(f"  Backbone         : ResNet-50  (~34.2 M params)")

    return session


if __name__ == "__main__":
    inspect_teacher("teacher_model/1k3d68.onnx")
