# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1_inspect_teacher_scrfd.py                                                  ║
# ║                                                                              ║
# ║  CONFIRMED specs for scrfd_10g_bnkps.onnx (from insightface scrfd.py        ║
# ║  and scrfd2onnx.py):                                                         ║
# ║                                                                              ║
# ║  Input  : name='input.1'  shape=(1, 3, 640, 640)  float32                   ║
# ║           Preprocessing: cv2.dnn.blobFromImage swapRB=True                  ║
# ║           => blob = (RGB - 127.5) / 128.0,  NHWC->NCHW                      ║
# ║                                                                              ║
# ║  Outputs (9 tensors — fmc=3 strides × 3 head types):                        ║
# ║    Index  Name       Shape@640     Meaning                                   ║
# ║    0      score_8    (12800, 1)    cls scores  stride-8  (80×80 × 2 anchors) ║
# ║    1      score_16   ( 3200, 1)    cls scores  stride-16 (40×40 × 2 anchors) ║
# ║    2      score_32   (  800, 1)    cls scores  stride-32 (20×20 × 2 anchors) ║
# ║    3      bbox_8     (12800, 4)    ltrb deltas stride-8  (raw, ÷stride later)║
# ║    4      bbox_16    ( 3200, 4)    ltrb deltas stride-16                     ║
# ║    5      bbox_32    (  800, 4)    ltrb deltas stride-32                     ║
# ║    6      kps_8      (12800, 10)   kps offsets stride-8  (5pts × xy)         ║
# ║    7      kps_16     ( 3200, 10)   kps offsets stride-16                     ║
# ║    8      kps_32     (  800, 10)   kps offsets stride-32                     ║
# ║                                                                              ║
# ║  IMPORTANT: bbox_* and kps_* are RAW (not multiplied by stride).             ║
# ║  InsightFace's scrfd.py multiplies them by stride at decode time:            ║
# ║      bbox_preds = net_outs[idx + fmc] * stride                               ║
# ║      kps_preds  = net_outs[idx + fmc*2] * stride                             ║
# ║  The student must reproduce the same raw (pre-stride) values.                ║
# ║                                                                              ║
# ║  _num_anchors = 2  → anchor centers are REPEATED twice per grid cell         ║
# ║  _feat_stride_fpn = [8, 16, 32]                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os
import numpy as np
import onnx
import onnxruntime as ort
import cv2

INPUT_SIZE   = 640
INPUT_MEAN   = 127.5
INPUT_STD    = 128.0
FMC          = 3
STRIDES      = [8, 16, 32]
NUM_ANCHORS  = 2

# Expected output shapes for a 640×640 input
EXPECTED_OUTPUTS = [
    ("score_8",  (12800, 1)),
    ("score_16", ( 3200, 1)),
    ("score_32", (  800, 1)),
    ("bbox_8",   (12800, 4)),
    ("bbox_16",  ( 3200, 4)),
    ("bbox_32",  (  800, 4)),
    ("kps_8",    (12800, 10)),
    ("kps_16",   ( 3200, 10)),
    ("kps_32",   (  800, 10)),
]


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Mirrors cv2.dnn.blobFromImage(img, 1/128, (W,H), (127.5,127.5,127.5), swapRB=True).
    Input : (H, W, 3) BGR uint8
    Output: (1, 3, H, W) float32  RGB normalised
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - INPUT_MEAN) / INPUT_STD
    return img.transpose(2, 0, 1)[np.newaxis]   # NCHW


def letterbox(img_bgr: np.ndarray, size: int = INPUT_SIZE) -> tuple:
    """
    Aspect-ratio preserving resize + zero-pad (mirrors SCRFD detect() logic).
    Returns (padded_bgr, scale_factor).
    """
    h, w = img_bgr.shape[:2]
    scale  = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized  = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas   = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


def inspect_teacher(model_path: str):
    print("=" * 70)
    print(f"Inspecting SCRFD teacher : {model_path}")
    print("=" * 70)

    model = onnx.load(model_path)
    print(f"\nModel IR version : {model.ir_version}")
    print(f"Opset version    : {model.opset_import[0].version}")
    print(f"Producer         : {model.producer_name!r}")

    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    print("\n--- INPUT ---")
    for inp in session.get_inputs():
        print(f"  Name  : {inp.name}")
        print(f"  Shape : {inp.shape}")
        print(f"  Type  : {inp.type}")

    print("\n--- OUTPUTS ---")
    outputs = session.get_outputs()
    for o in outputs:
        print(f"  Name  : {o.name:12s}  Shape : {o.shape}")

    n_outputs = len(outputs)
    assert n_outputs == 9, f"Expected 9 outputs (scrfd_bnkps), got {n_outputs}"
    print(f"\n  9 outputs confirmed: fmc={FMC}, strides={STRIDES}, "
          f"num_anchors={NUM_ANCHORS}, use_kps=True")

    # ── Test run ──────────────────────────────────────────────
    dummy_bgr  = np.random.randint(0, 256, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    blob       = preprocess_image(dummy_bgr)
    input_name = session.get_inputs()[0].name
    raw_outs   = session.run(None, {input_name: blob})

    print("\n--- TEST RUN (random 640×640 input) ---")
    for (name, expected_shape), arr in zip(EXPECTED_OUTPUTS, raw_outs):
        ok = "✓" if arr.shape == expected_shape else "✗"
        print(f"  {ok} {name:10s}  got={arr.shape}  expected={expected_shape}  "
              f"range=[{arr.min():.3f}, {arr.max():.3f}]")

    file_mb = os.path.getsize(model_path) / (1024 ** 2)
    print(f"\n--- MODEL SIZE ---")
    print(f"  File size  : {file_mb:.1f} MB")
    print(f"  Backbone   : Basic ResNet  (~3.86 M params without KPS head, "
          f"~4.23 M with KPS)")
    print(f"  Target     : Student with MobileNetV2 backbone + lite FPN neck")

    return session


if __name__ == "__main__":
    inspect_teacher("teacher_model/scrfd_10g_bnkps.onnx")
