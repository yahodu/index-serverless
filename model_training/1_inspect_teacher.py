# 1_inspect_teacher.py
"""
Step 1: Understand what the teacher model expects and produces.
This is like carefully studying the painting before you start drawing.
"""

import onnxruntime as ort
import onnx
import numpy as np

def inspect_teacher(model_path: str):
    print("=" * 60)
    print(f"Inspecting: {model_path}")
    print("=" * 60)

    # Load the ONNX model graph
    model = onnx.load(model_path)
    print(f"\nModel IR version: {model.ir_version}")
    print(f"Opset version: {model.opset_import[0].version}")
    print(f"Producer: {model.producer_name}")

    # Create inference session
    session = ort.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    print("\n--- INPUTS ---")
    for inp in session.get_inputs():
        print(f"  Name: {inp.name}")
        print(f"  Shape: {inp.shape}")
        print(f"  Type: {inp.type}")

    print("\n--- OUTPUTS ---")
    for out in session.get_outputs():
        print(f"  Name: {out.name}")
        print(f"  Shape: {out.shape}")
        print(f"  Type: {out.type}")

    # Test with dummy input
    # AntelopeV2 face recognition: input is (1, 3, 112, 112), RGB, normalized
    dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: dummy_input})

    print(f"\n--- TEST RUN ---")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output[0].shape}")
    print(f"  Output dtype: {output[0].dtype}")
    print(f"  Output range: [{output[0].min():.4f}, {output[0].max():.4f}]")
    print(f"  Output norm: {np.linalg.norm(output[0], axis=1)}")

    # Count parameters (approximate from model size)
    import os
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    approx_params = file_size_mb / 4 * 1e6  # rough: 4 bytes per float32 param
    print(f"\n--- MODEL SIZE ---")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Approx parameters: {approx_params / 1e6:.1f}M")

    return session

if __name__ == "__main__":
    session = inspect_teacher("teacher_model/glintr100.onnx")
