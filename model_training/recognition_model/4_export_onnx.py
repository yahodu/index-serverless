# 4_export_onnx.py
"""
Step 4: Export YOUR trained model to ONNX format.

This produces an ONNX file with YOUR weights — entirely your creation.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import json
import os
import sys

# Import model definitions
from train_student import StudentModelSmall, StudentModelMedium, StudentModelLarge


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 17,
    simplify: bool = True,
):
    """Export a trained student model to ONNX format."""

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']

    # Recreate model
    model_size = config['model_size']
    embedding_dim = config['embedding_dim']

    if model_size == 'small':
        model = StudentModelSmall(embedding_dim)
    elif model_size == 'medium':
        model = StudentModelMedium(embedding_dim)
    elif model_size == 'large':
        model = StudentModelLarge(embedding_dim)
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Dummy input (same format as teacher: 1 x 3 x 112 x 112)
    dummy_input = torch.randn(1, 3, 112, 112)

    # Export
    print(f"Exporting to: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'},
        },
        do_constant_folding=True,
    )

    # Verify the exported model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model is valid")

    # Optionally simplify (reduces redundant operations)
    if simplify:
        try:
            import onnxsim
            print("Simplifying ONNX model...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(onnx_model, output_path)
            print("  ✓ Model simplified")
        except ImportError:
            print("  ⚠ onnxsim not installed, skipping simplification")
            print("    Install with: pip install onnx-simplifier")

    # Test inference
    print("\nTesting ONNX inference...")
    session = ort.InferenceSession(
        output_path,
        providers=['CPUExecutionProvider']
    )

    test_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
    output = session.run(None, {'input': test_input})[0]

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Output norm: {np.linalg.norm(output):.4f}")

    # Model size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Model file size: {file_size_mb:.1f} MB")

    # Compare with PyTorch output
    with torch.no_grad():
        pt_output = model(torch.tensor(test_input)).numpy()
    diff = np.abs(output - pt_output).max()
    print(f"  Max diff (ONNX vs PyTorch): {diff:.8f}")
    assert diff < 1e-4, f"Output mismatch! Max diff: {diff}"
    print("  ✓ ONNX output matches PyTorch output")

    print(f"\n{'='*60}")
    print(f"YOUR MODEL has been exported to: {output_path}")
    print(f"This model is YOURS — trained with your own weights!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Find the best checkpoint
    checkpoint_dir = "checkpoints"
    runs = sorted(os.listdir(checkpoint_dir))
    if not runs:
        print("No training runs found. Run 3_train_student.py first.")
        sys.exit(1)

    latest_run = os.path.join(checkpoint_dir, runs[-1])
    best_model = os.path.join(latest_run, "best_model.pth")

    if not os.path.exists(best_model):
        print(f"Best model not found at {best_model}")
        sys.exit(1)

    export_to_onnx(
        checkpoint_path=best_model,
        output_path="my_face_recognition.onnx",
        opset_version=17,
        simplify=True,
    )
