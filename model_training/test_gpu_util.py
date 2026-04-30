import onnxruntime as ort
import numpy as np

model_path = "1k3d68.onnx"

session = ort.InferenceSession(
    model_path,
    providers=["CUDAExecutionProvider"]
)

# Dummy input (adjust shape based on your model)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Replace dynamic dims (None) with 1
input_shape = [dim if isinstance(dim, int) else 1 for dim in input_shape]

dummy_input = np.random.rand(*input_shape).astype(np.float32)

outputs = session.run(None, {input_name: dummy_input})

print("Ran inference successfully")
print("Execution providers:", session.get_providers())
