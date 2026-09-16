import torch
import onnx
import os

# Source .pt file
torchscript_model_path = "T1_walk.pt"
# Target .onnx file
onnx_output_path = "T1_walk.onnx"
# Input shape of .pt network
dummy_input_shape = (1, 62)

# Load model
script_model = torch.jit.load(torchscript_model_path)
script_model.eval()

# Prepare dummy Input
dummy_input = torch.randn(dummy_input_shape)

# Export onnx
torch.onnx.export(
    script_model,                         # Model
    dummy_input,                          # Dummy input
    onnx_output_path,                     # Target path
    export_params=True,
    opset_version=11,                     # Compatible with onnxruntime 1.10.0
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
