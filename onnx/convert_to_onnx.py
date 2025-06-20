import os
import sys
import torch
import json
import yaml

# Add the parent directory to Python path to import src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.segmentation.models.unet import SegmentationModels

# === CONFIG ===
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
ARCH = "fpn"
DEVICE = "cpu"  # or "cuda" if you want

# === Read model dimensions from config YAML ===
CONFIG_PATH = '../configs/config.yaml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
resize = config['model']['input_size']
TARGET_HEIGHT, TARGET_WIDTH = resize
ONNX_MODEL_NAME = f"unet_resnet34_{TARGET_HEIGHT}x{TARGET_WIDTH}_fixed.onnx"

# === Load Lightning model checkpoint path from config ===
ckpt_path = config['inference']['model_path']

# === Load LightningModule ===
"""
Loads the PyTorch Lightning model from checkpoint and extracts the pure torch.nn.Module for ONNX export.
"""
lightning_model = SegmentationModels.load_from_checkpoint(
    ckpt_path,
    arch=ARCH,
    encoder_name=ENCODER,
    in_channels=3,
    out_classes=1,
    activation="sigmoid"
)
lightning_model.eval()

# === Extract pure torch model ===
"""
Get the underlying torch.nn.Module from the Lightning model for ONNX export.
"""
torch_model = lightning_model.model
torch_model.eval()
torch_model.to(DEVICE)

# === Prepare dummy input ===
"""
Create a dummy input tensor with the correct shape for ONNX export.
Batch size is dynamic, height and width are fixed from config.
"""
dummy_input = torch.randn(1, 3, TARGET_HEIGHT, TARGET_WIDTH, device=DEVICE)

# Only batch dimension is dynamic for flexibility.
dynamic_axes = {
     "input": {0: "batch_size"},
     "output": {0: "batch_size"},
}

# === Export to ONNX ===
"""
Export the torch model to ONNX format with fixed height/width and dynamic batch size.
"""
torch.onnx.export(
    torch_model,           # Only the pure torch model!
    dummy_input,
    ONNX_MODEL_NAME,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=dynamic_axes,
)

print(f"✅ Model with fixed size ({TARGET_HEIGHT}x{TARGET_WIDTH}) exported as {ONNX_MODEL_NAME}")
