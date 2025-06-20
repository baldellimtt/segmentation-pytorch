import os
import glob
from PIL import Image
import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from torchvision import transforms
import yaml
import argparse

def preprocess_image(image_path, resize_size):
    """
    Preprocess an image for ONNX inference.
    Steps:
      - Open and convert to RGB
      - ToTensor
      - Normalize with ImageNet mean/std
      - Resize to the given size
    Args:
        image_path (str): Path to the input image
        resize_size (tuple): (height, width) for resizing
    Returns:
        np.ndarray: Preprocessed image as numpy array, shape [1, C, H, W]
    """
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.Resize(resize_size)
    ])
    tensor = transform(image).unsqueeze(0)  # [1, C, H, W]
    return tensor.numpy()

def postprocess_output(output, resize_size):
    """
    Postprocess ONNX model output to obtain a binary mask image.
    Steps:
      - (Optional) Sigmoid activation (if not present in model)
      - Interpolate output to resize_size
      - Threshold at 0.5 and convert to uint8 mask
    Args:
        output (np.ndarray): Raw ONNX model output
        resize_size (tuple): (height, width) for resizing
    Returns:
        PIL.Image: Binary mask image
    """
    pred = output[0][0]  # [C, H, W]
    pred_tensor = torch.from_numpy(pred).unsqueeze(0).float()  # [1, C, H, W]
    pred_interp = F.interpolate(
        pred_tensor,
        size=resize_size,
        mode="bilinear",
        align_corners=False
    ).squeeze(0).numpy()  # [C, H, W]
    mask = (pred_interp > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask[0])  # channel 0
    return mask_img

def run_onnx_inference(image_path, ort_session, output_dir, resize_size):
    """
    Run ONNX inference on a single image and save the predicted mask.
    Args:
        image_path (str): Path to the input image
        ort_session (onnxruntime.InferenceSession): ONNX session
        output_dir (str): Directory to save the output mask
        resize_size (tuple): (height, width) for resizing
    """
    input_tensor = preprocess_image(image_path, resize_size)
    ort_inputs = {"input": input_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)
    mask = postprocess_output(ort_outputs, resize_size)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_onnx_prediction.png")
    mask.save(output_path)
    print(f"✅ {image_path} -> {output_path}")

if __name__ == "__main__":
    """
    ONNX batch inference entrypoint.
    Loads config, model, and runs inference on all PNG images in the input folder.
    """
    parser = argparse.ArgumentParser(description="ONNX inference script with config support")
    parser.add_argument('--config', type=str, default='../configs/config.yaml', help='Path to config YAML')
    parser.add_argument('--onnx_model', type=str, default='unet_resnet34.onnx', help='Path to ONNX model')
    parser.add_argument('--images_dir', type=str, help='Override images dir')
    parser.add_argument('--outputs_dir', type=str, default='./onnx_outputs', help='Output dir')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    resize_size = tuple(config['model']['input_size'])
    images_dir = args.images_dir or config['inference']['input_images_dir']
    outputs_dir = args.outputs_dir
    onnx_model_path = args.onnx_model

    os.makedirs(outputs_dir, exist_ok=True)
    ort_session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"]
    )
    image_files = glob.glob(os.path.join(images_dir, "*.png"))
    print(f"Found {len(image_files)} images in '{images_dir}'")
    for img_path in image_files:
        run_onnx_inference(img_path, ort_session, outputs_dir, resize_size)
