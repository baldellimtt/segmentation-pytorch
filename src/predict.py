import os
import sys
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torchvision import transforms
import torch.nn.functional as F
import time
import os
import glob
from PIL import Image
import numpy as np
import argparse
from src.segmentation.utils.utils import load_torch_model
import yaml

def get_config_path(args_config):
    if os.path.isabs(args_config):
        return args_config
    # If relative, resolve with respect to project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args_config))

def run_inference(chunks, batch_size, model, resize_size):
    """
    Run inference on a list of images (chunks) using the given model.
    Applies preprocessing (ToTensor, Normalize, Resize) and postprocessing (sigmoid, threshold, resize).
    Args:
        chunks (list of np.ndarray): List of images as numpy arrays.
        batch_size (int): Batch size for inference.
        model (torch.nn.Module): The segmentation model.
        resize_size (tuple): (height, width) for resizing images and outputs.
    Returns:
        list of np.ndarray: List of binary masks (0/1) for each input image.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(resize_size),
        ]
    )
    start = time.time()
    outputs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        if transform:
            batch_tensor = torch.stack([transform(image) for image in batch])
        with torch.no_grad():
            output = model(batch_tensor.to(device))
        output = output.sigmoid()
        output = F.interpolate(
            output, size=(resize_size[0], resize_size[1]), mode="bilinear", align_corners=False
        )
        if len(output.shape) == 4:
            output_binary = output[:, 0].cpu().numpy()
        elif len(output.shape) == 3:
            output_binary = output.cpu().numpy()
        else:
            print(f"Unexpected output shape: {output.shape}")
            continue
        output_binary[output_binary > 0.5] = 1
        output_binary[output_binary <= 0.5] = 0
        if len(output_binary.shape) == 3 and output_binary.shape[0] == 1:
            output_binary = output_binary[0]
        outputs.append(output_binary)
    end = time.time()
    print("Inference time {}".format(end - start))
    return outputs

def predict_folder(model, images_path, outputs_path, resize_size, batch_size=1):
    """
    Run inference on all PNG images in a folder and save the predicted masks.
    Args:
        model (torch.nn.Module): The segmentation model.
        images_path (str): Directory containing input images.
        outputs_path (str): Directory to save output masks.
        resize_size (tuple): (height, width) for resizing.
        batch_size (int): Batch size for inference (default: 1).
    """
    os.makedirs(outputs_path, exist_ok=True)
    image_files = glob.glob(os.path.join(images_path, "*.png"))
    print(f"Found {len(image_files)} images to process")
    for image_path in image_files:
        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
            outputs = run_inference([image], batch_size, model, resize_size)
            output_binary = outputs[0]
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = os.path.join(outputs_path, f"{base_name}_prediction.png")
            output_image = Image.fromarray((output_binary * 255).astype(np.uint8))
            output_image.save(output_filename)
            print(f"Processed {image_path} -> {output_filename}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def predict_single(model, image_path, outputs_path, resize_size):
    """
    Run inference on a single image and save the predicted mask.
    Args:
        model (torch.nn.Module): The segmentation model.
        image_path (str): Path to the input image.
        outputs_path (str): Directory to save the output mask.
        resize_size (tuple): (height, width) for resizing.
    """
    os.makedirs(outputs_path, exist_ok=True)
    try:
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        outputs = run_inference([image], batch_size=1, model=model, resize_size=resize_size)
        output_binary = outputs[0]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = os.path.join(outputs_path, f"{base_name}_prediction.png")
        output_image = Image.fromarray((output_binary * 255).astype(np.uint8))
        output_image.save(output_filename)
        print(f"Processed {image_path} -> {output_filename}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    """
    Entrypoint for segmentation prediction.
    Loads model and config, then runs batch or single-image prediction based on CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Segmentation prediction script (batch or single image)")
    parser.add_argument('--mode', choices=['batch', 'single'], default='batch', help='Prediction mode: batch (default) or single')
    parser.add_argument('--image', type=str, help='Path to a single image (required for single mode)')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()

    config_path = get_config_path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    resize_size = tuple(config['model']['input_size'])

    model, images_path, outputs_path = load_torch_model(config_path)

    if args.mode == 'batch':
        predict_folder(model, images_path, outputs_path, resize_size)
    elif args.mode == 'single':
        if not args.image:
            print("Error: --image is required in single mode")
        else:
            predict_single(model, args.image, outputs_path, resize_size) 