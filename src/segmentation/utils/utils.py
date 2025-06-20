import sys
import os
import yaml
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.segmentation.models.unet import SegmentationModels

"""
Utility functions for segmentation model loading and configuration.
"""

def load_torch_model(config_path="configs/config.yaml"):
    """
    Load a PyTorch Lightning segmentation model and inference paths from a YAML config file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        tuple: (model, images_path, outputs_path)
            model (SegmentationModels): Loaded PyTorch Lightning model (in eval mode)
            images_path (str): Directory containing input images for inference
            outputs_path (str): Directory to save output predictions
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get model parameters and paths from config
    model_cfg = config["model"]
    inference_cfg = config["inference"]

    model = SegmentationModels.load_from_checkpoint(
        inference_cfg["model_path"],
        arch=model_cfg.get("architecture", "fpn"),
        encoder_name=model_cfg.get("backbone", "resnet34"),
        in_channels=model_cfg.get("in_channels", 3),
        out_classes=model_cfg.get("out_classes", 1),
    )
    model.eval()

    images_path = inference_cfg["input_images_dir"]
    outputs_path = inference_cfg["outputs_dir"]
    return (
        model,
        images_path,
        outputs_path,
    )

def load_config(config_path):
    """
    Carica e restituisce il contenuto di un file di configurazione YAML.
    Args:
        config_path (str): Percorso al file YAML di configurazione.
    Returns:
        dict: Configurazione caricata dal file YAML.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config