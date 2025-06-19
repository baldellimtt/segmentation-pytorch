# Binary Segmentation Using PyTorch

This repository contains code for training, inference, and analysis of binary segmentation models (e.g., for detecting growth on filter plates) using PyTorch Lightning. It is also possible to export trained models to ONNX and run predictions on them.

<p align="center">
  <img src="https://www.ilpost.it/wp-content/uploads/2024/03/08/1709898064-3.-Memoriale-Brion-Altivole-TV-evi.jpg" alt="Logo TensorFlow" width="500"/>
</p>

## Project Structure

```
segmentation-pytorch/
├── src/segmentation/
│   ├── models/         # Models (Unet, FPN, ...)
│   ├── data/           # DataModule, Dataset, Transforms
│   └── utils/          # Utilities (metrics, visualization, ...)
├── scripts/            # CLI scripts for training, inference, dataset splitting
├── checkpoints/        # Saved models
├── outputs/            # Prediction outputs
├── onnx/               # ONNX model conversion and inference
├── config.yaml         # Centralized YAML configuration
└── README.md
```

## Installing Dependencies

- If you use **conda** (recommended):
  ```bash
  conda env create -f environment.yml
  conda activate segmentation-pytorch
  ```
- Or, if you use **pip**:
  ```bash
  pip install -r requirements.txt
  ```

Tested with Python 3.11 and CUDA 12.6. Update the configuration if needed for your system.

## Configuration

All configuration (model, paths, augmentation, resize, etc.) is centralized in `config.yaml` (in the project root).

**Note on relative and absolute paths:**
- You can use either absolute or relative paths in `config.yaml` for all entries (`images_dir`, `masks_dir`, etc.).
- If you use a relative path, it will always be interpreted relative to the project root, regardless of the folder from which you launch the scripts.
- Example of a relative path:
  ```yaml
  data:
    images_dir: data/images
    masks_dir: data/masks
  ```
- Example of an absolute path:
  ```yaml
  data:
    images_dir: /absolute/path/to/images
    masks_dir: /absolute/path/to/masks
  ```

Edit this file to:
- Change architecture, backbone, parameters
- Update image, mask, output, and checkpoint paths
- Customize augmentations (see example and comments in the file)
- Set the resize size for both training and inference:
  ```yaml
  model:
    input_size: [512, 512]  # (height, width)
  ```
- Set data directories:
  ```yaml
  data:
    images_dir: /path/to/images
    masks_dir: /path/to/masks
  ```
- Set the split ratios (optional):
  ```yaml
  split_ratios:
    train: 0.7
    val: 0.15
    test: 0.15
  ```

## Dataset Splitting

- **Automatic:** The dataset is automatically split into train/val/test every time you run training (`python scripts/train.py`). No split files (`train.txt`, `val.txt`, `test.txt`) are created, and you do not need to provide any split directory in the config.
- **Reproducible seed:** The seed used for splitting is configurable via the `seed` field in `config.yaml`. Change this value to obtain a different random split, or keep it fixed for reproducibility.

- **What you need in the config:**
  ```yaml
  data:
    images_dir: /path/to/images
    masks_dir: /path/to/masks
  ```
  The `splits_dir` field is no longer required.

- **Manual split (optional):** If you only want to see the split (for debugging, for example), you can run:
```bash
python scripts/utils/split_dataset.py
```
This command will print the split to the console, but will not create any files unless explicitly requested.

## Training

To start training:
```bash
python scripts/train.py
```

## Inference (Prediction)

All inference modes are unified in `scripts/predict.py`.

### Batch prediction (entire folder):
```bash
python scripts/predict.py --mode batch
```
Processes all PNG images in the folder specified by `inference.input_images_dir` in your config.

### Single image prediction:
```bash
python scripts/predict.py --mode single --image path/to/image.png
```
Processes only the specified image and saves the result in the output folder from the config.

- The resize size used for inference is always read from `model.input_size` in the config.
- You can specify a custom config file with `--config path/to/your_config.yaml` (default is `config.yaml` in the project root).

## Notes
- All parameters and paths are centralized in `config.yaml`.
- You can add new models, datasets, or augmentation pipelines by editing only the structure in `src/segmentation/` and the YAML config.
- Notebooks are for exploration only and do not contain business logic.

---
For questions or suggestions, open an issue or contact the maintainers.
