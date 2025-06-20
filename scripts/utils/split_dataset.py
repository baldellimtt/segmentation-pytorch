import os
import sys
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.segmentation.utils.utils import load_config

"""
CLI script to split a dataset of images and masks into train, val, and test folders.
Each split will have 'images/' and 'masks/' subfolders containing the corresponding files.
Usually, you do not need to run this manually, as it is called automatically by train.py.
"""

# --- Funzione split_dataset ---
def split_dataset(config, seed=None):
    """
    Divide il dataset di immagini e maschere in cartelle train, val e test.
    Restituisce le liste di path per ciascuno split.
    Args:
        config (dict): Configurazione.
        seed (int, opzionale): Seed per la randomizzazione.
    Returns:
        tuple: (train_paths, val_paths, test_paths)
    """
    import shutil
    import random
    from pathlib import Path

    # Calcola la root del progetto rispetto a questo file
    project_root = Path(__file__).resolve().parent.parent.parent
    splits_dir = project_root / config['data'].get('splits_dir', 'splits')
    images_dir = project_root / config['data']['images_dir']
    masks_dir = project_root / config['data']['masks_dir']

    if seed is not None:
        random.seed(seed)
    split_ratios = config['split_ratios']
    splits = ['train', 'val', 'test']
    # Crea lista di immagini e maschere
    image_files = sorted([f for f in images_dir.glob('*.png')])
    mask_files = sorted([f for f in masks_dir.glob('*.png')])
    if not image_files or not mask_files:
        raise FileNotFoundError(f"Nessuna immagine trovata in {images_dir} o nessuna maschera trovata in {masks_dir}. Verifica i path e che i file .png siano presenti.")
    assert len(image_files) == len(mask_files), 'Numero immagini e maschere non corrisponde.'
    # Shuffle
    combined = list(zip(image_files, mask_files))
    random.shuffle(combined)
    image_files, mask_files = zip(*combined)
    n = len(image_files)
    n_train = int(n * split_ratios['train'])
    n_val = int(n * split_ratios['val'])
    n_test = n - n_train - n_val
    split_indices = {
        'train': (0, n_train),
        'val': (n_train, n_train + n_val),
        'test': (n_train + n_val, n)
    }
    split_paths = {}
    for split in splits:
        split_dir = splits_dir / split
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'masks').mkdir(parents=True, exist_ok=True)
        start, end = split_indices[split]
        split_image_paths = []
        split_mask_paths = []
        for img_path, mask_path in zip(image_files[start:end], mask_files[start:end]):
            dest_img = split_dir / 'images' / img_path.name
            dest_mask = split_dir / 'masks' / mask_path.name
            shutil.copy(img_path, dest_img)
            shutil.copy(mask_path, dest_mask)
            split_image_paths.append(str(dest_img))
            split_mask_paths.append(str(dest_mask))
        split_paths[split] = (split_image_paths, split_mask_paths)
    print(f'Dataset suddiviso in train/val/test nella cartella: {splits_dir}')
    return split_paths['train'], split_paths['val'], split_paths['test']

if __name__ == "__main__":
    # Entry point for manual split
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml'))
    config = load_config(config_path)
    split_dataset(config) 