import os
import random
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
from roi import crop_leaf_roi

# --- Configuration ---
DATA_ROOT = Path("./Dhan-Shomadhan")
SOURCE_DIRS = [DATA_ROOT / "White Background", DATA_ROOT / "Field Background"]
CROPS_DIR = DATA_ROOT / "crops"
SPLITS_DIR = DATA_ROOT / "splits"
NUM_RUNS = 5
TEST_SIZE = 0.20
VAL_SIZE = 0.10

def prepare_dataset():
    """
    Scans source directories, creates stratified splits for 5 runs,
    and caches ROI-cropped images.
    """
    # Automatically delete old crops folder 
    if CROPS_DIR.exists():
        print(f"Found existing crops folder. Deleting {CROPS_DIR} to ensure a fresh run...")
        shutil.rmtree(CROPS_DIR)
    
    # 1. Create necessary directories
    CROPS_DIR.mkdir(exist_ok=True)
    SPLITS_DIR.mkdir(exist_ok=True)

    # 2. Scan source folders to get a master list of images
    print("Scanning source image folders...")
    samples = []
    for source_dir in SOURCE_DIRS:
        background_type = source_dir.name
        for disease_dir in source_dir.iterdir():
            if disease_dir.is_dir():
                disease_name = disease_dir.name
                for img_path in disease_dir.glob("*.jpg"):
                    stratify_label = f"{disease_name}_{background_type}"
                    samples.append({
                        "path": img_path,
                        "disease": disease_name,
                        "background": background_type,
                        "stratify_label": stratify_label
                    })
    
    if not samples:
        print("Error: No images found. Check your DATA_ROOT and source directories.")
        return
    
    total_images = len(samples)
    print(f"Found {total_images} total images.")
    
    filepaths = [s["path"] for s in samples]
    stratify_labels = [s["stratify_label"] for s in samples]

    # 3. Create 5 runs of stratified splits
    for i in range(1, NUM_RUNS + 1):
        run_dir = SPLITS_DIR / f"run_{i}"
        run_dir.mkdir(exist_ok=True)
        print(f"\nCreating splits for run {i}...")

        splitter_test = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=i)
        train_val_idx, test_idx = next(splitter_test.split(filepaths, stratify_labels))

        train_val_filepaths = [filepaths[j] for j in train_val_idx]
        train_val_labels = [stratify_labels[j] for j in train_val_idx]
        val_split_size = VAL_SIZE / (1 - TEST_SIZE)
        
        splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=val_split_size, random_state=i)
        train_idx_rel, val_idx_rel = next(splitter_val.split(train_val_filepaths, train_val_labels))
        
        train_idx = [train_val_idx[j] for j in train_idx_rel]
        val_idx = [train_val_idx[j] for j in val_idx_rel]
        
        split_map = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        for split_name, indices in split_map.items():
            split_path = run_dir / f"{split_name}.txt"
            with open(split_path, 'w') as f:
                for idx in indices:
                    sample = samples[idx]
                    f.write(f"{sample['path']},{sample['disease']},{sample['background']}\n")
            print(f"  - Wrote {split_name}.txt with {len(indices)} samples.")

    # 4. Cache all the ROI-cropped images
    print("\nCaching ROI-cropped images...")
    for i, sample in enumerate(samples):
        original_path = sample["path"]
        cached_path = CROPS_DIR / original_path.name
        
        if (i + 1) % 100 == 0:
            print(f"  Processing image {i + 1}/{total_images}...")
            
        if not cached_path.exists():
            crop_leaf_roi(str(original_path), str(cached_path))
    
    print("\nData preparation complete!")

if __name__ == '__main__':
    prepare_dataset()
