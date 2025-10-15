import torch
from torch.utils.data import Dataset
from PIL import Image
import os

# Import the new augmentation functions
from aug import get_base_train_transforms, get_strong_train_transforms

class RiceDiseaseDataset(Dataset):
    def __init__(self, data_root, split_file_path, is_train=False, transform=None):
        """
        Args:
            data_root (string): Path to the Dhan-Shomadhan directory.
            split_file_path (string): Path to the txt file with image info for the split.
            is_train (bool): If True, applies targeted training augmentations.
            transform (callable, optional): Used for validation/testing transforms.
        """
        self.crops_dir = os.path.join(data_root, 'crops')
        self.samples = []
        self.is_train = is_train
        
        self.class_to_idx = {
            "Brown Spot": 0, "Leaf Scald": 1, "Rice Blast": 2, 
            "Rice Tungro": 3, "Sheath Blight": 4
        }
        
        # If this is a training dataset, set up both augmentation pipelines
        if self.is_train:
            self.base_transform = get_base_train_transforms()
            self.strong_transform = get_strong_train_transforms()
            # Define which classes are the 'hard' ones that need strong augmentation
            self.hard_classes = {}
            # For testing, uncomment the line below
            # self.hard_classes = {self.class_to_idx["Brown Spot"], self.class_to_idx["Leaf Scald"]}
        else:
            # For validation/test sets, just use the single provided transform
            self.transform = transform

        with open(split_file_path, 'r') as f:
            for line in f.read().strip().split('\n'):
                if not line: continue
                original_path_str, class_name, _ = line.strip().split(',')
                img_filename = os.path.basename(original_path_str)
                crop_path = os.path.join(self.crops_dir, img_filename)
                
                if os.path.exists(crop_path):
                    self.samples.append((crop_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop_path, label = self.samples[idx]
        image = Image.open(crop_path).convert('RGB')
        
        if self.is_train:
            if label in self.hard_classes:
                image = self.strong_transform(image)
            else:
                image = self.base_transform(image)
        elif self.transform:
            image = self.transform(image)
            
        return image, label
