import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import pandas as pd
from src.preprocess import process
from torch.utils.data import Subset
from torch.utils.data import DataLoader

def fungi_collate_fn(batch):
    images, class_ids, toxicities, img_names = [], [], [], []
    for image, (class_id, toxicity), img_name in batch:
        # print(f"Individual image shape: {image.shape}")  # Should be [C, H, W]
        images.append(image)
        class_ids.append(class_id)
        toxicities.append(toxicity)
        img_names.append(img_name)

    images = torch.stack(images)  # Should result in shape [batch_size, C, H, W]
    #print(f"Batch images shape after stacking: {images.shape}")

    class_ids = torch.tensor(class_ids, dtype=torch.long)
    toxicities = torch.tensor(toxicities, dtype=torch.long)

    return images, (class_ids, toxicities), img_names

class FungiDataset(Dataset):
    def __init__(self, image_dir, labels_path, pre_load=True, crop_height=16, interpolate="bilinear", out_size=(300, 225), transform=None, class_ids_to_include=None):
        '''
        Args:
            image_dir: directory containing the images
            labels_path: path to the labels CSV file
            pre_load: True if images should be loaded into memory, False otherwise
            crop_height: Height for cropping images
            interpolate: Interpolation method
            out_size: Output size for resized images
            transform: Optional transform to be applied on a sample.
        '''
        self.image_dir = image_dir
        self.labels_path = labels_path
        self.pre_load = pre_load
        self.crop_h = crop_height
        self.interpolate = interpolate
        self.out_size = out_size
        self.transform = transform
        self.load_num = 0

        # Load metadata
        metadata = pd.read_csv(self.labels_path)

        # Filter to include only existing imgs
        image_files = set(os.listdir(self.image_dir))
        metadata = metadata[metadata['image_path'].isin(image_files)]

        if metadata.empty:
            raise ValueError('No matching images found in the image directory')
        
        # **Filter to include only specified class IDs**
        if class_ids_to_include is not None:
            metadata = metadata[metadata['class_id'].isin(class_ids_to_include)]
            metadata.reset_index(drop=True, inplace=True)

        # Reset index after filtering
        metadata.reset_index(drop=True, inplace=True)

        # Ensure labels are integers
        metadata['class_id'] = metadata['class_id'].astype(int)
        metadata['poisonous'] = metadata['poisonous'].astype(int)

        # **Remap class IDs to a continuous range starting from 0**
        unique_class_ids = sorted(metadata['class_id'].unique())
        class_id_to_idx = {original_id: idx for idx, original_id in enumerate(unique_class_ids)}
        metadata['class_id'] = metadata['class_id'].map(class_id_to_idx)

        # Update class IDs and calculate the number of species classes
        self.metadata = metadata
        self.image_paths = metadata['image_path'].tolist()
        self.class_ids = metadata['class_id'].tolist()
        self.toxicities = metadata['poisonous'].tolist()
        self.num_species_classes = len(unique_class_ids)

        # Pre-load images if required
        if self.pre_load:
            self.images = []
            for img_name in self.image_paths:
                img_path = os.path.join(self.image_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Image {img_path} could not be read.")
                    continue
        
                img_processed = process(img, crop_s=self.crop_h, interp_mode=self.interpolate, out_size=self.out_size)
                # Convert image to tensor and float32
                image = torch.from_numpy(img_processed).float()
                image = image.permute(2, 0, 1)

                # Normalize using ImageNet mean and std
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = (image - mean) / std
                self.images.append(image)
                self.load_num += 1
                if self.load_num / 3 == 0:
                  print(f"{self.load_num / 1000} images loaded")
        else:
            self.images = None  # Images will be loaded in __getitem__

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.pre_load:
            image = self.images[idx]
        else:
            img_name = self.image_paths[idx]
            img_path = os.path.join(self.image_dir, img_name)

            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Image at {img_path} could not be read.")
                
            image_processed = process(img, crop_s=self.crop_h, interp_mode=self.interpolate, out_size=self.out_size)

            # Convert image to tensor and float32
            img_tensor = torch.from_numpy(image_processed).float().permute(2, 0, 1)

            # Normalize using ImageNet mean and std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (img_tensor - mean) / std

            '''
            print(image.shape)
            print(f"Image dtype: {image.dtype}")  # Should be torch.float32
            print(f"Image min: {image.min()}, max: {image.max()}")  # Should be within expected range
            '''
        
        # labels
        class_id = self.class_ids[idx]
        toxicity = self.toxicities[idx]

        return image, (class_id, toxicity), img_name
    
    