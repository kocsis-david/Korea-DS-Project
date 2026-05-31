import cv2
import os
import re
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import glob
import random
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from src.preprocess import process, fungi_collate_fn

class FungiDataset(Dataset):
    def __init__(self, config):
        '''
        Args:
            image_dir: directory containing the images
            labels_path: path to the labels csv file
            train: True if training, False if validation
            pre_load: True if images should be loaded into memory, False otherwise
        '''

        '''
        Args:
            config: dictionary containing all initialization parameters:
                - image_dir: directory containing the images
                - labels_path: path to the labels CSV file
                - train: True if training, False if validation
                - pre_load: True if images should be loaded into memory, False otherwise
                - train_val_split: Fraction of data used for validation
                - batch_size: Batch size for the DataLoader
                - crop_height: Height for cropping images
                - interpolate: Interpolation method
                - out_size: Output size for resized images
        '''
        self.image_dir = config.get("image_dir")
        self.labels_path = config.get("labels_path")
        self.train = config.get("train", True)
        self.pre_load = config.get("pre_load", True)
        self.train_val_split = config.get("train_val_split", 0.2)
        self.batch_size = config.get("batch_size", 32)
        self.crop_h = config.get("crop_height", 16)
        self.interpolate = config.get("interpolate", "bilinear")
        self.out_size = config.get("out_size", [300, 225])

        (self.data, self.targets) = self._load_from_disk()
        # Create DataLoader
        self.loader = DataLoader(self, batch_size= self.batch_size, shuffle=self.train, num_workers=0, drop_last=self.train) # collate_fn = fungi_collate_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.pre_load:
            image = self.data[idx]
            toxicity = self.targets[idx, 1]  # Select toxicity (poisonous or edible)
            img_name = self.targets[idx, 0].split("/")[-1]  # Extract image name (if stored as path)
            return image, toxicity, img_name
        else:
            img_path = self.data[idx]
            img = cv2.imread(img_path)
            np_img = process(img, crop_s=self.crop_h, interp_mode=self.interpolate, out_size=self.out_size)
            toxicity = self.targets[idx, 1]  # Select toxicity (poisonous or edible)
            img_name = img_path.split("/")[-1]  # Extract image name
            return torch.from_numpy(np_img), toxicity, img_name
            

    def _load_from_disk(self):
        '''
        Load and match image files with their corresponding CSV entries. 
        Matching is necessary beacuse the image directory and the CSV file may not contain the data in the same order.

        Returns:
            data: tensor of image data if pre_load is True, otherwise tensor of image paths
            targets: tensor of class and poisonous labels
        '''
        # csv containing the labels
        metadata = pd.read_csv(self.labels_path)

        # image files in the image directory
        image_files = image_files = {f for f in os.listdir(self.image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg'))}
        
        # filter dataframe to include only existing images
        metadata = metadata[metadata['image_path'].isin(image_files)]

        if metadata.empty:
            raise ValueError('No matching images found in the image directory')

        # load images if pre_load is True, otherwise store image paths
        images = []
        for image_path in metadata['image_path']:
            if image_path.lower().endswith((".jpg", ".png", ".PNG", ".JPG")):
                img_path = cv2.imread(os.path.join(self.image_dir, image_path))

                if self.pre_load:
                    image = process(img_path, crop_s = self.crop_h, interp_mode = self.interpolate, out_size = self.out_size)
                    images.append(image)

                else:
                    images.append(img_path)
    
        # convert images and targets to tensors
        if self.pre_load:
            images_tensor = torch.stack([torch.from_numpy(img) for img in images])[:-3]
            print(images_tensor.shape)
        else: 
            images_tensor = images

        targets_tensor = torch.tensor(
            metadata[['class_id', 'poisonous']].values[:-3], dtype=torch.long
        )
        print(targets_tensor.shape)

        images_tensor.permute(0, 3, 1, 2)

        # split data into training and validation sets
        self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(images_tensor, targets_tensor, test_size=self.train_val_split, random_state=42)

        return (self.train_data, self.train_labels) if self.train else (self.val_data, self.val_labels)

    def get_loader(self):
        ''' Return the DataLoader for this dataset. '''
        return self.loader
    
    def get_data(self, train = True):
        if train is True:
            return self.train_data, self.train_labels
        else: return self.val_data, self.val_labels


'''  
# Testing DataLoader

config_train = {
    "image_dir": "/Users/czimbermark/Documents/Egyetem/Adatelemzes/Nagyhazi/FungiCLEF2024_ADC/data/x_train",
    "labels_path": "/Users/czimbermark/Documents/Egyetem/Adatelemzes/Nagyhazi/FungiCLEF2024_ADC/data/train_metadata_height.csv",
    "train": True,
    "pre_load": True,
    "batch_size": 32,
    "crop_height": 16,
    "interpolate": "bilinear",
    "out_size": [300, 225]
}

config_val = {
    "image_dir": "/Users/czimbermark/Documents/Egyetem/Adatelemzes/Nagyhazi/FungiCLEF2024_ADC/data/x_train",
    "labels_path": "/Users/czimbermark/Documents/Egyetem/Adatelemzes/Nagyhazi/FungiCLEF2024_ADC/data/train_metadata_height.csv",
    "train": False,
    "pre_load": True,
    "batch_size": 32,
    "crop_height": 16,
    "interpolate": "bilinear",
    "out_size": [300, 225]
}

# Initialize datasets using configuration dictionaries
train_dataset = FungiDataset(config_train)
print(len(train_dataset))

val_dataset = FungiDataset(config_val)
print(len(val_dataset))
    
# Retrieve DataLoader
train_loader = train_dataset.get_loader()

# Iterate through the DataLoader
# for batch_data, batch_targets in train_loader:
   # print(f"Batch data shape: {batch_data.shape}, Batch targets: {batch_targets}")
    
#'''