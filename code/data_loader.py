import os
import random
from random import shuffle
import numpy as np
import torch.nn as nn
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(self, root,image_size,mode,augmentation_prob):
        """Initializes image paths and preprocessing module."""
        self.root = root
		
		# GT : Ground Truth
        self.GT_root = root[:-1]+'_lab/'
        self.image_paths = list(os.listdir(root))
        self.GT_paths = list(os.listdir(self.GT_root))
        self.image_size = image_size
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        
        # Construct GT path - try both _segmentation.png and _lesion.png patterns
        name, ext = os.path.splitext(image_path)
        GT_path_segmentation = f"{name}_segmentation.png"
        GT_path_lesion = f"{name}_lesion.png"
        
        # Check which GT file exists
        if os.path.exists(self.GT_root + GT_path_segmentation):
            GT_path = GT_path_segmentation
        elif os.path.exists(self.GT_root + GT_path_lesion):
            GT_path = GT_path_lesion
        else:
            raise FileNotFoundError(f"Neither {GT_path_segmentation} nor {GT_path_lesion} found in {self.GT_root}")
        
        image = Image.open(self.root+image_path)
        GT = Image.open(self.GT_root+GT_path)
        
        # Convert to RGB to ensure compatibility with augmentations
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if GT.mode != 'L':  # Convert GT to grayscale if not already
            GT = GT.convert('L')
            
        aspect_ratio = image.size[1]/image.size[0]
        
        if aspect_ratio > 1:
            Transform = T.RandomRotation((90,90),expand=True)
            image = Transform(image)
            GT = Transform(GT)
            aspect_ratio = image.size[1]/image.size[0]
        
        Transform = []
        ResizeRange = self.image_size
        Transform.append(T.Resize((ResizeRange,ResizeRange)))
        p_transform = random.random()
        
        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            
            Transform = T.Compose(Transform)
            image = Transform(image)
            GT = Transform(GT)
            
            Transform = []
            Transform.append(T.RandomInvert(p=0.2))
            Transform.append(T.ColorJitter(brightness=0.33,contrast=0.2,hue=0.02))
            Transform = T.Compose(Transform)
            image = Transform(image)

            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)

            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)
            
            Transform = []
        
        # Transform.append(T.Resize((int(self.image_size*aspect_ratio)-int(self.image_size*aspect_ratio)%16,self.image_size)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        
        image = Transform(image)
        GT = Transform(GT)
        
        if GT.shape[0] > 1:
            Transform = T.Grayscale(num_output_channels=1)
            GT = Transform(GT)
            
        GT[GT<0.8] = 0
        GT[GT>0.8] = 1

        return image, GT, image_path

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers, mode, augmentation_prob):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
