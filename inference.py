# Import Dependencies 
import cv2 
import os, sys
from glob import glob 
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Optional, List

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel # Enables parallel processing 
import segmentation_models_pytorch as smp 
from albumentations.augmentations import transforms, geometric
from albumentations.core.composition import Compose, OneOf

from utils import AverageMeter
from archs import NestedUNet


# Config Class 
class CFG:
    model_name = 'NestedUNet'
    
    in_channels = 3
    input_h = 96
    input_w = 96
    num_workers = 2
    num_classes = 1
    batch_size = 64
    model_path = '/root/vasculature-segmentation/models/train_NestedUNet_woDS/model.pth'

config = CFG()

# Custom Dataset Class 
class CustomDataset(Dataset):
    def __init__(self,  
                 img_ids: Optional[List[str]] = None,
                 img_dir: str = './test/', 
                 transform=None):
        """
        Args:
            img_ids: Image name 
            img_dir: Image file directory.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        """
        
        if img_ids is None:
            img_ids = []
            for kidney_dir in glob(os.path.join(img_dir, '*')):
                kidney_name = kidney_dir.split('/')[-1]
                for image_path in glob(os.path.join(kidney_dir, 'images', '*')):
                    img_ids.append(kidney_name + '_' + os.path.basename(image_path).split('.')[0])
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        kidney, num, id = img_id.split('_')
        img = cv2.imread(os.path.join(self.img_dir, kidney + '_' + num, 'images', id + '.tif'))
        
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        img = img.astype('float32') / 255
        if img.ndim != 3:
            img = np.expand_dims(img, axis=2)
        img = img.transpose(2, 0, 1)
        
        return img, img_id
    
# Define Transforms
test_transform = Compose([
    geometric.Resize(config.input_h, config.input_w),
    transforms.Normalize(),
])

# Instantiate Dataset & Dataloader
test_dataset = CustomDataset(
    img_dir='./test/',
    transform=test_transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    drop_last=False
)

avg_meter = AverageMeter()

# Instantiate and load model
model = NestedUNet(config.num_classes)
model = DataParallel(model)
state_dict = torch.load(config.model_path)
model.load_state_dict(state_dict)
model.eval()

# Get output
def get_output(model, test_loader):
    model.eval()  # redundant if called outside, but for safety
    labels = []
    img_ids = []
    
    for batch_imgs, batch_ids in tqdm(test_loader):
        batch_imgs = batch_imgs.cuda()  # Move images to GPU
        with torch.no_grad():  # No need to track gradients
            batch_preds = model(batch_imgs)
        
        for pred, img_id in zip(batch_preds, batch_ids):
            pred = (pred * 255 / 3).byte().cpu().numpy()  # Adjust this line based on your model output
            labels.append(pred)
            img_ids.append(img_id)
    
    return labels, img_ids

outputs, img_ids = get_output(model, test_loader)

# RLE encode function 
def rle_encode(mask):
    # Flatten mask into a 1D array
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    # Find where the pixel values change
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    # Make sure it gets even number of elements
    if len(run) % 2 != 0:
        run = np.append(run, len(pixel))
    # Get the length
    run[1::2] -= run[::2]
    # Make it into a RLE format string
    rle = ' '.join(str(r) for r in run)
    # Change it to '1 0' if it's empty
    if rle == '':
        rle = '1 0'
    return rle

# Prepare submission dataframe
submission_data = {
    'id': [],
    'rle': []
}

for i in range(len(img_ids)):
    rle = rle_encode(outputs[i])
    submission_data['id'].append(img_ids[i])
    submission_data['rle'].append(rle)

submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('submission.csv', index=False)
submission_df.head(5)
