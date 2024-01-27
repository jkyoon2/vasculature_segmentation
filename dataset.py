import os
from typing import List
import cv2
import numpy as np
import torch
import torch.utils.data
from typing import Optional
from glob import glob

__all__ = ['train_96', 'train_128', 'train_256', 'train']

class Dataset(torch.utils.data.Dataset):
    def __init__(self,  
                 img_ids: Optional[List[str]] = None,
                 img_dir: str = './inputs/train/images/', 
                 mask_dir: str = './inputs/train/labels/', 
                 img_ext: str = '.tif',
                 mask_ext: str = '.tif',
                 transform=None):
        """
        Args:
            img_ids: Image name 
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        if img_dir is None:
            self.img_ids = [os.path.basename(image_path).split('.')[0] for image_path in glob(os.path.join(self.img_dir, '*'))]
        else:
            self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        if img.ndim != 3:
            img = np.expand_dims(img, axis=2)
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = np.expand_dims(mask, axis=-1)
        # if mask.ndim != 3:
            # mask = np.expand_dims(mask, axis=2)
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}