# PyTorch implementation of UNet++ (Nested U-Net)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

This repository contains code for a image segmentation model based on [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165) implemented in PyTorch. 

[**NEW**] Add support for multi-class segmentation dataset.

[**NEW**] Add support for PyTorch 1.x.


## Requirements
- PyTorch 1.x or 0.41

## Installation
1. Create an anaconda environment.
```sh
conda create -n=<env_name> python=3.6 anaconda
conda activate <env_name>
```
2. Install PyTorch.
```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
3. Install pip packages.
```sh
pip install -r requirements.txt
```

## Training on [2024 SenNet + HOA - Hacking the Human Vasculature in 3D](https://www.kaggle.com/competitions/blood-vessel-segmentation/data) dataset
1. Download dataset from [here](https://www.kaggle.com/competitions/blood-vessel-segmentation/data) to ./ and unzip. The file structure is the following:
```

└── ./
    ├── train
    |   ├── kidney_1_dense
    │   │   ├── images
    │   │   │   └── 0000.tif
    │   │   └── labels
    │   │       └── 0000.tif           
    │   ├── kidney_1_voi
    │   │   ├── images
    │   │   │   └── 0000.tif
    │   │   └── labels
    │   │       └── 0000.tif
    │   ├── kidney_2
    │   │   ├── images
    │   │   │   └── 0000.tif
    │   │   └── labels
    │   │       └── 0000.tif
    │   ├── kidney_3_dense
    │   │   ├── images
    │   │   │   └── 0000.tif
    │   │   └── labels
    │   │       └── 0000.tif
    │   ├── kidney_3_sparse
    │   │   ├── images
    │   │   │   └── 0000.tif
    │   │   └── labels
    │   │       └── 0000.tif  
    ├── test
```
2. Preprocess.
```sh
python preprocess.py
```
3. Train the model.
```sh
python train.py --dataset train --arch NestedUNet
```
4. Evaluate.
```sh
python val.py --name test_NestedUNet_woDS
```
### (Optional) Using LovaszHingeLoss
1. Clone LovaszSoftmax from [bermanmaxim/LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax).
```
git clone https://github.com/bermanmaxim/LovaszSoftmax.git
```
2. Train the model with LovaszHingeLoss.
```
python train.py --dataset train --arch NestedUNet --loss LovaszHingeLoss
```

## Training on original dataset
Make sure to put the files as the following structure (e.g. the number of classes is 2):
```
inputs
└── <dataset name>
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
        └── 1
            ├── 0a7e06.png
            ├── 0aab0a.png
            ├── 0b1761.png
            ├── ...
```

1. Train the model.
```
python train.py --dataset <dataset name> --arch NestedUNet --img_ext .tif --mask_ext .tif
```
2. Evaluate.
```
python val.py --name <dataset name>_NestedUNet_woDS
```

## Results
### DSB2018 (96x96)

Here is the results on DSB2018 dataset (96x96) with LovaszHingeLoss.

| Model                           |   IoU   |  Loss   |
|:------------------------------- |:-------:|:-------:|
| U-Net                           |  0.839  |  0.365  |
| Nested U-Net                    |  0.842  |**0.354**|
| Nested U-Net w/ Deepsupervision |**0.843**|  0.362  |
