import os
from glob import glob
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from typing import Optional
import argparse

def main(size: Optional[int] = None):
    img_size = size
    
    # 만약 kidney_3_dense/images 디렉토리가 없다면 해당 경로 만들기 
    kidney_3_path = './train/kidney_3_dense'
    if not os.path.exists(os.path.join(kidney_3_path, 'images')):
        # 해당 디렉토리를 만들기 
        os.makedirs(os.path.join(kidney_3_path, 'images'), exist_ok=True)
    
    # labels에서 이름을 읽어와서
    label_paths = glob(os.path.join(kidney_3_path, 'labels', '*' + '.tif'))
    
    # temp 변수에 저장된 값과 kidney_3_sparse/images에서의 이름이 동일한 파일을 읽어와서
    for label in tqdm(label_paths):
        id = os.path.basename(label)
        img_path = os.path.join('./train/kidney_3_sparse/images', id)
        target_path = os.path.join('./train/kidney_3_dense/images', id)
        shutil.copy2(img_path, target_path)

    # train 디렉토리 안에 있는 5개의 디렉토리 경로 가져오기
    paths = glob('./train/*')
    
    if img_size is None: 
        os.makedirs('inputs/train/images', exist_ok=True)
        os.makedirs('inputs/train/labels', exist_ok=True)
    else:        
        os.makedirs('inputs/train_%d/images' % img_size, exist_ok=True)
        os.makedirs('inputs/train_%d/labels' % img_size, exist_ok=True)


    for path in tqdm(paths):
        # 각 디렉토리에서 images와 labels 디렉토리 경로 가져오기
        prefix = '_'.join(path.split('_')[1:])
        
        images_dir = os.path.join(path, 'images')
        labels_dir = os.path.join(path, 'labels')
        
        # 이미지와 레이블 파일들의 경로 가져오기
        image_files = glob(os.path.join(images_dir, '*.tif'))
        label_files = glob(os.path.join(labels_dir, '*.tif'))
        
        # 이미지와 레이블 파일들을 입력 디렉토리로 복사하면서 이름 변경하기
        for image_file in image_files:
            img = cv2.imread(image_file)
            
            # 이미지 크기 변경하기
            if img_size is not None:
                img = cv2.resize(img, (img_size, img_size))
                target_dir = 'inputs/train_%d/images' % img_size
            else:
                target_dir = 'inputs/train/images'
            
            # 디렉토리 이름에서 숫자 부분 추출하기
            dir_name = os.path.basename(os.path.dirname(image_file))
            
            # 이미지 파일 이름 변경하기
            image_name = prefix + '_' + os.path.basename(image_file)
            
            
            target_file = os.path.join(target_dir, image_name)
            if not os.path.exists(target_file):
                cv2.imwrite(target_file, img)
        
        for label_file in label_files:
            mask = cv2.imread(label_file)

            # 이미지 크기 변경하기
            if img_size is not None:
                mask = cv2.resize(mask, (img_size, img_size))
                target_dir = 'inputs/train_%d/labels' % img_size
            else:
                target_dir = 'inputs/train/labels'
                
            # 디렉토리 이름에서 숫자 부분 추출하기
            dir_name = os.path.basename(os.path.dirname(label_file))
            
            # 레이블 파일 이름 변경하기
            label_name = prefix + '_' + os.path.basename(label_file)
            
            target_file = os.path.join(target_dir, label_name)
            if not os.path.exists(target_file):
                cv2.imwrite(target_file, mask)
        
    # ./inputs/train_{img_size}/images에 있는 사진 파일의 개수와 /labels에 있는 사진 파일의 개수 같은지 확인하고 넘어가기 
    if isinstance(img_size, int):
        assert(len(glob('inputs/train_%d/images' % img_size)) == len(glob('inputs/train_%d/labels' % img_size)))
    else: 
        assert(len(glob('inputs/train/images')) == len(glob('inputs/train/labels')))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--img_size', type=int, 
                        help='Size of image (default: original size)')
    args = parser.parse_args()
    main(args.img_size)
