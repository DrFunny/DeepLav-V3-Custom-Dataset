import os
from random import shuffle
from PIL import Image
import numpy as np

from tqdm import tqdm

def create_dataset_splits(data_dir):
    print("creating dataset splits...")
    fnames = os.listdir(data_dir)
    fnames_stripped = []
    for fname in fnames:
        head, _ = fname.split('.')
        fnames_stripped.append(head)
    
    shuffle(fnames_stripped)
    
    ntrain = int(len(fnames_stripped) * 0.8)
    fnames_train = fnames_stripped[:ntrain]
    fnames_val = fnames_stripped[ntrain:]
    
    with open('dataset/train.txt', 'w') as wf:
        wf.write("\n".join(fnames_train))
    
    with open('dataset/val.txt', 'w') as wf:
        wf.write("\n".join(fnames_val))
    print("done.")
        
def transform_labels(label_dir, new_label_dir):
    print("transforming labels...")
    palette = {
                (0, 0, 255) : 0 ,
                (255, 0, 0) : 1 
            }
    
    def convert_from_color_segmentation(arr_3d):
        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    
        for c, i in palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i
    
        return arr_2d
    
    label_files = os.listdir(label_dir)
    for l_f in tqdm(label_files):
        arr = np.array(Image.open(label_dir + l_f).convert("RGB"))
        arr_2d = convert_from_color_segmentation(arr)
        Image.fromarray(arr_2d).save(new_label_dir + l_f)
    print('done.')
    
if __name__ == '__main__':
    
    create_dataset_splits('data/raw/images')
    transform_labels('data/raw/masks/', 'data/raw/masks_aug/')
    