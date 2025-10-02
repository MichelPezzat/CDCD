from torch.utils.data import Dataset
import numpy as np
import io
import pandas as pd
from PIL import Image, ImageFile
import os
import json
import random
from synthesis.utils.misc import instantiate_from_config
from tqdm import tqdm
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img



class MIMIC_CXRDataset(Dataset):
    def __init__(self, data_root, negative_sample_path, phase = 'TRAIN', im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)

        self.image_folder = os.path.join(data_root, 'mimic-crx/physionet.org/files/mimic-cxr-jpg/2.1.0/')
        caption_file = os.path.join(data_root,'LLAVARAD_ANNOTATIONS_'+phase+'.csv')
        self.annotations = pd.read_csv(caption_file)

        self.num = len(self.annotations)
        self.phase = phase
        print(negative_sample_path)
        self.negative_sample_path = negative_sample_path
 
        if self.phase == 'TRAIN' and self.negative_sample_path != None:
            print("negative_sample_path:", negative_sample_path)
            extra_img = pd.read_csv(os.path.join(self.negative_sample_path,'Data_entry_2017_2020.csv'))
            self.extra_img = extra_img['Image Index']
            print("negative_sample_path:", negative_sample_path, len(self.extra_img))
            print("check path:", self.extra_img[0])
        else:
            self.extra_img = None
            

        print("load caption file done")


    def __len__(self):
        return len(self.annotations)
 
    def __getitem__(self, index):
        
        image_name = self.annotations['path'][index]
        image_path = os.path.join(self.image_folder, image_name)
        image = load_img(image_path)
        #print(image_path)
        
        
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']

        
        caption_list = self.annotations['conversations'][index].split(':')[-1][2:-4].split('.')

        caption = random.choice(caption_list).lower()

        #print(caption)
        # else:
        if self.phase == 'TRAIN' and self.extra_img[0] != None:

            for i in range(10):
                idx = random.randint(0, len(self.extra_img)-1)
                #neg_img_path = self.A_paths[idx % self.A_size]
                neg_img_name = self.extra_img[idx % len(self.extra_img)]
                img = load_img(os.path.join(self.negative_sample_path,neg_img_name))
                image = np.array(image).astype(np.uint8)                
                img = self.transform(image = img)['image']
                if i == 0:
                    neg_img = np.expand_dims(img, axis=0)
                else:
                    img = np.expand_dims(img, axis=0)
                    neg_img = np.concatenate((neg_img, img), axis=0)
            #print("check neg_img:", np.shape(neg_img))
            data = {
                    'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
                    'text': caption,
                    'negative_img': np.transpose(neg_img.astype(np.float32), (0, 3, 1, 2)),
                }   
        else:
            # neg_img = None
            data = {
                    'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
                    'text': caption,
                    # 'negative_img': np.transpose(neg_img.astype(np.float32), (0, 3, 1, 2)),
                }
        
    
        return data


