from torch.utils.data import Dataset
import numpy as np
import io
import pandas as pd
from PIL import Image
import os
import json
import random
from synthesis.utils.misc import instantiate_from_config
from tqdm import tqdm
import pickle

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding="utf-8") as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

class ChestXray8Dataset(Dataset):
    def __init__(self, data_root, images_files,negative_sample_path, phase = 'train', im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.images_files = files_to_list(images_files)
        self.images_files = [Path(images_files).parent / x for x in self.images_files]
        data_path = os.path.join(data_root, "Data_Entry_2017.csv")
        #self.name_list = pickle.load(open(pickle_path, 'rb'), encoding="bytes")
        data = pd.read_csv(data_path)
        self.negative_sample_path = negative_sample_path
        #self.num = len(self.name_list)
        self.phase = phase
        if self.phase == 'train' and self.negative_sample_path != None:
            # print("negative_sample_path:", negative_sample_path)
            with open(negative_sample_path, 'r') as f:
                self.extra_img = json.load(f)
            # self.extra_img = os.path.join()
            print("negative_sample_path:", negative_sample_path, len(self.extra_img))
            print("check path:", self.extra_img[0])
        else:
            self.extra_img = None

        # load all caption file to dict in memory
        self.caption_dict =  data['Finding Labels']


        # print("check name_list:", len(self.name_list))
        # exit()

        # for index in tqdm(range(self.num)):
        #     name = self.name_list[index]
        #     this_text_path = os.path.join(data_root, 'text', 'text', name+'.txt')
        #     image_path = os.path.join(self.image_folder, name+'.jpg')
        #     if not os.path.exists(image_path) or not os.path.exists(this_text_path):
        #         print("missing file:", image_path, this_text_path)




        print("load caption file done")


    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        #name = self.name_list[index]
        image_path = self.image_files[index]
        # if os.path.exists(image_path):
        #     print(index, image_path)
        image = load_img(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']
        caption_list = self.caption_dict[index]
        caption = caption_list.replace('|', '').lower()
        # else:
        if self.phase == 'train' and self.extra_img != None:
            neg_sample = self.extra_img[index]
            for i in range(len(neg_sample)):
                img = load_img(os.path.join(self.image_folder, neg_sample[i]))
                img = np.array(img).astype(np.uint8)
                img = self.transform(image = img)['image']
                if i == 0:
                    neg_img = np.expand_dims(img, axis=0)
                else:
                    img = np.expand_dims(img, axis=0)
                    neg_img = np.concatenate((neg_img, img), axis=0) 
            # print("check data loader:", np.shape(image), np.shape(neg_img))
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


