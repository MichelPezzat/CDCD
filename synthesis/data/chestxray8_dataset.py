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

<<<<<<< HEAD
<<<<<<< HEAD
=======

class ChestXray8Dataset(Dataset):
    def __init__(self, data_root, image_files, negative_sample_path, phase = 'train', im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.image_path = os.path.join(data_root, 'images')
        self.image_index = files_to_list(image_files)
        self.image_files = [os.path.join(self.image_path, x) for x in self.image_index]
        #self.root = os.path.join(data_root, phase)
        caption_file = os.path.join(data_root, "Data_Entry_2017_v2020.csv")
        #self.name_list = pickle.load(open(pickle_path, 'rb'), encoding="bytes")
        caption = pd.read_csv(caption_file)
        self.names_list = caption['Image Index']
        self.negative_sample_path = negative_sample_path
        self.num = len(self.names_list)
=======
>>>>>>> d74a0e337b8516c4db74d8e622966d39852e9476
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
<<<<<<< HEAD
=======

class ChestXray8Dataset(Dataset):
    def __init__(self, data_root, image_files, negative_sample_path, phase = 'train', im_preprocessor_config=None):
        self.transform = instantiate_from_config(im_preprocessor_config)
        self.image_path = os.path.join(data_root, 'images')
        self.image_index = files_to_list(image_files)
        self.image_files = [os.path.join(self.image_path, x) for x in self.image_index]
        #self.root = os.path.join(data_root, phase)
        caption_file = os.path.join(data_root, "Data_Entry_2017_v2020.csv")
        #self.name_list = pickle.load(open(pickle_path, 'rb'), encoding="bytes")
        caption = pd.read_csv(caption_file)
        self.names_list = caption['Image Index']
        self.negative_sample_path = negative_sample_path
        self.num = len(self.names_list)
>>>>>>> 739735d6095bfcbc3ebf457b3ebddea38396e35f
=======
>>>>>>> 5527de0ac126209a4aa701c4f4221bcff6c70ddf
>>>>>>> d74a0e337b8516c4db74d8e622966d39852e9476
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
<<<<<<< HEAD
<<<<<<< HEAD
        self.caption_dict =  data['Finding Labels']
=======
        #self.caption_labels =  caption['Finding Labels']
>>>>>>> 739735d6095bfcbc3ebf457b3ebddea38396e35f
=======
        #self.caption_labels =  caption['Finding Labels']
=======
        self.caption_dict =  data['Finding Labels']
>>>>>>> 5527de0ac126209a4aa701c4f4221bcff6c70ddf
>>>>>>> d74a0e337b8516c4db74d8e622966d39852e9476


        # print("check name_list:", len(self.name_list))
        # exit()
<<<<<<< HEAD
<<<<<<< HEAD
=======
        caption_labels = {}
        for index in tqdm(range(self.num)):
             name = self.names_list[index]
             if name in self.image_index:
                caption_labels[index] = caption['Finding Labels'][index]
        #    this_text_path = os.path.join(data_root, 'text', 'text', name+'.txt')
        #    image_path = os.path.join(self.image_folder, name+'.jpg')
        #     if not os.path.exists(image_path) or not os.path.exists(this_text_path):
        #         print("missing file:", image_path, this_text_path)
        self.caption_labels = list(caption_labels.values())
        #print(len(self.image_files),len(self.caption_labels))
=======
>>>>>>> d74a0e337b8516c4db74d8e622966d39852e9476

        # for index in tqdm(range(self.num)):
        #     name = self.name_list[index]
        #     this_text_path = os.path.join(data_root, 'text', 'text', name+'.txt')
        #     image_path = os.path.join(self.image_folder, name+'.jpg')
        #     if not os.path.exists(image_path) or not os.path.exists(this_text_path):
        #         print("missing file:", image_path, this_text_path)
<<<<<<< HEAD
=======
        caption_labels = {}
        for index in tqdm(range(self.num)):
             name = self.names_list[index]
             if name in self.image_index:
                caption_labels[index] = caption['Finding Labels'][index]
        #    this_text_path = os.path.join(data_root, 'text', 'text', name+'.txt')
        #    image_path = os.path.join(self.image_folder, name+'.jpg')
        #     if not os.path.exists(image_path) or not os.path.exists(this_text_path):
        #         print("missing file:", image_path, this_text_path)
        self.caption_labels = list(caption_labels.values())
        #print(len(self.image_files),len(self.caption_labels))
>>>>>>> 739735d6095bfcbc3ebf457b3ebddea38396e35f
=======
>>>>>>> 5527de0ac126209a4aa701c4f4221bcff6c70ddf
>>>>>>> d74a0e337b8516c4db74d8e622966d39852e9476




        print("load caption file done")


    def __len__(self):
<<<<<<< HEAD
<<<<<<< HEAD
=======
        return len(self.image_files)
 
    def __getitem__(self, index):
        #name = self.name_list[index]
        #image_path = os.path.join(self.image_folder, name+'.jpg')
        # if os.path.exists(image_path):
        #     print(index, image_path)
        #print(index)
        image_path = self.image_files[index]
        image = load_img(image_path)
        #print(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']
        caption_list = self.caption_labels[index]
        caption = caption_list.replace('|', ' ').lower()
        #print(caption)
        # else:
        if self.phase == 'train' and self.extra_img != None:

            for i in range(10):
                idx = random.randint(0, self.__len__()-1)
                #neg_img_path = self.A_paths[idx % self.A_size]
                neg_img_path = self.image_files[indx % self.__len__()]
                img = load_img(neg_img_path)
                image = np.array(image).astype(np.uint8)                
=======
>>>>>>> d74a0e337b8516c4db74d8e622966d39852e9476
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
<<<<<<< HEAD
=======
        return len(self.image_files)
 
    def __getitem__(self, index):
        #name = self.name_list[index]
        #image_path = os.path.join(self.image_folder, name+'.jpg')
        # if os.path.exists(image_path):
        #     print(index, image_path)
        #print(index)
        image_path = self.image_files[index]
        image = load_img(image_path)
        #print(image_path)
        image = np.array(image).astype(np.uint8)
        image = self.transform(image = image)['image']
        caption_list = self.caption_labels[index]
        caption = caption_list.replace('|', ' ').lower()
        #print(caption)
        # else:
        if self.phase == 'train' and self.extra_img != None:

            for i in range(10):
                idx = random.randint(0, self.__len__()-1)
                #neg_img_path = self.A_paths[idx % self.A_size]
                neg_img_path = self.image_files[indx % self.__len__()]
                img = load_img(neg_img_path)
                image = np.array(image).astype(np.uint8)                
>>>>>>> 739735d6095bfcbc3ebf457b3ebddea38396e35f
=======
>>>>>>> 5527de0ac126209a4aa701c4f4221bcff6c70ddf
>>>>>>> d74a0e337b8516c4db74d8e622966d39852e9476
                img = self.transform(image = img)['image']
                if i == 0:
                    neg_img = np.expand_dims(img, axis=0)
                else:
                    img = np.expand_dims(img, axis=0)
<<<<<<< HEAD
<<<<<<< HEAD
=======
                    neg_img = np.concatenate((neg_img, img), axis=0)
            # print("check neg_img:", np.shape(neg_img))
            data = {
                    'image': np.transpose(A.astype(np.float32), (2, 0, 1)),
                    'label': A_label,
                    'negative_img': np.transpose(neg_img.astype(np.float32), (0, 3, 1, 2)),
                }   
=======
>>>>>>> d74a0e337b8516c4db74d8e622966d39852e9476
                    neg_img = np.concatenate((neg_img, img), axis=0) 
            # print("check data loader:", np.shape(image), np.shape(neg_img))
            data = {
                    'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
                    'text': caption,
                    'negative_img': np.transpose(neg_img.astype(np.float32), (0, 3, 1, 2)),
                }
<<<<<<< HEAD
=======
                    neg_img = np.concatenate((neg_img, img), axis=0)
            # print("check neg_img:", np.shape(neg_img))
            data = {
                    'image': np.transpose(A.astype(np.float32), (2, 0, 1)),
                    'label': A_label,
                    'negative_img': np.transpose(neg_img.astype(np.float32), (0, 3, 1, 2)),
                }   
>>>>>>> 739735d6095bfcbc3ebf457b3ebddea38396e35f
=======
>>>>>>> 5527de0ac126209a4aa701c4f4221bcff6c70ddf
>>>>>>> d74a0e337b8516c4db74d8e622966d39852e9476
        else:
            # neg_img = None
            data = {
                    'image': np.transpose(image.astype(np.float32), (2, 0, 1)),
                    'text': caption,
                    # 'negative_img': np.transpose(neg_img.astype(np.float32), (0, 3, 1, 2)),
                }
        
    
        return data


