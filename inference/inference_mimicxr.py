# ------------------------------------------
# Modified based on VQ-Diffusion Project.
# Licensed under the MIT License.
# ------------------------------------------

import os
import sys
import pickle
from tqdm import tqdm
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import cv2
import argparse
import numpy as np
import torchvision
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
from synthesis.utils.io import load_yaml_config
from synthesis.modeling.build import build_model
from synthesis.utils.misc import get_model_parameters_info

class VQ_Diffusion():
    def __init__(self, config, path):
        self.info = self.get_model(ema=True, model_path=path, config_path=config)
        self.model = self.info['model']
        self.epoch = self.info['epoch']
        self.model_name = self.info['model_name']
        self.model = self.model.cuda()
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad=False

    def get_model(self, ema, model_path, config_path):
        if 'OUTPUT' in model_path: # pretrained model
            model_name = model_path.split(os.path.sep)[-3]
        else: 
            model_name = os.path.basename(config_path).replace('.yaml', '')

        config = load_yaml_config(config_path)
        model = build_model(config)
        model_parameters = get_model_parameters_info(model)
        
        print(model_parameters)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")

        if 'last_epoch' in ckpt:
            epoch = ckpt['last_epoch']
        elif 'epoch' in ckpt:
            epoch = ckpt['epoch']
        else:
            epoch = 0

        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print('Model missing keys:\n', missing)
        print('Model unexpected keys:\n', unexpected)

        if ema==True and 'ema' in ckpt:
            print("Evaluate EMA model")
            ema_model = model.get_ema_model()
            missing, unexpected = ema_model.load_state_dict(ckpt['ema'], strict=False)
        
        return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}

    def inference_generate_sample_with_class(self, text, truncation_rate, save_root, batch_size,fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['label'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+'r',
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root_, save_base_name+'.jpg')
            im = Image.fromarray(content[b])
            im.save(save_path)

    def inference_generate_sample_with_condition(self, text, truncation_rate, batch_size, fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['text'] = text
        data_i['image'] = None
        condition = text
        
        str_cond = str(condition)
        #save_root_ = os.path.join(save_root, str_cond)
        #os.makedirs(save_root, exist_ok=True)
        #os.makedirs(save_root_, exist_ok=True)

        if fast != False:
            add_string = 'r,fast'+str(fast-1)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0,
                replicate=batch_size,
                content_ratio=1,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
            ) # B x C x H x W

        # save results
        content = model_out['content']
        content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
        return content
        #for b in range(content.shape[0]):
         #   cnt = b
          #  save_base_name = '{}'.format(str(cnt).zfill(6))
            #print("check str_cond:",str_cond)
           # str_cond = str_cond.replace(" ","").replace(".","_").replace("/","")
            #print("check new str_cond:",str_cond)
            #exit()
            #save_base_name = str_cond + save_base_name
            #save_path = os.path.join(save_root, str(count)+'.png')

            #im = Image.fromarray(content[b])
            #im.save(save_path)


class DummyDataset(Dataset):
    def __init__(self, df):

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

       # sample = {
          #  "id": self.df.iloc[idx]["id"],
         #   "prompt": self.df.iloc[idx]["annotated_prompt"],
        #}
        sample = {
            "id": idx,
            "prompt": self.df[idx].lower(),
        }
        return sample


if __name__ == '__main__':
    VQ_Diffusion = VQ_Diffusion(config="/home/michel/data/Text2Image/mimicxr_train/configs/config.yaml", path="/home/michel/data/Text2Image/mimicxr_train/checkpoint/last.pth")
    #VQ_Diffusion.inference_generate_sample_with_condition("no findings",truncation_rate=0.86, save_root="RESULT",batch_size=2,count=2)  # fast is a int from 2 to 10
    #VQ_Diffusion.inference_generate_sample_with_condition("a beautiful smiling woman",truncation_rate=0.85, save_root="RESULT",batch_size=8)

    #VQ_Diffusion = VQ_Diffusion(config='OUTPUT/pretrained_model/config_imagenet.yaml', path='OUTPUT/pretrained_model/imagenet_pretrained.pth')
    #VQ_Diffusion.inference_generate_sample_with_class(493,truncation_rate=0.86, save_root="RESULT",batch_size=8)
    print("Loading data...")

    #print(f"Loading CSV: {LLAVARAD_ANNOTATIONS_TEST.csv}")
    captions = [
    "Small right-sided pleural effusion",
    "No acute cardiopulmonary process",
    "Small left-sided pleural effusion",
    "Large right-sided pleural effusion",
    "Bilateral pleural effusions",
    "Large left-sided pleural effusion",]

    #df = pd.read_csv('LLAVARAD_ANNOTATIONS_TEST.csv')
    dataset = DummyDataset(captions)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=False)
    
    #df = df.drop_duplicates(subset=['annotated_prompt']).reset_index(drop=True)

        
  
    save_root = "/home/michel/data/Text2Image/mimicxr_train_cd_step_t80/syn_test"
    SYNTHETIC_PATHS = []
    #count = 0

    print("Generating Images...")
    for batch in tqdm(dataloader):
        ALL_ID = batch["id"]
        PROMPTS = batch["prompt"]
        #count += 1
        #PROMPTS = PROMPTS.lower()
        
        outputs = VQ_Diffusion.inference_generate_sample_with_condition(PROMPTS, truncation_rate=0.85,batch_size=len(PROMPTS))


        for id, output in zip(ALL_ID, outputs):
            filename = "SyntheticImg_{}.png".format(id)
            savepath = os.path.join( save_root, filename)
            image = Image.fromarray(output)
            image.save(savepath)
            SYNTHETIC_PATHS.append(filename)

    # Append the filenames to the original CSV
    df["synthetic_filename"] = SYNTHETIC_PATHS
    filename = "generations_with_metadata.csv"
    df.to_csv(os.path.join(args.savedir, filename), index=False)
    print("Saved to: ", os.path.join(args.savedir, filename))




    #num = len(captions)
    #caption_txt = open("mimicxr_input_caption_inf.txt", 'w')
    #count = 0
    #for index in tqdm(range(num)):
     #   caps = captions[index]
      #  count += 1
       # cap = caps.lower()
       # print(index,  cap)
       # caption_txt.write(cap)
       # caption_txt.write('\n')
       # VQ_Diffusion.inference_generate_sample_with_condition(cap,truncation_rate=0.85, 
        #     save_root="/home/michel/data/Text2Image/mimicxr_train_cd_step_t80/syn_test",batch_size=1, count=count)

#print("total cap count:", count)
#caption_txt.close()




