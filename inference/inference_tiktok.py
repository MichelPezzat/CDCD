# ------------------------------------------
# CDCD for Dance-to-Music
# Licensed under the MIT License.
# written by Ye ZHU
# ------------------------------------------

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import cv2
import argparse
import numpy as np
import torchvision
from PIL import Image
import librosa
from librosa.core import load
from librosa.util import normalize
import soundfile as sf
import noisereduce as nr
import time


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

    def inference_generate_sample_with_condition(self, text, truncation_rate, save_root, batch_size,fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['text'] = [text]
        data_i['image'] = None
        condition = text

        str_cond = str(condition)
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

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
        for b in range(content.shape[0]):
            cnt = b
            save_base_name = '{}'.format(str(cnt).zfill(6))
            save_path = os.path.join(save_root_, save_base_name+'.png')
            im = Image.fromarray(content[b])
            im.save(save_path)


    def inference_music(self, music, motion, video, mask, truncation_rate, save_root, batch_size,fast=False):
        os.makedirs(save_root, exist_ok=True)

        data_i = {}
        data_i['music'] = music
        data_i['motion'] = motion
        data_i['video'] = video
        data_i['condiation_mask'] = mask
        data_i['negative_music'] = None
        # save_root_ = os.path.join(save_root)
        os.makedirs(save_root, exist_ok=True)
        if fast != False:
            add_string = 'r,fast'+str(fast-1)
        else:
            add_string = 'r'
        with torch.no_grad():
            model_out = self.model.generate_content(
                batch=data_i,
                filter_ratio=0.1, # ensure that it actually generate from full mask
                replicate=batch_size,
                content_ratio=0.5,
                return_att_weight=False,
                sample_type="top"+str(truncation_rate)+add_string,
            )
        content = model_out['content']
        print("Check content:", content.size())
        # file_audio = os.path.join(save_root,'generated_sample.wav')
        generated_audio = content.squeeze().detach().cpu().numpy()
        # sf.write(file_audio, generated_audio, 22050)

        return generated_audio




def beat_detect(x, sr=22050):
    onsets = librosa.onset.onset_detect(x, sr=sr, wait=1, delta=0.2, pre_avg=1, post_avg=1, post_max=1, units='time')
    n = np.ceil( len(x) / sr)
    beats = [0] * int(n)
    for time in onsets:
        beats[int(np.trunc(time))] = 1
    return beats


def beat_scores(gt, syn):
    assert len(gt) == len(syn)
    total_beats = sum(gt)
    cover_beats = sum(syn)

    hit_beats = 0
    for i in range(len(gt)):
        if gt[i] == 1 and gt[i] == syn[i]:
            hit_beats += 1

    return cover_beats/total_beats, hit_beats/total_beats




if __name__ == '__main__':
    # VQ_Diffusion = VQ_Diffusion(config='OUTPUT/pretrained_model/config_text.yaml', path='OUTPUT/pretrained_model/human_pretrained.pth')
    # VQ_Diffusion.inference_generate_sample_with_condition("a man with beard",truncation_rate=0.86, save_root="RESULT",batch_size=2,fast=2)  # fast is a int from 2 to 10
    # VQ_Diffusion.inference_generate_sample_with_condition("a beautiful smiling woman",truncation_rate=0.85, save_root="RESULT",batch_size=8)

    # VQ_Diffusion = VQ_Diffusion(config='OUTPUT/pretrained_model/config_imagenet.yaml', path='OUTPUT/pretrained_model/imagenet_pretrained.pth')
    # VQ_Diffusion.inference_generate_sample_with_class(493,truncation_rate=0.86, save_root="RESULT",batch_size=8)
    
    VQ_Diffusion = VQ_Diffusion('/data/zhuye/D2M-Diffusion/tiktok_train_cd/configs/config.yaml', path='/data/zhuye/D2M-Diffusion/tiktok_train_cd/checkpoint/last.pth')


    sr = 22050
    # music, sampling_rate = load('/data/zhuye/aistplus/audio_s6/test/gPO_sBM_c01_d11_mPO1_ch02_seg1.wav', sr=22050)
    # motion = np.load('/data/zhuye/aistplus/motion_s6/test/gPO_sBM_c01_d11_mPO1_ch02_seg1.npy')
    # video = np.load('/data/zhuye/aistplus/video_s6/test/gPO_sBM_c01_d11_mPO1_ch02_seg1.npy')
    # genre = np.load('/home/zhuye/VQ-Diffusion/synthesis/data/test_genre_s6.npy')[0]
    # gt_beats = beat_detect(music)
    # music = torch.from_numpy(music).float()#.unsqueeze(0).unsqueeze(1)
    # motion = torch.from_numpy(motion).float().unsqueeze(0)
    # video = torch.from_numpy(video).float().unsqueeze(0)
    # genre = torch.from_numpy(genre).unsqueeze(0)

    # print("check input condition:", music.size(), motion.size(), video.size(), genre)


    # # VQ_Diffusion = VQ_Diffusion('/data/zhuye/D2M-Diffusion/aist_train_v0/configs/config.yaml', path='/data/zhuye/D2M-Diffusion/aist_train_v0/checkpoint/last.pth')
    # generated_audio = VQ_Diffusion.inference_music(music.unsqueeze(0).unsqueeze(1), motion, video, genre, mask=None, truncation_rate=0.86, save_root='RESULT', batch_size=1)
    # syn_beats = beat_detect(generated_audio)
    # print("Check gt and syn beats:", gt_beats, syn_beats)
    # score_cover, score_hit = beat_scores(gt_beats, syn_beats)
    # print(score_cover, score_hit)



    ### overall scores
    testing_music = [line.rstrip() for line in open('/home/zhuye/D2M-GAN/dataset/tiktok_audio_test_segment.txt')]
    cond_motion = [line.rstrip() for line in open('/home/zhuye/D2M-GAN/dataset/tiktok_motion_test_segment.txt')]
    cond_video = [line.rstrip() for line in open('/home/zhuye/D2M-GAN/dataset/tiktok_video_test_segment.txt')]
    #genres = np.load('/home/zhuye/VQ-Diffusion_d2m/synthesis/data/test_genre.npy')
    total_cover_score = 0
    total_hit_score = 0
    start_time = time.time()
    for i, f in enumerate(testing_music):
        print(i)
        print(testing_music[i])
        print(cond_motion[i])
        print(cond_video[i])
        #start_time = time.time()
        music, sampling_rate = load(testing_music[i]) 
        motion = np.load(cond_motion[i])
        video = np.load(cond_video[i])
        #genre = genres[i]
        gt_beats = beat_detect(music)
        music = torch.from_numpy(music).float()#.unsqueeze(0).unsqueeze(1)
        motion = torch.from_numpy(motion).float().unsqueeze(0)
        video = torch.from_numpy(video).float().unsqueeze(0)
        #genre = torch.from_numpy(genre).unsqueeze(0)
        generated_audio = VQ_Diffusion.inference_music(music.unsqueeze(0).unsqueeze(1), motion, video, mask=None, truncation_rate=0.86, save_root='RESULT_tiktok', batch_size=1)
        generated_audio = nr.reduce_noise(y=generated_audio, sr=22050)
        syn_beats = beat_detect(generated_audio)
        score_cover, score_hit = beat_scores(gt_beats, syn_beats)
        total_cover_score += score_cover
        total_hit_score += score_hit
        #end_time = time.time()
        #time_cost = end_time - start_time
        print("Inference:", score_cover, score_hit)
        #print("Time cost:", time_cost)
        file_audio = 'generated_sample_' + str(i) + '.wav'
        file_audio = os.path.join('/home/zhuye/VQ-Diffusion_d2m/RESULT_tiktok', file_audio)
        sf.write(file_audio, generated_audio, 22050)
        
        
        # cmd_gt = 'ffmpeg -i ' + v_path + ' -i ' + gt_audio_path + ' -c:v copy -map 0:v:0 -map 1:a:0 ' + out_path_gt
        # cmd2 = 'ffmpeg -i ' + v_path + ' -i ' + gen_audio_path + ' -c:v copy -map 0:v:0 -map 1:a:0 ' + out_path_gen
        # subprocess.call(cmd_gt, shell=True)

    
    end_time = time.time()
    print("Time cost:", (end_time-start_time)/len(testing_music))
    print("Score Summary for cover and hit:", total_cover_score/len(testing_music), total_hit_score/len(testing_music))



    
 






