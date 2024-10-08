# ------------------------------------------
# Conditioned Discrete Contrastive Diffusion for Semantic Image Synthesis
# written By Ye Zhu
# ------------------------------------------

import torch
import math
from torch import nn
from synthesis.utils.misc import instantiate_from_config, instantiate_from_config_semantic
import time
import numpy as np
from PIL import Image
import os

from torch.cuda.amp import autocast




def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    if "ckpt_path" in config.params:
        print("Deleting the restore-ckpt path from the config...")
        config.params.ckpt_path = None
    if "downsample_cond_size" in config.params:
        print("Deleting downsample-cond-size from the config and setting factor=0.5 instead...")
        config.params.downsample_cond_size = -1
        config.params["downsample_cond_factor"] = 0.5
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
            print("Deleting the first-stage restore-ckpt path from the config...")
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
            print("Deleting the cond-stage restore-ckpt path from the config...")
    except:
        pass

    model = instantiate_from_config_semantic(config)
    if sd is not None:
        model.load_state_dict(sd)
        print("Missing Keys in State Dict")
        print("Unexpected Keys in State Dict")
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}





class Semantic_CD(nn.Module):
    def __init__(
        self,
        *,
        content_info={'key': 'image'},
        condition_info={'key': 'semantic'},
        negative_samples={'key': 'negative_img'},
        content_codec_path,
        condition_codec_path,
        diffusion_config
    ):
        super().__init__()
        self.content_info = content_info
        self.condition_info = condition_info
        self.negative_info = negative_samples
        self.content_codec = load_model_from_config(content_codec_config)
        self.condition_codec = load_model_from_config(condition_codec_config)
        self.transformer = instantiate_from_config(diffusion_config)
        self.truncation_forward = False

    def parameters(self, recurse=True, name=None):
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            names = name.split('+')
            params = []
            for n in names:
                try: # the parameters() method is not overwritten for some classes
                    params += getattr(self, name).parameters(recurse=recurse, name=name)
                except:
                    params += getattr(self, name).parameters(recurse=recurse)
            return params

    @property
    def device(self):
        return self.transformer.device

    def get_ema_model(self):
        return self.transformer

    @torch.no_grad()
    def prepare_condition(self, batch, condition=None):
        cond_key = self.condition_info['key']
        cond = batch[cond_key] if condition is None else condition
        if torch.is_tensor(cond):
            cond = cond.to(self.device)
        cond = self.condition_codec.get_tokens(cond)
        cond_ = {}
        for k, v in cond.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cond_['condition_' + k] = v
        return cond_

    @autocast(enabled=False)
    @torch.no_grad()
    def prepare_content(self, batch, with_mask=False):
        cont_key = self.content_info['key']
        cont = batch[cont_key]
        # print("check original cont size:", cont.size())
        if torch.is_tensor(cont):
            cont = cont.to(self.device)
        if not with_mask:
            cont = self.content_codec.get_tokens(cont)
        else:
            mask = batch['mask'.format(cont_key)]
            cont = self.content_codec.get_tokens(cont, mask, enc_with_mask=False)
        cont_ = {}
        for k, v in cont.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            # print("check k and v:", k, v, v.size())
            cont_['content_' + k] = v

        negative_key = self.negative_info['key']
        try:
            n_cont = batch[negative_key]
            # print("check original neg cont size:", n_cont.size())
        except:
            n_cont = None
        if n_cont != None:
            for i in range(n_cont.size()[0]):
                n_cont_i = n_cont[i,:,:,:]
                if torch.is_tensor(n_cont_i):
                    n_cont_i = n_cont_i.to(self.device)
                    n_cont_i = self.content_codec.get_tokens(n_cont_i)
                    for k, v in n_cont_i.items():
                        v = v.to(self.device) if torch.is_tensor(v) else v
                        # print("check k and v in negative samples:", k, v, v.size())
                    if i == 0:
                        negative_token = v.unsqueeze(0)
                    else:
                        negative_token = torch.cat((negative_token,v.unsqueeze(0)),0)
            cont_['negative_token'] = negative_token
        else:
            cont_['negative_token'] = None

        # print("check final neg token:", negative_token.size())


        return cont_

    @autocast(enabled=False)
    @torch.no_grad()
    def prepare_input(self, batch):
        input = self.prepare_condition(batch)
        input.update(self.prepare_content(batch))
        return input

    def p_sample_with_truncation(self, func, sample_type):
        truncation_rate = float(sample_type.replace('q', ''))
        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            import random
            if random.random() < truncation_rate:
                out = func(out, args[1], args[2], **kwards)
            return out
        return wrapper


    def predict_start_with_truncation(self, func, sample_type):
        if sample_type[-1] == 'p':
            truncation_k = int(sample_type[:-1].replace('top', ''))
            content_codec = self.content_codec
            save_path = self.this_save_path
            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                val, ind = out.topk(k = truncation_k, dim=1)
                probs = torch.full_like(out, -70)
                probs.scatter_(1, ind, val)
                return probs
            return wrapper
        elif sample_type[-1] == 'r':
            truncation_r = float(sample_type[:-1].replace('top', ''))
            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                # notice for different batches, out are same, we do it on out[0]
                temp, indices = torch.sort(out, 1, descending=True) 
                temp1 = torch.exp(temp)
                temp2 = temp1.cumsum(dim=1)
                temp3 = temp2 < truncation_r
                new_temp = torch.full_like(temp3[:,0:1,:], True)
                temp6 = torch.cat((new_temp, temp3), dim=1)
                temp3 = temp6[:,:-1,:]
                temp4 = temp3.gather(1, indices.argsort(1))
                temp5 = temp4.float()*out+(1-temp4.float())*(-70)
                probs = temp5
                return probs
            return wrapper

        else:
            print("wrong sample type")

    @torch.no_grad()
    def generate_content(
        self,
        *,
        batch,
        condition=None,
        filter_ratio = 0.5,
        temperature = 1.0,
        content_ratio = 0.0,
        replicate=1,
        return_att_weight=False,
        sample_type="top0.85r",
    ):
        self.eval()
        if condition is None:
            condition = self.prepare_condition(batch=batch)
        else:
            condition = self.prepare_condition(batch=None, condition=condition)
        
        if replicate != 1:
            for k in condition.keys():
                if condition[k] is not None:
                    condition[k] = torch.cat([condition[k] for _ in range(replicate)], dim=0)
            
        content_token = None

        if len(sample_type.split(',')) > 1:
            if sample_type.split(',')[1][:1]=='q':
                self.transformer.p_sample = self.p_sample_with_truncation(self.transformer.p_sample, sample_type.split(',')[1])
        if sample_type.split(',')[0][:3] == "top" and self.truncation_forward == False:
            self.transformer.predict_start = self.predict_start_with_truncation(self.transformer.predict_start, sample_type.split(',')[0])
            self.truncation_forward = True

        if len(sample_type.split(',')) == 2 and sample_type.split(',')[1][:4]=='fast':
            trans_out = self.transformer.sample_fast(condition_token=condition['condition_token'],
                                                condition_mask=condition.get('condition_mask', None),
                                                condition_embed=condition.get('condition_embed_token', None),
                                                content_token=content_token,
                                                filter_ratio=filter_ratio,
                                                temperature=temperature,
                                                return_att_weight=return_att_weight,
                                                return_logits=False,
                                                print_log=False,
                                                sample_type=sample_type,
                                                skip_step=int(sample_type.split(',')[1][4:]))

        else:
            trans_out = self.transformer.sample(condition_token=condition['condition_token'],
                                            condition_mask=condition.get('condition_mask', None),
                                            condition_embed=condition.get('condition_embed_token', None),
                                            content_token=content_token,
                                            filter_ratio=filter_ratio,
                                            temperature=temperature,
                                            return_att_weight=return_att_weight,
                                            return_logits=False,
                                            print_log=False,
                                            sample_type=sample_type)


        content = self.content_codec.decode(trans_out['content_token'])  #(8,1024)->(8,3,256,256)
        self.train()
        out = {
            'content': content
        }
        

        return out

    @torch.no_grad()
    def reconstruct(
        self,
        input
    ):
        if torch.is_tensor(input):
            input = input.to(self.device)
        cont = self.content_codec.get_tokens(input)
        cont_ = {}
        for k, v in cont.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cont_['content_' + k] = v
        rec = self.content_codec.decode(cont_['content_token'])
        return rec

    @torch.no_grad()
    def sample(
        self,
        batch,
        clip = None,
        temperature = 1.,
        return_rec = True,
        filter_ratio = [0, 0.5, 1.0],
        content_ratio = [1], # the ratio to keep the encoded content tokens
        return_att_weight=False,
        return_logits=False,
        sample_type="normal",
        **kwargs,
    ):
        self.eval()
        condition = self.prepare_condition(batch)
        content = self.prepare_content(batch)

        content_samples = {'input_image': batch[self.content_info['key']]}
        if return_rec:
            content_samples['reconstruction_image'] = self.content_codec.decode(content['content_token'])  

        for fr in filter_ratio:
            for cr in content_ratio:
                num_content_tokens = int((content['content_token'].shape[1] * cr))
                if num_content_tokens < 0:
                    continue
                else:
                    content_token = content['content_token'][:, :num_content_tokens]
                if sample_type == 'debug':
                    trans_out = self.transformer.sample_debug(condition_token=condition['condition_token'],
                                                        condition_mask=condition.get('condition_mask', None),
                                                        condition_embed=condition.get('condition_embed_token', None),
                                                        content_token=content_token,
                                                        filter_ratio=fr,
                                                        temperature=temperature,
                                                        return_att_weight=return_att_weight,
                                                        return_logits=return_logits,
                                                        content_logits=content.get('content_logits', None),
                                                        sample_type=sample_type,
                                                        **kwargs)

                else:
                    trans_out = self.transformer.sample(condition_token=condition['condition_token'],
                                                        condition_mask=condition.get('condition_mask', None),
                                                        condition_embed=condition.get('condition_embed_token', None),
                                                        content_token=content_token,
                                                        filter_ratio=fr,
                                                        temperature=temperature,
                                                        return_att_weight=return_att_weight,
                                                        return_logits=return_logits,
                                                        content_logits=content.get('content_logits', None),
                                                        sample_type=sample_type,
                                                        **kwargs)

                content_samples['cond1_cont{}_fr{}_image'.format(cr, fr)] = self.content_codec.decode(trans_out['content_token'])

                if return_att_weight:
                    content_samples['cond1_cont{}_fr{}_image_condition_attention'.format(cr, fr)] = trans_out['condition_attention'] # B x Lt x Ld
                    content_att = trans_out['content_attention']
                    shape = *content_att.shape[:-1], self.content.token_shape[0], self.content.token_shape[1]
                    content_samples['cond1_cont{}_fr{}_image_content_attention'.format(cr, fr)] = content_att.view(*shape) # B x Lt x Lt -> B x Lt x H x W
                if return_logits:
                    content_samples['logits'] = trans_out['logits']
        self.train() 
        output = {'condition': batch[self.condition_info['key']]}   
        output.update(content_samples)
        return output

    def forward(
        self,
        batch,
        name='none',
        **kwargs
    ):
        input = self.prepare_input(batch)
        output = self.transformer(input, **kwargs)
        return output
