import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch
import json
import cv2
import random
import time
import argparse
import numpy as np
import pandas as pd

import warnings
import torchvision
import glob
from PIL import Image, ImageOps
from collections import OrderedDict, defaultdict
from torch.utils.data import ConcatDataset
from datasets import load_dataset
import torchvision.transforms as transforms

from synthesis.utils.io import load_yaml_config
from synthesis.utils.misc import instantiate_from_config
#from synthesis.utils.cal_metrics import get_PSNR, get_mse_loss, get_l1_loss, get_SSIM
from synthesis.modeling.build import build_model
from synthesis.utils.misc import format_seconds
from synthesis.distributed.launch import launch
from synthesis.distributed.distributed import reduce_dict, synchronize, all_gather
from synthesis.utils.misc import get_model_parameters_info, get_model_buffer

def image_post_process(image):
    def convert(t):
        t = np.asarray(t)
        t = t.clip(0, 255).astype(np.uint8)
        return t
    if isinstance(image, (torch.Tensor, np.ndarray)):
        image = convert(image)
    elif isinstance(image, dict):
        for k in image:
            image[k] = convert(image[k])
    elif isinstance(image, list):
        for k in range(len(image)):
            image[k] = convert(image[k])
    else:
        raise ValueError

    return image

def image_pre_process(image, data = None):
    #image_tensor = np.array(image).astype(np.uint8)
    image_tensor = data.transform(image = image)['image']
    #print(image_tensor.shape)
    #image_tensor = np.transpose(image_tensor, (2,0,1)) if len(image_tensor.shape)==3 else image_tensor
    image_tensor = transforms.ToTensor()(image_tensor) 
    #image_tensor = image_tensor*255
    #print(image_tensor.shape)
    return image_tensor

def save_list_results(images, save_root, batch_idx, local_rank=0):
    '''
    images: list of tensors, each tensor is with [N, C, H, W]
    '''
    os.makedirs(save_root, exist_ok=True)
    N, _, H, W = images[0].shape
    images = [im.permute(0, 2, 3, 1).numpy() for im in images]
    for i in range(N):
        file_name = '{:6d}_rank{}.png'.format(batch_idx*N + i, local_rank)
        im = Image.fromarray(image_post_process(images[0][i]))
        for j in range(1, len(images)):
            im_tmp = Image.new(im.mode, ((j+1)*W, H)) 
            im_tmp.paste(im, box=(0,0))
            im_tmp.paste(Image.fromarray(image_post_process(images[j][i])), box=(j*W, 0))
            im = im_tmp
        im.save(os.path.join(save_root, file_name))
        print('saved in {}'.format(os.path.join(save_root, file_name)))

def save_image_dict(images, save_dir, batch_idx, local_rank=0, make_grid=True, ignored_keys=None, suffix=None):
    for k, v in images.items():
        if ignored_keys is not None and k in ignored_keys:
            continue
        if suffix is None:
            save_dir_ = os.path.join(save_dir, k)
        else:
            save_dir_ = os.path.join(save_dir, k+'_'+suffix)
        os.makedirs(save_dir_, exist_ok=True)
        save_path = os.path.join(save_dir_, '{:06d}_rank{}'.format(batch_idx, local_rank))
        if torch.is_tensor(v) and v.dim() == 4 and v.shape[1] in [1, 3]: # image
            im = v
            im = im.to(torch.uint8) # N x 3 x H x W

            # save images
            if make_grid:
                im_grid = torchvision.utils.make_grid(im)
                im_grid = im_grid.permute(1, 2, 0).to('cpu').numpy()
                im_grid = Image.fromarray(im_grid)

                im_grid.save(save_path + '.png')
                print('save {} to {}'.format(k, save_path+'.png'))
            else:
                if v.shape[1] == 3: # for this, we only save generated images
                    for i in range(im.shape[0]):
                        # import pdb; pdb.set_trace()
                        im_ = im[i].permute(1, 2, 0).to('cpu').numpy()
                        im_ = Image.fromarray(im_)
                        save_path_ = save_path + '_{}.png'.format(i)
                        im_.save(save_path_)
                    print('save {} to {}'.format(k, save_path_))

        else: # may be other values, such as 
            with open(save_path+'.txt', 'a') as f:
                f.write(str(v)+'\n')
                f.close()
            print('save {} to {}'.format(k, save_path+'.txt'))


def save_image_pair(images, save_root, batch_idx, local_rank=0):
    '''
    images: list or tuple, each element in it is a tensor, [N, C, H, W]
    '''
    os.makedirs(save_root, exist_ok=True)
    
    for i in range(len(images)):
        im_grid_tmp = torchvision.utils.make_grid(images[i], nrow=images[i].shape[0]) # 3, H, W
        if i == 0:
            im = im_grid_tmp
        else:
            im = torch.cat((im, im_grid_tmp), dim=1)

    # im1_grid = torchvision.utils.make_grid(images1, nrow=images1.shape[0]) # 3, H, W
    # im2_grid = torchvision.utils.make_grid(images2, nrow=images1.shape[0]) # 3, H, W
    # im = torch.cat((im1_grid, im2_grid), dim=1)
    im = im.permute(1, 2, 0).to('cpu').numpy().astype(np.uint8)
    im = Image.fromarray(im)
    file_name = '{}_rank{}.png'.format(str(batch_idx).zfill(6), local_rank)
    im.save(os.path.join(save_root, file_name))
    print('saved in {}'.format(os.path.join(save_root, file_name)))


def get_model_and_dataset(args=None, model_name='2020-11-09T13-33-36_faceshq_vqgan'):
    if os.path.isfile(model_name):
        # import pdb; pdb.set_trace()
        if model_name.endswith(('.pth', '.ckpt')):
            model_path = model_name
            config_path = os.path.join(os.path.dirname(model_name), '..', 'configs', 'config.yaml')
        elif model_name.endswith('.yaml'):
            config_path = model_name
            model_path = os.path.join(os.path.dirname(model_name), '..', 'checkpoint', 'last.pth')
        else:
            raise RuntimeError(model_name)
        
        if 'OUTPUT' in model_name: # pretrained model
            model_name = model_path.split(os.path.sep)[-3]
        else: # just give a config file, such as test_openai_dvae.yaml, which is no need to train, just test
            model_name = os.path.basename(config_path).replace('.yaml', '')
    else:
        model_path = os.path.join( model_name, 'checkpoint', 'last.pth')
        config_path = os.path.join(os.path.join( model_name, 'configs', 'config.yaml'))

    args.model_path = model_path
    args.config_path = config_path

    config = load_yaml_config(config_path)
    model = build_model(config)
    model_parameters = get_model_parameters_info(model)
    # import pdb; pdb.set_trace()
    print(model_parameters)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu")
    else:
        ckpt = {}
    if 'last_epoch' in ckpt:
        epoch = ckpt['last_epoch']
    elif 'epoch' in ckpt:
        epoch = ckpt['epoch']
    else:
        epoch = 0

    if 'model' in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    elif 'state_dict' in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        missing, unexpected = [], []
        print("====> Warning! No pretrained model!")

    print('Model missing keys:\n', missing)
    print('Model unexpected keys:\n', unexpected)

    if args is not None and args.ema and 'ema' in ckpt:
        print("Evaluate EMA model")
        # import pdb; pdb.set_trace()
        if hasattr(model, 'get_ema_model') and callable(model.get_ema_model):
            ema_model = model.get_ema_model()
            # missing, unexpected = model.get_ema_model().load_state_dict(ckpt['ema'], strict=False)
        else:
            ema_model = model
            # missing, unexpected = model.load_state_dict(ckpt['ema'], strict=False)
        
        if args.ema_no_buffer:
            ema_param = OrderedDict()
            for n, p in ema_model.named_parameters():
                ema_param[n] = ckpt['ema'][n]
            missing, unexpected = ema_model.load_state_dict(ema_param, strict=False)
            skipped_buffer_name = []
            for k in ckpt['ema'].keys():
                if k not in ema_param:
                    skipped_buffer_name.append(k)
            if len(skipped_buffer_name) == 0:
                raise ValueError('No buffer found in ema model, please set args.ema_no_buffer to False')
            print('EMA model skipped buffer:\n', skipped_buffer_name)
        else:
            missing, unexpected = ema_model.load_state_dict(ckpt['ema'], strict=False)

    data_type = 'validation_datasets'
    if args is not None and args.data_type == 'train':
        data_type = 'train_datasets'

    val_dataset = []
    for ds_cfg in config['dataloader'][data_type]:
        ds = instantiate_from_config(ds_cfg)
        val_dataset.append(ds)
    if len(val_dataset) > 1:
        val_dataset = ConcatDataset(val_dataset)
    else:
        val_dataset = val_dataset[0]
    
    return {'model': model, 'data': val_dataset, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}



def caculate_flops_and_params(local_rank=0, args=None):
    from thop import profile, clever_format
    info = get_model_and_dataset(args=args, model_name=args.name)
    model = info['model']
    data = info['data']
    
    model = model.cuda()
    model.train()
    dataloader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=1, shuffle=False, drop_last=False)
    for itr, batch in enumerate(dataloader):
        input = {
                'batch': batch,
                'return_loss': True,
                }
        # import pdb; pdb.set_trace()
        macs, params = profile(model, inputs=(batch,))
        
        params = torch.DoubleTensor([0]).to(model.device)
        for p in model.parameters():
            params += p.nelement()
            
        macs, params = clever_format([macs, params], "%.3f")
        break
    print('rank: {},'.format(local_rank), macs, params)


def inference_generate_sample_with_condition(local_rank=0, args=None):
    info = get_model_and_dataset(args=args, model_name=args.name)
    model = info['model']
    data = info['data']
    epoch = info['epoch']
    model_name = info['model_name']

    if args is not None and args.distributed:
        print('DDP')
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=False)
        dataloader = torch.utils.data.DataLoader(data, 
                                             batch_size=1, 
                                             shuffle=False,
                                             num_workers=1, 
                                             pin_memory=True, 
                                             sampler=sampler, 
                                             drop_last=True)
    else:
        model = model.cuda()
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

        try:
            print("Lt count is :")
            print([float('{:.2f}'.format(i)) for i in torch.sqrt(model.transformer.Lt_count).tolist()])
            print("Lt history is :")
            print([float('{:.2f}'.format(i)) for i in torch.sqrt(model.transformer.Lt_history).tolist()])
        except:
            print(' Lt_count or Lt_history is not provided')
        else:
            pass

    ####################################################
    model.eval()
    for param in model.parameters(): 
        param.requires_grad=False
    #####################################################

    full_data = load_dataset("raman07/SynthCheX-75K", trust_remote_code=True)
    full_data = [full_data['train'][i] for i in range(len(full_data['train'])) if full_data['train'][i]['labels_dict']['No Finding']==1] 
    full_data = full_data[10000:10700]

    
    # count_cond = 100 // (args.world_size if args is not None else 1)
    count_cond = 800                                  # how many text
    # count_per_cond = 
    return_att_weight = False # True #False
    filter_ratio = [0.0] #[0.3, 0.5, 50, 100]
    count_per_cond = len(filter_ratio) * args.batch_size
    content_ratio = [1.0] #, 0.5] # [0.0, 0.0, 0.3, 0.7]
    chosed_fr = []
    chosed_cr = []

    # save images
    save_root = os.path.join(args.save_dir, model_name+'_{}'.format(args.data_type))
    # import pdb; pdb.set_trace()
    if args.ema:
        if args.ema_no_buffer:
            save_root += '_emaNoBuffer'
        else:
            save_root += '_ema'
    save_root = save_root + '_e{}_generate_text'.format(epoch) + "_" + args.sample_type

    os.makedirs(save_root, exist_ok=True)
    print('results will be saved in {}'.format(save_root))
   
    # temp_image_path = "scripts/elephant.jpg"


    #mask_image = Image.open(args.mask_path).convert('L')
    #mask_tensor = transforms.ToTensor()(mask_image)
    #assert mask_tensor.size()[0] == 1
    #assert mask_tensor.size()[1] == 256
    #assert mask_tensor.size()[2] == 256

    #mask = (mask_tensor>0).unsqueeze(0).expand(-1,3,-1,-1).float()
    #out_tensor = image_tensor0*(1-mask)+mask*0.35
    #out_image = transforms.ToPILImage()(image_tensor.squeeze(0))
    #this_out_path = args.image_path.replace('.jpg', '_mask.jpg')
    #out_image.save(this_out_path)

    import torch.nn.functional as F
    #mask_token = F.interpolate(mask_tensor.unsqueeze(0), size=[32,32])
    

    start_gen = time.time()
    count_cond = min(count_cond, len(full_data))
    labels_dict_edited = []
    labels_dict_org = []
    paths_org = []
    paths_edited = []
    for i in range(len(full_data)):
    # for i, data_i in enumerate(dataloader):
        if i >= count_cond:
            break
        else:
            print("number of count is {}/{} -------".format(i, count_cond))

        condition_info = model.condition_info if not isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.module.condition_info
        current_image_data = full_data[i]
        
        #temp_image = Image.open(temp_image_path).convert('RGB')
        orig_img = current_image_data['image']
        image_tensor0 = image_pre_process(orig_img, data = data).unsqueeze(0)
        image_tensor = image_tensor0*255

        label_dict = current_image_data['labels_dict']
        labels_dict_org.append(label_dict)
        label_dict['No Finding'] = 1
        labels_dict_edited.append(label_dict)


        text =  'No acute cardiopulmonary process. Unremarkable chest radiographic examination' # default empty string
     
        data_i = {}
        data_i['text'] = [text]
        data_i['image'] = image_tensor
        data_i['edit_text'] = 'Left pectoral pacemarker in place. The position of the leads is as expected. Otherwise unremarkable chest radiographic examination' 
        condition = text

        mask_path = os.path.join( args.mask_paths, 'img_'+f"{i:03d}"+'_mask.png') 
        mask_image = Image.open(mask_path).convert('L')
        #mask_image = ImageOps.invert(mask_image)
        mask_tensor = image_pre_process(mask_image, data = data)
        #print(mask_tensor.size())
        assert mask_tensor.size()[0] == 1
        assert mask_tensor.size()[1] == 256
        assert mask_tensor.size()[2] == 256


        
        mask = (mask_tensor>0).unsqueeze(0).expand(-1,3,-1,-1).float()
        out_tensor = image_tensor0*(1-mask)+mask*0.35
        out_image = transforms.ToPILImage()(out_tensor.squeeze(0))


        mask_token = F.interpolate(mask_tensor.unsqueeze(0), size=[32,32])        

        # condition = 'a cartoon illustration of a yellow devil'
        # # condition = 'bride at the vector art illustration'
        # # condition = 'a photo of cat.'
        # # condition = 'it is an apple!'
        # data_i[condition_info['key']][0] = condition

        if torch.is_tensor(condition):
            if condition.numel() == 1:
                str_cond = str(condition.view(-1).numpy()[0])
            else:
                str_cond = str(condition)
        else:
            if condition[-1] == '.':
                condition = condition[:-1]
            str_cond = str(condition)
        
        save_root_ = os.path.join(save_root, str_cond)
        os.makedirs(save_root_, exist_ok=True)

        # save_condition
        with open(os.path.join(save_root_, 'condition.txt'), 'w') as fc:
            fc.write(str_cond)
            fc.close()

        # generate samples in a batch manner
        count_per_cond_ = 0
        if count_per_cond_ < count_per_cond:
            assert len(content_ratio) == 1
            cr = content_ratio[0]
            for fr in filter_ratio:
                start_batch = time.time()
                if True:
                    model_out = model.generate_content(
                        batch=data_i,
                        filter_ratio=fr,
                        replicate=args.batch_size,
                        content_ratio=cr,
                        return_att_weight=return_att_weight,
                        sample_type=args.sample_type,
                        mask_token=mask_token,
                    ) # B x C x H x W
                    # model_out = model.sample(data_i, return_rec=False, filter_ratio=fr_, content_ratio=cr_)

            
                # save results
                content = model_out['content']
                content = content.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
                for b in range(content.shape[0]):
                    cnt = count_per_cond_ + b


                
                    save_base_name = 'rank_{}_{}_fr{}_cr{}'.format(local_rank, str(cnt).zfill(6), fr, cr)
                    save_path = os.path.join(save_root_, save_base_name+str(i)+'edit.png')
                    im = Image.fromarray(content[b])
                    im.save(save_path)

                    save_path_org = os.path.join(save_root_,save_base_name+str(i)+'orig.jpg')
                    orig_img.save(save_path_org)


                    paths_org.append(save_path_org)
                    paths_edited.append(save_path)
                    print('Rank {}, Total time {}, batch time {:.2f}s, saved in {}'.format(local_rank, format_seconds(time.time()-start_gen), time.time()-start_batch, save_path))

                    return_att_weight = False
                    if return_att_weight == True:
                        att_save_dir = os.path.join(save_root_, save_base_name + '_attention')
                        os.makedirs(att_save_dir, exist_ok=True)
                        condition_attention = model_out['condition_attention'].to('cpu') # B x Lt x Ld
                        content_attention = model_out['content_attention'].to('cpu') # B x Lt x H x W
                        cond_att_save_path = os.path.join(att_save_dir, 'condition_attention')
                        cont_att_save_path = os.path.join(att_save_dir, 'content_attention')
                        torch.save(condition_attention, cond_att_save_path+'.pth')
                        torch.save(content_attention, cont_att_save_path+'.pth')
                    
                        cond_att_f = open(cond_att_save_path+'.txt', 'w')
                        cont_att_f = open(cont_att_save_path+'.txt', 'w')

                        for cont_idx in range(content_attention.shape[1]):
                            cond_att_f.write(str(cont_idx)+'\n'+str(condition_attention[b, cont_idx, :])+'\n')
                            cont_att_f.write(str(cont_idx)+'\n'+str(content_attention[b, cont_idx])+'\n')
                            # save content attention as image

                            cont_att_im = (content_attention[b, cont_idx]/content_attention[b, cont_idx].max() * 255).numpy().astype(np.uint8)
                            cont_att_im = Image.fromarray(cont_att_im)
                            cont_att_im.save(os.path.join(att_save_dir, '{}_content_attention.png'.format(cont_idx)))
                        cond_att_f.close()
                        cont_att_f.close()     
            
                print('==> batch time {}s'.format(round(time.time() - start_batch, 1)))
    
                count_per_cond_ = len(glob.glob(os.path.join(save_root_, 'rank_*_*_fr*_cr*.png')))
    origin_images_info = {}
    origin_images_info['paths']=paths_org
    origin_images_info['chexpert_labels']=labels_dict_org
    
    edited_images_info = {}
    edited_images_info['paths']=paths_edited
    edited_images_info['chexpert_labels']=labels_dict_edited
    
    df = pd.DataFrame(origin_images_info)
    df.to_csv('origin_images_dice_PM_nm.csv', index=False)

    df = pd.DataFrame(edited_images_info)
    df.to_csv('edited_images_dice_PM_nm.csv', index=False)    

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--save_dir', type=str, default='RESULT_inpaint', 
                        help='directory to save results') 

    parser.add_argument('--name', type=str, default='/home/michel/data/Text2Image/mimicxr_train_100/', 
                        help='the name of this experiment, if not provided, set to'
                             'the name of config file') 
    parser.add_argument('--func', type=str, default='inference_generate_sample_with_condition', 
                        help='the name of inference function') 
    # args for ddp
    parser.add_argument('--num_node', type=int, default=1,
                        help='number of nodes for distributed training')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default='auto', 
                        help='url used to set up distributed training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    parser.add_argument('--batch_size', type=int, default=1,  
                        help='batch size while inference')         # by default is 8
    parser.add_argument('--data_type', type=str, default='val',
                        choices=['val', 'train'],
                        help='evaluate ema model')                       
    parser.add_argument('--ema', action='store_true', default=False,
                        help='evaluate ema model')
    parser.add_argument('--ema_no_buffer', action='store_true', default=False,
                        help='upadte buffers in ema model')

    parser.add_argument('--debug', action='store_true', # default=True,
                        help='set as debug mode')
    parser.add_argument('--sample_type', type=str, default='top0.85r,edit', help='normal|top1|top3|...|debug')
    parser.add_argument('--txt_file', type=str, default="mimicxr_input_caption_inf.txt", help='txt file')
    parser.add_argument('--image_path', type=str, default="/home/michel/data/Text2Image/mimicxr_train_cd_step_t80/syn_test", help='input_image')
    parser.add_argument('--mask_paths', type=str, default='/home/michel/data/Masks/only_sane', help='input_mask')
    parser.add_argument('--caption', type=str, default='No acute cardiopulmonary process', help='input_mask')

    args = parser.parse_args()
    args.cwd = os.path.abspath(os.path.dirname(__file__))

    # modify args for debugging
    if args.debug:
        args.name = 'debug'
        if args.gpu is None:
            args.gpu = 0

    return args


inference_func_map = {
    # 'inference_reconstruction': inference_reconstruction,
    'inference_generate_sample_with_condition': inference_generate_sample_with_condition,
}


if __name__ == '__main__':
    args = get_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable ddp.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        if args.num_node == 1:
            args.dist_url == "auto"
        else:
            assert args.num_node > 1
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node

    args.distributed = args.world_size > 1

    assert args.name != ''
    # if args.name == '':
    #     args.name = 'OUTPUT/dalle_d24h16_PredCond_DalleTextEmbedding_gcc_lr3e-6none_Warmup4.5e-4_plateau_ema_g32/checkpoint/000016e.pth'

    # import pdb; pdb.set_trace()
    if args.func == 'caculate_flops_and_params':
        args.gpu = 0
    launch(inference_func_map[args.func], args.ngpus_per_node, args.num_node, args.node_rank, args.dist_url, args=(args,))

