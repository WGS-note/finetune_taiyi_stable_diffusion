# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/12/6 3:17 下午
# @File: speedup.py
'''
stable diffusion speedup test

model: Taiyi-Stable-Diffusion-1B-Chinese-v0.1
'''
import time
import torch
import torch.nn as nn
from torch import autocast
from diffusers import StableDiffusionPipeline, DiffusionPipeline, StableDiffusionOnnxPipeline
from diffusers import LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler

import warnings
warnings.filterwarnings("ignore")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1')

def original(prompt, save_path):

    pipe = StableDiffusionPipeline.from_pretrained(pre_model_path).to(device)

    start_time = time.time()

    image = pipe(prompt).images[0]

    image.save(save_path)

    end_time = time.time()
    print('original - Full-Precision - Inference Run Time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

'''   更换采样器: DDIM、DDPM   '''
def scheduler(Scheduler, prompt, save_path):
    # ori：PNDScheduler
    if Scheduler == 'DDIM_ori':
        scheduler = DDIMScheduler()
    elif Scheduler == 'DDIM':
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    elif Scheduler == 'DDPM_ori':
        scheduler = DDPMScheduler()
    elif Scheduler == 'DDPM':
        scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    elif Scheduler == 'LMD':
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    else:
        raise ValueError('This type is not supported')

    pipe = StableDiffusionPipeline.from_pretrained(pre_model_path, scheduler=scheduler).to(device)

    start_time = time.time()

    image = pipe(prompt).images[0]

    image.save(save_path)

    end_time = time.time()
    print('{} - Full-Precision - Inference Run Time: {:.0f}分 {:.0f}秒'.format(Scheduler, (end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

'''   cuDNN自动调优器   '''
def cudnn_auto_tuner(prompt, save_path):
    pipe = StableDiffusionPipeline.from_pretrained(pre_model_path).to(device)

    torch.backends.cudnn.benchmark = True

    start_time = time.time()

    image = pipe(prompt).images[0]

    image.save(save_path)

    end_time = time.time()
    print('original - cudnn_auto_tuner - Full-Precision - Inference Run Time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

'''   use TensorFloat32   '''
def use_tf32(prompt, save_path):
    pipe = StableDiffusionPipeline.from_pretrained(pre_model_path).to(device)

    torch.backends.cuda.matmul.allow_tf32 = True

    start_time = time.time()

    image = pipe(prompt).images[0]

    image.save(save_path)

    end_time = time.time()
    print('original - use_tf32 - Full-Precision - Inference Run Time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

'''   amp   '''
def use_amp(prompt, save_path):
    pipe = StableDiffusionPipeline.from_pretrained(pre_model_path).to(device)

    start_time = time.time()

    with autocast("cuda"):
        image = pipe(prompt).images[0]

    image.save(save_path)

    end_time = time.time()
    print('original - use_amp - Full-Precision - Inference Run Time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

'''   half precision   '''
def half_precision(prompt, save_path):
    pipe = StableDiffusionPipeline.from_pretrained(pre_model_path, revision="fp16", torch_dtype=torch.float16).to(device)

    start_time = time.time()

    image = pipe(prompt).images[0]

    image.save(save_path)

    end_time = time.time()
    print('original - Half-Precision - Inference Run Time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

'''   Memory Efficient Attention   '''
def memory_efficient_att(prompt, save_path):
    pipe = StableDiffusionPipeline.from_pretrained(pre_model_path, revision="fp16", torch_dtype=torch.float16).to(device)

    # pipe.enable_xformers_memory_efficient_attention()

    start_time = time.time()

    with torch.inference_mode():
        image = pipe(prompt).images[0]

    image.save(save_path)

    end_time = time.time()
    print('original - Memory Efficient Att- Half-Precision - Inference Run Time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

'''   amp + hp'''
def amp_hp(prompt, save_path):
    pipe = StableDiffusionPipeline.from_pretrained(pre_model_path, revision="fp16", torch_dtype=torch.float16).to(device)

    start_time = time.time()

    with autocast("cuda"):
        image = pipe(prompt).images[0]

    image.save(save_path)

    end_time = time.time()
    print('AMP - Half-Precision - Inference Run Time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

'''   cuDNN自动调优器 + HP   '''
@torch.inference_mode()
def generate1(prompt, save_path):

    # torch.backends.cudnn.benchmark = True

    pipe = StableDiffusionPipeline.from_pretrained(pre_model_path, revision="fp16", torch_dtype=torch.float16).to(device)

    start_time = time.time()

    image = pipe(prompt).images[0]

    image.save(save_path)

    end_time = time.time()
    print('original - Half-Precision - Inference Run Time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

def generate2(prompt, save_path):
    # 15s
    pipe = StableDiffusionPipeline.from_pretrained(pre_model_path, revision="fp16", torch_dtype=torch.float16, use_auth_token=True).to(device)

    start_time = time.time()

    with torch.inference_mode(), torch.autocast("cuda"):
        image = pipe(prompt).images[0]

    image.save(save_path)

    end_time = time.time()
    print('Inference Run Time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

if __name__ == '__main__':

    '''
    
    docker run -d --gpus '"device=0,1,2,3"' \
       --rm -it --name t2i_speedup \
       -v /data/wgs/text2img:/home \
       wgs-torch:5.0 \
       sh -c "python -u /home/speedup/speedup.py 1>>/home/log/speedup.log 2>>/home/log/speedup.err"
    
    '''

    pre_model_path = '/home/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1'

    prompt = '一只穿着宇航服的哈士奇'

    # original(prompt=prompt, save_path='/home/gen_imgs/speed_1_.jpg')
    #
    # scheduler(Scheduler='DDIM_ori', prompt=prompt, save_path='/home/gen_imgs/speed_2.jpg')
    # scheduler(Scheduler='DDIM', prompt=prompt, save_path='/home/gen_imgs/speed_3.jpg')
    # scheduler(Scheduler='DDPM_ori', prompt=prompt, save_path='/home/gen_imgs/speed_4.jpg')
    # scheduler(Scheduler='DDPM', prompt=prompt, save_path='/home/gen_imgs/speed_5.jpg')
    # scheduler(Scheduler='LMD', prompt=prompt, save_path='/home/gen_imgs/speed_6.jpg')
    #
    # cudnn_auto_tuner(prompt=prompt, save_path='/home/gen_imgs/speed_7_.jpg')
    #
    # use_tf32(prompt=prompt, save_path='/home/gen_imgs/speed_8_.jpg')
    #
    # use_amp(prompt=prompt, save_path='/home/gen_imgs/speed_9_.jpg')
    #
    # half_precision(prompt=prompt, save_path='/home/gen_imgs/speed_10_.jpg')

    # memory_efficient_att(prompt=prompt, save_path='/home/gen_imgs/speed_11_.jpg')

    # amp_hp(prompt=prompt, save_path='/home/gen_imgs/speed_12_.jpg')

    # generate1(prompt=prompt, save_path='/home/gen_imgs/speed_13_.jpg')

    generate2(prompt=prompt, save_path='/home/gen_imgs/speed_14_.jpg')



