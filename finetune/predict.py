# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/11/24 10:23 上午
# @File: predict.py
'''
微调太乙 Stable-Diffusion 预测
'''
import os, time, sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import os, time
import torch
import argparse
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.models.model_utils import add_module_args, configure_optimizers, get_total_steps
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from transformers import BertTokenizer, BertModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from fengshen.data.taiyi_stable_diffusion_datasets.taiyi_datasets import add_data_args, load_data

from text2img import Text2Img

import warnings
warnings.filterwarnings("ignore")

# model_path = './pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1'
# ckpt_path = './pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/ckpt/last.ckpt'
#
#
# tokenizer = BertTokenizer.from_pretrained(model_path, subfolder="tokenizer")
# text_encoder = BertModel.from_pretrained(model_path, subfolder="text_encoder")
# pipe = StableDiffusionPipeline.from_pretrained(
#         model_path, text_encoder=text_encoder, tokenizer=tokenizer,
# ).to('cuda:0')
# print('--- 1')
#
# prompt = '一直穿着宇航服的哈士奇'
# image = pipe(prompt).images[0]
# print(image)
# image.save('./gen_imgs/tmp1.png')
# print('--- 2')
#
# pipe.save_pretrained('./pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/ckpt/tmp.ckpt')
# print('--- 3')

if __name__ == '__main__':
        '''
        docker run -d --gpus '"device=2"' \
               --rm -it --name t2i_inference \
               --shm-size 12G \
               -v /data/wgs/text2img:/home \
               wgs-torch:3.1 \
               sh -c "python3 -u /home/finetune/predict.py 1>>/home/log/predict.log 2>>/home/log/predict.err"
        '''

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'

        # m = Text2Img(model_name='sd_fintune_best', use_full_precision=False, device=device)
        m = Text2Img(model_name='sd_fintune_1000', use_full_precision=True, device=device)

        m.generate(prompt='business, logo', save_path='./gen_imgs/A_business_1.png', height=320, width=320)
        m.generate(prompt='business, logo', save_path='./gen_imgs/A_business_2.png', height=320, width=320)
        m.generate(prompt='business, logo', save_path='./gen_imgs/A_business_3.png', height=320, width=320)
        m.generate(prompt='business, logo', save_path='./gen_imgs/A_business_4.png', height=320, width=320)
        m.generate(prompt='business, logo', save_path='./gen_imgs/A_business_5.png', height=320, width=320)
        m.generate(prompt='business, logo', save_path='./gen_imgs/A_business_6.png', height=320, width=320)

        m.generate(prompt='party, logo', save_path='./gen_imgs/A_party_1.png', height=320, width=320)
        m.generate(prompt='party, logo', save_path='./gen_imgs/A_party_2.png', height=320, width=320)
        m.generate(prompt='party, logo', save_path='./gen_imgs/A_party_3.png', height=320, width=320)
        m.generate(prompt='party, logo', save_path='./gen_imgs/A_party_4.png', height=320, width=320)
        m.generate(prompt='party, logo', save_path='./gen_imgs/A_party_5.png', height=320, width=320)
        m.generate(prompt='party, logo', save_path='./gen_imgs/A_party_6.png', height=320, width=320)

        m.generate(prompt='Christmas, logo', save_path='./gen_imgs/A_Christmas_1.png', height=320, width=320)
        m.generate(prompt='Christmas, logo', save_path='./gen_imgs/A_Christmas_2.png', height=320, width=320)
        m.generate(prompt='Christmas, logo', save_path='./gen_imgs/A_Christmas_3.png', height=320, width=320)
        m.generate(prompt='Christmas, logo', save_path='./gen_imgs/A_Christmas_4.png', height=320, width=320)
        m.generate(prompt='Christmas, logo', save_path='./gen_imgs/A_Christmas_5.png', height=320, width=320)
        m.generate(prompt='Christmas, logo', save_path='./gen_imgs/A_Christmas_6.png', height=320, width=320)

        m.generate(prompt='cloud computing, logo', save_path='./gen_imgs/A_cloud_computing_1.png', height=320, width=320)
        m.generate(prompt='cloud computing, logo', save_path='./gen_imgs/A_cloud_computing_2.png', height=320, width=320)
        m.generate(prompt='cloud computing, logo', save_path='./gen_imgs/A_cloud_computing_3.png', height=320, width=320)
        m.generate(prompt='cloud computing, logo', save_path='./gen_imgs/A_cloud_computing_4.png', height=320, width=320)
        m.generate(prompt='cloud computing, logo', save_path='./gen_imgs/A_cloud_computing_5.png', height=320, width=320)
        m.generate(prompt='cloud computing, logo', save_path='./gen_imgs/A_cloud_computing_6.png', height=320, width=320)

        m.generate(prompt='active, logo', save_path='./gen_imgs/A_active_1.png', height=320, width=320)
        m.generate(prompt='active, logo', save_path='./gen_imgs/A_active_2.png', height=320, width=320)
        m.generate(prompt='active, logo', save_path='./gen_imgs/A_active_3.png', height=320, width=320)
        m.generate(prompt='active, logo', save_path='./gen_imgs/A_active_4.png', height=320, width=320)
        m.generate(prompt='active, logo', save_path='./gen_imgs/A_active_5.png', height=320, width=320)
        m.generate(prompt='active, logo', save_path='./gen_imgs/A_active_6.png', height=320, width=320)


        '''
        20s
        1min 3e
        1h 180e
        24 * 180 = 4320 
        
        4*24 = 69
        '''

