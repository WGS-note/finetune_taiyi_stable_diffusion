# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2023/2/20 21:12
# @File: demo2.py
import torch
from multiprocess.pool import ThreadPool
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import argparse
from transformers import BertTokenizer, BertModel
from torchvision import utils as vutils
from transformers import BertTokenizer, BertModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler
from torch.nn import functional as F

import cv2
import numpy as np
import matplotlib.pyplot as plt

def demo1():
    '''

    :return:
    '''

    # model_path = '/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/SD-diffuser-icons'
    model_path = '/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1'
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", ignore_mismatched_sizes=True)

    tmp = torch.randn(size=(9, 4, 320, 320))
    res = vae.encode(tmp).latent_dist.sample()

    print(vae)

    print(res.shape)

def split_image(src_path, rownum, colnum, file=None):
    img = cv2.imread(src_path)

    # cv2.imshow('', img)

    # cv2.imwrite(path, img)
    size = img.shape[0:2]
    w = size[1]
    h = size[0]
    # print(file, w, h)
    # 每行的高度和每列的宽度
    row_height = h // rownum
    col_width = w // rownum
    for i in range(rownum):
        for j in range(colnum):
            save_path = '/Users/wangguisen/Documents/test/{}'.format(str((i+1)*(j+1)) + '.jpg')
            row_start = j * col_width
            row_end = (j+1) * col_width
            col_start = i * row_height
            col_end = (i+1) * row_height
            # print(row_start, row_end, col_start, col_end)
            # cv2图片： [高， 宽]
            child_img = img[col_start:col_end, row_start:row_end]
            if child_img.size == 0:
                continue
            cv2.imwrite(save_path, child_img)

if __name__ == '__main__':
    '''
    
    '''

    # 可以遍历文件夹
    # file_path = r'我是路径（文件夹路径）'
    # for file in file_names:
    # src_path 具体图片路径，包含后缀
    src_path = '/Users/wangguisen/Documents/test/Furniture.jpg'
    row = 4
    col = 100
    split_image(src_path, row, col)



