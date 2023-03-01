# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2023/2/6 10:08
# @File: demo1.py
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import argparse
from transformers import BertTokenizer, BertModel
import cv2
from torchvision import utils as vutils
import numpy as np

'''----------------------------------------------------------------------------------------------------'''

def add_data_args(parent_args):
    parser = parent_args.add_argument_group('taiyi stable diffusion data args')
    # 支持传入多个路径，分别加载
    parser.add_argument(
        "--datasets_path", type=str, default=None, required=True, nargs='+',
        help="A folder containing the training data of instance images.",
    )
    # csv暂不支持，还没修改
    parser.add_argument(
        "--datasets_type", type=str, default=None, required=True, choices=['txt', 'csv'], nargs='+',
        help="dataset type, txt or csv, same len as datasets_path",
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", default=False,
        help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--thres", type=float, default=0.2,)
    return parent_args

class TXTDataset(Dataset):

    def __init__(self, foloder_name, tokenizer, thres=0.2, size=128, center_crop=False):
        print(f'Loading folder data from {foloder_name}.')
        self.image_paths = []
        self.tokenizer = tokenizer

        # 这里都存的是地址，避免初始化时间过多。
        for each_file in os.listdir(foloder_name):
            if each_file.endswith('.jpg'):
                self.image_paths.append(os.path.join(foloder_name, each_file))
            elif each_file.endswith('.png'):
                self.image_paths.append(os.path.join(foloder_name, each_file))
            elif each_file.endswith('.txt'):
                self.caption_path = os.path.join(foloder_name, each_file)

        # 拿出prompt
        self.caption_dict = {}
        with open(self.caption_path, 'r') as f:
            for line in f.readlines():
                k, v = eval(line).popitem()
                self.caption_dict[k] = v

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        print('Done loading data. Len of images:', len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])   # ./data/naerdataset/game/006.jpg
        example = {}
        instance_image = Image.open(img_path)

        if not instance_image.mode == "RGB":
            instance_image = cv2.cvtColor(np.asarray(instance_image), cv2.COLOR_RGBA2RGB)
            instance_image = 255 - instance_image * 3
            instance_image = Image.fromarray(instance_image)

        # # 通过裁剪去水印！裁掉1/10的图片。
        # instance_image = instance_image.crop(
        #     (0, 0, instance_image.size[0], instance_image.size[1] - instance_image.size[1] // 10))

        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            self.caption_dict[img_path.split('/')[-1]],
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example

def process_pool_read_txt_dataset(args, input_root=None, tokenizer=None, thres=0.2):
    root_path = input_root
    p = ProcessPoolExecutor(max_workers=24)
    # 此处输入为文件夹，图片按类型划分
    all_folders = os.listdir(root_path)
    all_datasets = []
    res = []

    for filename in all_folders:
        if filename == '.DS_Store':
            continue
        if filename[-3:] == 'txt':
            continue

        each_folder_path = os.path.join(root_path, filename)   # ./data/naerdataset/game
        res.append(p.submit(TXTDataset, each_folder_path, tokenizer, thres, args.resolution, args.center_crop))

    p.shutdown()
    for future in res:
        all_datasets.append(future.result())
    dataset = ConcatDataset(all_datasets)
    return dataset

def load_data(args, tokenizer):

    assert len(args.datasets_path) == len(args.datasets_type), "datasets_path num not equal to datasets_type"
    all_datasets = []
    for path, type in zip(args.datasets_path, args.datasets_type):
        # print('---', path, type)   # --- ./data/naerdataset/ txt
        if type == 'txt':
            all_datasets.append(process_pool_read_txt_dataset(args, input_root=path, tokenizer=tokenizer, thres=args.thres))
        else:
            raise ValueError('unsupport dataset type: %s' % type)

    return {'train': ConcatDataset(all_datasets)}

if __name__ == '__main__':
    '''
    sh ./dk/demo1.sh
    '''

    args_parser = argparse.ArgumentParser()
    args_parser = add_data_args(args_parser)
    args = args_parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1',
                                              subfolder="tokenizer")
    datasets = load_data(args, tokenizer=tokenizer)  # <class 'torch.utils.data.dataset.ConcatDataset'>

    for step, data in enumerate(DataLoader(datasets['train'])):
        print(step, data['instance_images'].shape, len(data['instance_prompt_ids']))
        print('==========')







