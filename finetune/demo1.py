# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2023/2/6 10:08
# @File: demo1.py
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

import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    parser.add_argument('--dataloader_workers', default=2, type=int)
    parser.add_argument('--train_batchsize', default=16, type=int)
    parser.add_argument('--val_batchsize', default=16, type=int)
    parser.add_argument('--test_batchsize', default=16, type=int)
    parser.add_argument("--thres", type=float, default=0.2,)
    return parent_args

class TXTDataset(Dataset):

    def __init__(self, foloder_name, tokenizer, thres=0.2, size=512, center_crop=False):
    # def __init__(self, foloder_name, tokenizer, thres=0.2, size=128, center_crop=False):
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

        '''   !!!   '''
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
        print(img_path)
        example = {}
        instance_image = Image.open(img_path)

        # if not instance_image.mode == "RGB":
        #     instance_image = instance_image.convert("RGB")

        # print('---', idx, ' --- ', instance_image.size)

        ''' !!! '''
        # # 通过裁剪去水印！裁掉1/10的图片。
        # instance_image = instance_image.crop(
        #     (0, 0, instance_image.size[0], instance_image.size[1] - instance_image.size[1] // 10))

        example["instance_images"] = self.image_transforms(instance_image)

        # if list(example["instance_images"].shape) != [3, 200, 200]:
        #     print('---', self.caption_dict[img_path.split('/')[-1]], ' --- ', example["instance_images"].shape)
        #     # ppp = '/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/data/svg_data/{}'.format(img_path.split('/')[-1])
        #     # vutils.save_image(example["instance_images"], ppp, normalize=False)

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

def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    # 生成一个三维画布图片
    rgb = np.zeros((row, col, 3), dtype='float32')

    # 获取图片每个通道数据
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    # 把 alpha 通道的值转换到 0-1 之间
    a = np.asarray(a, dtype='float32') / 255.0

    # 得到想要生成背景图片每个通道的值
    R, G, B = background

    # 将图片 a 绘制到另一幅图片 b 上，如果有 alpha 通道，那么最后覆盖的结果值将是 c = a * alpha + b * (1 - alpha)
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    # 把最终数据类型转换成 uint8
    return np.asarray(rgb, dtype='uint8')


def weight_transform(newmodel, old_weight_path, new_weight_savepath):
    old_state = torch.load(str(old_weight_path))
    if hasattr(old_state, "state_dict"):
        old_state = {key.replace('module.', ''): value for key, value in old_state.state_dict().items()}
    else:
        old_state = {key.replace('module.', ''): value for key, value in old_state.items()}

    new_state = newmodel.state_dict()

    new_dict = {}
    oldkeys = old_state.keys()
    newkeys = new_state.keys()

    pair_keys = {
        'encode1.0.weight': 'out.0.weight',
        'encode1.0.bias': 'out.0.bias',
        'encode1.1.weight': 'out.1.weight',
        'encode1.1.bias': 'out.1.bias',
        'encode1.1.running_mean': 'out.1.running_mean',
        'encode1.1.running_var': 'out.1.running_var',
        'encode1.1.num_batches_tracked': 'out.1.num_batches_tracked',
        'encode2.0.weight': 'out.4.weight',
        'encode2.0.bias': 'out.4.bias',
        'encode2.1.weight': 'out.5.weight',
        'encode2.1.bias': 'out.5.bias',
        'encode2.1.running_mean': 'out.5.running_mean',
        'encode2.1.running_var': 'out.5.running_var',
        'encode2.1.num_batches_tracked': 'out.5.num_batches_tracked',
        'out1.0.weight': 'out.7.weight',
        'out1.0.bias': 'out.7.bias'
    }
    #
    for key in newkeys:
        if key in oldkeys and 'outconv' not in key:
            new_dict[key] = old_state[key]
        else:
            new_dict[key] = new_state[key]

            new_dict[key][0] = old_state[key][0]
            new_dict[key][1] = old_state[key][1]
            new_dict[key][2] = old_state[key][2]
            new_dict[key][3] = old_state[key][2]
            new_dict[key][4] = old_state[key][3]

    torch.save(new_dict, new_weight_savepath)

def demo1():
    img_path = '/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/data/svg_data/1_Business/6.png'

    instance_image = Image.open(img_path)

    print(instance_image)

    exit()

    if not instance_image.mode == "RGB":
        instance_image = cv2.cvtColor(np.asarray(instance_image), cv2.COLOR_RGBA2RGB)

        print(instance_image.shape)

        cv2.imwrite('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/data/svg_data/11.png', instance_image)

        instance_image = 255 - instance_image * 3

        cv2.imwrite('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/data/svg_data/22.png', instance_image)

    instance_image = Image.fromarray(instance_image)

    image_transforms = transforms.Compose(
        [
            transforms.Resize(200, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(200) if False else transforms.RandomCrop(200),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    instance_image = image_transforms(instance_image)

    ppp = '/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/data/svg_data/33.png'
    vutils.save_image(instance_image, ppp, normalize=False)





if __name__ == '__main__':
    '''
    sh ./dk/demo1.sh
    '''

    # for i in range(1, 116):
    #     print("{" + "'{}.png': ".format(i) + "'wedding, logo'}")

    demo1()
    exit()

    # instance_image = Image.open('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/data/svg_data/1_Business/3.png')
    # # instance_image = Image.open('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/data/svg_data/38_Wedding/1.jpg')
    # print('---', instance_image.size, instance_image.mode)
    # print(instance_image)

    im = cv2.imread('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/data/svg_data/1_Business/10.png')
    print(im.shape)

    new_im = 255 - im * 3
    print(new_im.shape)
    # cv2.imwrite('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/data/svg_data/2.png', new_im)

    new_im = Image.fromarray(new_im)
    print('---', new_im.size)

    # if not instance_image.mode == "RGB":
    #     # instance_image = instance_image.convert("RGB")
    #     instance_image = 255 - instance_image * 3
    #
    # print('---', instance_image.size)
    # #
    image_transforms = transforms.Compose(
        [
            transforms.Resize(200, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(200) if False else transforms.RandomCrop(200),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    instance_image = image_transforms(new_im)
    print('---', instance_image.shape)
    # cv2.imwrite('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/data/svg_data/2.png', instance_image)

    # #
    # # print('---', instance_image.shape)
    # #
    ppp = '/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/data/svg_data/1.png'
    vutils.save_image(instance_image, ppp, normalize=False)
    # instance_image.save(ppp)
    #
    # print()

    # args_parser = argparse.ArgumentParser()
    # args_parser = add_data_args(args_parser)
    # args = args_parser.parse_args()
    #
    # tokenizer = BertTokenizer.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1',
    #                                           subfolder="tokenizer")
    # datasets = load_data(args, tokenizer=tokenizer)  # <class 'torch.utils.data.dataset.ConcatDataset'>
    #
    # for step, data in enumerate(DataLoader(datasets['train'])):
    #     pass
    #     # print(step, data['instance_images'].shape, len(data['instance_prompt_ids']))
    #     # print('==========')

