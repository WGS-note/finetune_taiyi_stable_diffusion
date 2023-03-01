# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/11/3 9:56 上午
# @File: run_taiyi_sd_zh_en.py
'''
https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1
'''
import time
import torch
import pandas as pd, numpy as np
from torch.cuda.amp import autocast as autocast
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, BertForTokenClassification
from transformers import CLIPProcessor, CLIPModel
# from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

def gen_img(prompt, path):
    start_time = time.time()

    image = pipe(prompt, guidance_scale=7.5).images[0]
    image.save(path)

    end_time = time.time()
    print('inference run time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()


if __name__ == '__main__':

    pre_modepath = "./pre_models/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1"

    print('---------- full precision ----------')
    print('------------------------------')
    '''   全精度   '''
    start_time = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(pre_modepath).to(device)
    end_time = time.time()
    print('Loading Full precision pre-trained model runtime: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

    prompt = '君不见黄河之水天上来'
    gen_img(prompt, path='./gen_imgs/taiyi1-EN.png')

    prompt = '君不见黄河之水天上来，Van Gogh style'
    gen_img(prompt, path='./gen_imgs/taiyi2-EN.png')

    prompt = '君不见黄河之水天上来，科幻，国画'
    gen_img(prompt, path='./gen_imgs/taiyi3-EN.png')

    prompt = '一只穿着宇航服的哈士奇'
    gen_img(prompt, path='./gen_imgs/taiyi4-EN.png')

    prompt = '一只穿着宇航服的哈士奇，科幻'
    gen_img(prompt, path='./gen_imgs/taiyi5-EN.png')

    prompt = '一只穿着宇航服的哈士奇，Van Gogh style'
    gen_img(prompt, path='./gen_imgs/taiyi6-EN.png')

    prompt = '一只吃胡萝卜的兔子正在发呆'
    gen_img(prompt, path='./gen_imgs/taiyi7-EN.png')

    prompt = '一只在吃白菜、头顶有帽子的小兔子'
    gen_img(prompt, path='./gen_imgs/taiyi8-EN.png')

    prompt = '一只在吃白菜头顶有帽子的小兔子'
    gen_img(prompt, path='./gen_imgs/taiyi81-EN.png')

    prompt = '''
        突破极限，奔逸绝尘！由网易暴雪举办的《魔兽世界》“巫妖王之怒”打本吧脚男：闪击纳克萨玛斯将于11月5日-6日每日19:00起正式打响！克尔苏加德再次坐镇天灾要塞，为扭转局势，击碎巫妖王的野心，冒险者们将再度吹响出征的号角，直面巫妖王的左膀右臂和他的邪恶爪牙！
        八支国服顶尖公会将响应召唤，发起纳克萨玛斯竞速挑战，志在冲击世界第一！究竟谁能笑到最后，他们又能否打破世界纪录？敬请届时锁定暴雪游戏频道，观看精彩赛况直播！
        '''
    gen_img(prompt, path='./gen_imgs/taiyi9-EN.png')

    prompt = '纳尔应该怎么玩啊'
    gen_img(prompt, path='./gen_imgs/taiyi10-EN.png')

    prompt = '我都不知道我在说什么，但是这句话训练样本里肯定是没有的'
    gen_img(prompt, path='./gen_imgs/taiyi11-EN.png')

    print('---------- half precision ----------')
    print('------------------------------')
    '''   半精度   '''
    start_time = time.time()
    torch.backends.cudnn.benchmark = True
    # device_map="auto"
    pipe = StableDiffusionPipeline.from_pretrained(pre_modepath, torch_dtype=torch.float16).to(device)
    end_time = time.time()
    print('Loading Half precision pre-trained model runtime: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

    prompt = '君不见黄河之水天上来'
    gen_img(prompt, path='./gen_imgs/taiyi1-EN-half.png')

    prompt = '君不见黄河之水天上来，Van Gogh style'
    gen_img(prompt, path='./gen_imgs/taiyi2-EN-half.png')

    prompt = '君不见黄河之水天上来，科幻，国画'
    gen_img(prompt, path='./gen_imgs/taiyi3-EN-half.png')

    prompt = '一只穿着宇航服的哈士奇'
    gen_img(prompt, path='./gen_imgs/taiyi4-EN-half.png')

    prompt = '一只穿着宇航服的哈士奇，科幻'
    gen_img(prompt, path='./gen_imgs/taiyi5-EN-half.png')

    prompt = '一只穿着宇航服的哈士奇，Van Gogh style'
    gen_img(prompt, path='./gen_imgs/taiyi6-EN-half.png')

    prompt = '一只吃胡萝卜的兔子正在发呆'
    gen_img(prompt, path='./gen_imgs/taiyi7-EN-half.png')

    prompt = '一只在吃白菜、头顶有帽子的小兔子'
    gen_img(prompt, path='./gen_imgs/taiyi8-EN-half.png')

    prompt = '一只在吃白菜头顶有帽子的小兔子'
    gen_img(prompt, path='./gen_imgs/taiyi81-EN-half.png')

    prompt = '''
            突破极限，奔逸绝尘！由网易暴雪举办的《魔兽世界》“巫妖王之怒”打本吧脚男：闪击纳克萨玛斯将于11月5日-6日每日19:00起正式打响！克尔苏加德再次坐镇天灾要塞，为扭转局势，击碎巫妖王的野心，冒险者们将再度吹响出征的号角，直面巫妖王的左膀右臂和他的邪恶爪牙！
            八支国服顶尖公会将响应召唤，发起纳克萨玛斯竞速挑战，志在冲击世界第一！究竟谁能笑到最后，他们又能否打破世界纪录？敬请届时锁定暴雪游戏频道，观看精彩赛况直播！
            '''
    gen_img(prompt, path='./gen_imgs/taiyi9-EN-half.png')

    prompt = '纳尔应该怎么玩啊'
    gen_img(prompt, path='./gen_imgs/taiyi10-EN-half.png')

    prompt = '我都不知道我在说什么，但是这句话训练样本里肯定是没有的'
    gen_img(prompt, path='./gen_imgs/taiyi11-EN-half.png')

    print('******************************')
    print()
    print()
    print()


