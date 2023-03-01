# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/12/1 2:48 下午
# @File: text2img.py
'''
onnx

run text2image base on huggingface Pre-training diffusion model

path: 172.17.1.133 /data/wgs/text2img
docker: wgs-torch:3.0

支持预训练模型：
https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1
https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1
https://huggingface.co/hakurei/waifu-diffusion
https://huggingface.co/runwayml/stable-diffusion-v1-5
https://huggingface.co/nitrosocke/mo-di-diffusion
https://huggingface.co/CompVis/ldm-text2im-large-256

微调模型：
Taiyi-Stable-Diffusion-1B-Chinese-v0.1
'''
import time
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DiffusionPipeline, OnnxStableDiffusionPipeline
from diffusers import LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler
from diffusers.pipelines.stable_diffusion import pipeline_onnx_stable_diffusion

from convert_stable_diffusion_to_onnx import convert_models

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class Text2Img():

    __support_model__ = {
        'Taiyi-Stable_Diffusion-zh': '/home/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1',
        'Taiyi-Stable_Diffusion-zh-en': '/home/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1',
        'Runwayml-Stable_Diffusion-v1-5': '/home/pre_models/stable-diffusion-v1-5',
        'Hakurei-Waifu_Diffusion': '/home/pre_models/waifu-diffusion',
        'Nitrosocke-Mo_Di_Diffusion': '/home/pre_models/mo-di-diffusion',
        'CompVis-ldm_text2im_large': '/home/pre_models/ldm-text2im-large-256',
        'Taiyi-SD-finetune': '/home/pre_models/Taiyi-SD-finetune',
    }

    def __init__(self, model_name, use_full_precision=True):
        if model_name not in self.__support_model__.keys():
            raise ValueError('This model is not supported yet, the `model_name` has to be in {}'.format(list(self.__support_model__.keys())))

        self.model_name = model_name
        self.use_full_precision = use_full_precision

        self.pipe = self.load_pre_model()

    def load_pre_model(self):
        start_time = time.time()
        pre_model_path = self.__support_model__.get(self.model_name)

        if self.use_full_precision:
            torch_dtype = torch.float32
            self.print_flag = 'Full-Precision'
        else:
            if self.model_name == 'CompVis-ldm_text2im_large':
                raise ValueError('This model is not supported Half-Precision')
            torch.backends.cudnn.benchmark = True

            torch_dtype = torch.float16
            self.print_flag = 'Half-Precision'

        if self.model_name == 'CompVis-ldm_text2im_large':
            pipe = DiffusionPipeline.from_pretrained(pre_model_path, torch_dtype=torch_dtype)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(pre_model_path, torch_dtype=torch_dtype)

        end_time = time.time()
        print('Loading {} pre-trained model runtime: {:.0f}分 {:.0f}秒'.format(self.model_name, (end_time - start_time) // 60, (end_time - start_time) % 60))
        print()

        return pipe.to(device)

    def generate(self, prompt, save_path,
                 negative_prompt=None,
                 height=512,
                 width=512,
                 num_inference_steps=50,
                 guidance_scale=7.5,
                 num_images_per_prompt=1,
                 eta=0.0,
                 ):
        '''
        + prompt:
        + save_path:
        + negative_prompt: ，反向文本，提升图像质量，但是推理时间会变慢，negative_prompt = '广告, ，, ！, 。, ；, 资讯, 新闻, 水印'
        + height: 生成图像的高度(以像素为单位)
        + width: 生成图像的宽度(以像素为单位)
        + num_inference_steps: 去噪步骤的数量, 更多的去噪步骤通常会导致更高质量的图像，但代价是更慢的推断
        + guidance_scale: 无分类指导因子 - 实现稳定扩散, 能让生成图像匹配文字提示, 取值范围 0～20, 过高会牺牲图像质量或多样性, 建议值 7～8.5
        + num_images_per_prompt: 每个 prompt 要生成的图像数量，经测试单卡生成最多2张
        + eta: Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to schedulers.DDIMScheduler, will be ignored for others.
        '''
        start_time = time.time()

        image = self.pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt, eta=eta).images[0]

        image.save(save_path)

        extrema = image.convert("L").getextrema()
        if extrema == (0, 0):
            print('the pixel are all black')

        # images = self.pipe(prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, num_images_per_prompt=num_images_per_prompt, eta=eta).images
        # for idx, image in enumerate(images):
        #     image.save(f"{save_path}_{idx}.png")
        #     print('------', f"{save_path}_{idx}.png", ' --- ok')

        end_time = time.time()
        print('{} - {} - Inference Run Time: {:.0f}分 {:.0f}秒'.format(self.model_name, self.print_flag, (end_time - start_time) // 60, (end_time - start_time) % 60))
        print()
        # torch.cuda.empty_cache()

if __name__ == '__main__':

    '''
    
    docker run -d --gpus '"device=0,1,2,3"' \
       --rm -it --name t2i_speedup \
       -v /data/wgs/text2img:/home \
       wgs-torch:6.1 \
       sh -c "python -u /home/speedup/text2img.py 1>>/home/log/text2img.log 2>>/home/log/text2img.err"
    
    docker run --rm -it -v /data/wgs/text2img:/workspace pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel bash
    python -u /workspace/speedup/text2img.py 1>>/workspace/log/text2img.log 2>>/workspace/log/text2img.err
    '''

    # # convert_models(model_path='/home/pre_models/Taiyi-SD-finetune', output_path='/home/pre_models/Taiyi-SD-finetune_onnx', opset=12, fp16=False)
    # # convert_models(model_path='home/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1', output_path='home/pre_models/Taiyi-SD_onnx', opset=12, fp16=True)
    # convert_models(model_path='/home/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1', output_path='/home/pre_models/Taiyi-SD_onnx', opset=12, fp16=True)

    pipe = OnnxStableDiffusionPipeline.from_pretrained(
        # "/home/pre_models/Taiyi-SD-finetune_onnx",
        "/home/pre_models/Taiyi-SD_onnx",
        revision="onnx",
        provider="CUDAExecutionProvider",
        torch_dtype=torch.float16
    )
    # .to(device)

    # prompt = "灵魂莲花迅捷斥候"
    prompt = "一只穿着宇航服的哈士奇"
    start_time = time.time()
    image = pipe(prompt).images[0]
    image.save('/home/gen_imgs/A_onnx_4.jpg')
    end_time = time.time()
    print('Inference Run Time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    exit()

    # m = Text2Img(model_name='Taiyi-Stable_Diffusion-zh', use_full_precision=True)
    # m.generate(prompt="迅捷斥候", save_path='./gen_imgs/A_naer_o_1.png')
    # m.generate(prompt="迅捷斥候", save_path='./gen_imgs/A_naer_o_2.png')
    # m.generate(prompt="迅捷斥候", save_path='./gen_imgs/A_naer_o_3.png')

    negative_prompt = '广告, ，, ！, 。, ；, 资讯, 新闻, 水印'

    '''
    , 复杂
    , 高清
    , 4k壁纸
    , 唯美
    , 插画
    '''


