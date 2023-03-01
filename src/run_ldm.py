# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/11/4 10:37 上午
# @File: run_ldm.py
'''
https://huggingface.co/CompVis/ldm-text2im-large-256
'''

import spacy
print(spacy.__version__)
exit()

# !pip install diffusers transformers
from diffusers import DiffusionPipeline

model_id = "CompVis/ldm-text2im-large-256"

# load model and scheduler
ldm = DiffusionPipeline.from_pretrained(model_id)

# run pipeline in inference (sample random noise and denoise)
prompt = "A painting of a squirrel eating a burger"
images = ldm([prompt], num_inference_steps=50, eta=0.3, guidance_scale=6)["sample"]

# save images
for idx, image in enumerate(images):
    image.save(f"squirrel-{idx}.png")





