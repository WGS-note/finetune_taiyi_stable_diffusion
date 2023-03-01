import oneflow as torch
from diffusers import OneFlowStableDiffusionPipeline
import random


# 模型：CompVis/stable-diffusion-v1-4，fp16
# pipe = OneFlowStableDiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4",
#     use_auth_token=True,
#     revision="fp16",
#     torch_dtype=torch.float16,
# )

# 模型：CompVis/stable-diffusion-v1-4，fp32
# pipe = OneFlowStableDiffusionPipeline.from_pretrained(
#     "CompVis/stable-diffusion-v1-4",
#     use_auth_token=True,
#     torch_dtype=torch.float32
# )

# 模型：IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1，fp32（fp16不支持，纯中文版不支持）
pipe = OneFlowStableDiffusionPipeline.from_pretrained(
    "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1",
    torch_dtype=torch.float32
)


pipe = pipe.to("cuda")

prompt = "猫"

with torch.autocast("cuda"):
    images = pipe(prompt, height=512, width=512, num_inference_steps=50).images
    for i, image in enumerate(images):
        image.save(f"{i}.png")


# 增加seed方法，时间差异不大，8秒
# randomSeed = random.randint(0, 2147483647)
# print("seed:",randomSeed)
# generator = torch.Generator("cuda").manual_seed(randomSeed)
# with torch.autocast("cuda"):
#     images = pipe(prompt, height=512, width=512, num_inference_steps=50, generator=generator).images
#     for i, image in enumerate(images):
#         image.save(f"{i}.png")
        

# 不能使用下面方法，和原来框架时间相似，23秒
# image = pipe(prompt, height=512, width=512, num_inference_steps=50).images[0]

