# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/11/4 10:24 上午
# @File: run_sd-inp.py
'''
https://huggingface.co/nitrosocke/mo-di-diffusion
'''
from diffusers import StableDiffusionPipeline
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def demo():
    model_id = "nitrosocke/mo-di-diffusion"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "a magical princess with golden hair, modern disney style"
    image = pipe(prompt).images[0]

    image.save("./magical_princess.png")

def gen_img(pipe, prompt, path):
    start_time = time.time()
    # image = pipe(prompt, guidance_scale=6).images[0]
    image = pipe(prompt).images[0]
    image.save(path)
    end_time = time.time()
    print('inference run time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()


if __name__ == '__main__':
    pre_modepath = "./pre_models/stable-diffusion-v1-5"

    start_time = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(pre_modepath).to(device)
    end_time = time.time()
    print('Loading pre-trained model runtime: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

    # 秋天的颐和园好美
    prompt = 'The Summer Palace is so beautiful in autumn'
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-1.png')

    # 哈士奇和柯基生成的狗长什么样？
    prompt = "What is Husky and Corgi's child like?"
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-2.png')

    # 一只驾驶宇宙飞船的柯基
    prompt = 'A corgi piloting a spaceship'
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-3.png')

    # 君不见黄河之水天上来
    prompt = 'The water of the Yellow River comes from heaven'
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-4.png')

    # 君不见黄河之水天上来，Van Gogh style
    prompt = 'The water of the Yellow River comes from heaven, Van Gogh style'
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-5.png')

    # 君不见黄河之水天上来，科幻，国画
    prompt = 'The water of the Yellow River comes from heaven, science fiction, traditional Chinese painting'
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-6.png')

    # 一只穿着宇航服的哈士奇
    prompt = 'A husky in a space suit'
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-7.png')

    # 一只穿着宇航服的哈士奇，科幻
    prompt = 'A husky in a space suit, science fiction'
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-8.png')

    # 一只穿着宇航服的哈士奇，Van Gogh style
    prompt = 'A husky in a space suit, Van Gogh style'
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-9.png')

    # 一只吃胡萝卜的兔子正在发呆
    prompt = 'A rabbit eating a carrot is in a daze'
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-10.png')

    # 一只在吃白菜、头顶有帽子的小兔子
    prompt = 'A rabbit with a hat on his head eating cabbage'
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-11.png')

    # 突破极限，奔逸绝尘！由网易暴雪举办的《魔兽世界》“巫妖王之怒”打本吧脚男：闪击纳克萨玛斯将于11月5日-6日每日19:00起正式打响！克尔苏加德再次坐镇天灾要塞，为扭转局势，击碎巫妖王的野心，冒险者们将再度吹响出征的号角，直面巫妖王的左膀右臂和他的邪恶爪牙！
    prompt = '''
            Break the limit, run away from the dust! World of Warcraft "Wrath of the Lich King" will be held by NetEase Blizzard from November 5 to 6 at 19:00 every day. With Kel 'thuzad once again at the helm of Scourge's fortress, the ADVENTURers will once again blow their horns against the Lich King's men and his evil minions in order to turn the tide and break the Lich King's ambitions!
            '''
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-12.png')

    # 八支国服顶尖公会将响应召唤，发起纳克萨玛斯竞速挑战，志在冲击世界第一！究竟谁能笑到最后，他们又能否打破世界纪录？敬请届时锁定暴雪游戏频道，观看精彩赛况直播！
    prompt = '''
            Eight top national guilds will answer the call and launch the Naxamas Race Challenge, aiming to beat the world number one! Who will have the last laugh, and will they break the world record? Please lock the snow game channel at that time, watch the exciting game live!
            '''
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-13.png')

    # 纳尔应该怎么玩啊
    prompt = 'How should Gnar play'
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-14.png')

    # 我都不知道我在说什么，但是这句话训练样本里肯定是没有的
    prompt = "I don't even know what I'm talking about, but it's definitely not in the training sample, okay"
    gen_img(pipe=pipe, prompt=prompt, path='./gen_imgs/mdd-15.png')

    print('******************************')
    print()
    print()
    print()
