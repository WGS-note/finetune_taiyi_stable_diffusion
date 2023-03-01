# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/11/2 11:12 上午
# @File: run_zh.py
'''
https://huggingface.co/svjack/Stable-Diffusion-Pokemon-zh
'''
import time
import torch
import pandas as pd, numpy as np
from torch.cuda.amp import autocast as autocast
from diffusers import LMSDiscreteScheduler
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, BertForTokenClassification
from transformers import CLIPProcessor, CLIPModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = "cpu"

class StableDiffusionPipelineWrapper(StableDiffusionPipeline):

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            **kwargs,
    ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length:])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(batch_size, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it

        # Unlike in other pipelines, latents need to be generated in the target device
        # for 1-to-1 results reproducibility with the CompVis implementation.
        # However this currently doesn't work in `mps`.
        latents_shape = (batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if latents is None:
            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                    self.device
                )
            else:
                latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            ###text_embeddings
            # print("before :" ,text_embeddings.shape)
            eh_shape = text_embeddings.shape
            if i == 0:
                eh_pad = torch.zeros((eh_shape[0], eh_shape[1], 768 - 512))
                eh_pad = eh_pad.to(self.device)
                # text_embeddings = torch.concat([text_embeddings, eh_pad], -1)
                text_embeddings = torch.cat([text_embeddings, eh_pad], -1)

            # print("after :" ,text_embeddings.shape)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                self.device
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

def generate_img(pipeline_wrap, config):
    start_time = time.time()
    imgs = pipeline_wrap(prompt=config['prompt'],
                         height=config['height'],
                         width=config['width'],
                         num_inference_steps=config['num_inference_steps'],
                         eta=config['eta'],
                         guidance_scale=config['guidance_scale'],
                         num_images_per_prompt=config['num_images_per_prompt'])

    image = imgs.images[0]
    image.save(config['save_path'])
    end_time = time.time()
    print('inference run time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

if __name__ == '__main__':

    '''   load encoder、vae、unet   '''
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
                                     beta_schedule="scaled_linear", num_train_timesteps=1000)

    start_time = time.time()
    pretrained_model_name_or_path = "./Stable-Diffusion-Pokemon-zh"

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = BertForTokenClassification.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")

    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

    end_time = time.time()
    print('Loading pre-trained model runtime: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))

    '''   define diffusion pipeling   '''
    # tokenizer.model_max_length = 77
    tokenizer.model_max_length = 512

    pipeline_wrap = StableDiffusionPipelineWrapper(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
    )

    pipeline_wrap.safety_checker = lambda images, clip_input: (images, False)
    pipeline_wrap = pipeline_wrap.to(device)

    '''   test   '''
    # config = {
    #     'prompt': '一只在吃白菜、头顶有帽子的小兔子',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 3.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out1.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': '''
    #     突破极限，奔逸绝尘！由网易暴雪举办的《魔兽世界》“巫妖王之怒”打本吧脚男：闪击纳克萨玛斯将于11月5日-6日每日19:00起正式打响！克尔苏加德再次坐镇天灾要塞，为扭转局势，击碎巫妖王的野心，冒险者们将再度吹响出征的号角，直面巫妖王的左膀右臂和他的邪恶爪牙！
    #     八支国服顶尖公会将响应召唤，发起纳克萨玛斯竞速挑战，志在冲击世界第一！究竟谁能笑到最后，他们又能否打破世界纪录？敬请届时锁定暴雪游戏频道，观看精彩赛况直播！
    #     ''',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 3.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out2.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': '纳尔应该怎么玩啊',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 3.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out3.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': '我都不知道我在说什么，但是这句话训练样本里肯定是没有的',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 3.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out4.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': '一只在吃香蕉的兔子',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 3.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out5.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': '今天你微笑了吗',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 3.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out6.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': '一只在吃白菜，头顶有帽子的小兔子',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 3.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out11.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': '一只在吃白菜头顶有帽子的小兔子',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 3.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out12.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)

    # config = {
    #     'prompt': '一只吃胡萝卜的兔子',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 0.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out7.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': '蓝色的龙',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 0.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out8.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': '一只蓝色的龙在喷火',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 0.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out9.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': '一个头上戴着盆栽的卡通人物',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 0.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out10.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': 'a rabbit eating a carrot',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 0.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out11.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': 'how does Gnar play',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 0.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out12.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)
    #
    # config = {
    #     'prompt': 'a photo of an astronaut riding a horse on mars',
    #     'height': 512,
    #     'width': 512,
    #     'num_inference_steps': 100,
    #     'eta': 0.0,
    #     'guidance_scale': 7.5,
    #     'num_images_per_prompt': 1,
    #     'save_path': './gen_imgs/out13.png',
    # }
    # generate_img(pipeline_wrap=pipeline_wrap, config=config)

    config = {
        'prompt': '一副油画',
        'height': 512,
        'width': 512,
        'num_inference_steps': 100,
        'eta': 0.0,
        'guidance_scale': 7.5,
        'num_images_per_prompt': 1,
        'save_path': './gen_imgs/out20.png',
    }
    generate_img(pipeline_wrap=pipeline_wrap, config=config)

    config = {
        'prompt': '一副山水画',
        'height': 512,
        'width': 512,
        'num_inference_steps': 100,
        'eta': 0.0,
        'guidance_scale': 7.5,
        'num_images_per_prompt': 1,
        'save_path': './gen_imgs/out21.png',
    }
    generate_img(pipeline_wrap=pipeline_wrap, config=config)

    config = {
        'prompt': '一只关于兔子吃胡萝卜的卡图画',
        'height': 512,
        'width': 512,
        'num_inference_steps': 100,
        'eta': 0.0,
        'guidance_scale': 7.5,
        'num_images_per_prompt': 1,
        'save_path': './gen_imgs/out22.png',
    }
    generate_img(pipeline_wrap=pipeline_wrap, config=config)

    config = {
        'prompt': '一只关于兔子吃胡萝卜的素描',
        'height': 512,
        'width': 512,
        'num_inference_steps': 100,
        'eta': 0.0,
        'guidance_scale': 7.5,
        'num_images_per_prompt': 1,
        'save_path': './gen_imgs/out23.png',
    }
    generate_img(pipeline_wrap=pipeline_wrap, config=config)

    print('******************************')
    print()
    print()
    print()


