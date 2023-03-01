# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/11/28 10:54 上午
# @File: finetuneSDM.py
'''
fine-tune stable diffusion
'''
import numpy as np, time
import math
import os
import logging
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, BertForTokenClassification, BertModel
from transformers import CLIPProcessor, CLIPModel
from accelerate import Accelerator, accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from accelerate import notebook_launcher

from get_args import args, dataset_name_mapping, transform_named_tuple_to_dict, transform_dict_to_named_tuple
from finetune_utils import *

logger = get_logger(__name__)

'''   requires_grad_   '''
REQUIRES_GRAD = False

'''   加载预训练 stable diffusion  '''
def load_pre_model():
    start_time = time.time()

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = BertModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(REQUIRES_GRAD)
    text_encoder.requires_grad_(REQUIRES_GRAD)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    end_time = time.time()
    print('Loading pre-trained model runtime: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))

    return tokenizer, text_encoder, vae, unet

'''   Initialize the optimizer   '''
def initialize():

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
        # tensor_format="pt"
    )

    return optimizer_cls, optimizer, noise_scheduler

'''   Build the dataset   '''
def build_dataset():
    dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)

    column_names = dataset["train"].column_names
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)

    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}")

    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}")

    return dataset, image_column, caption_column

'''   Preprocessing the datasets   '''
def preprocessing_datasets():
    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples, tokenizer=tokenizer, caption_column=caption_column, is_train=True)

        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)

    # print(train_dataset)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]
        padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")

        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
            # "attention_mask": padded_tokens.attention_mask,
        }

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size)

    # for step, batch in enumerate(train_dataloader):
    #     print(step)
    #     print(batch['pixel_values'].shape, batch['input_ids'])

    return train_dataset, train_dataloader

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    args = config
    unet = model

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
            device_placement=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            logging_dir=logging_dir,
        )

    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())
        ema_unet.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args_dict = transform_named_tuple_to_dict(args)
        args_dict["max_train_steps"] = args.num_train_epochs * num_update_steps_per_epoch
        args = transform_dict_to_named_tuple(args_dict)

    args_dict = transform_named_tuple_to_dict(args)
    args_dict["num_train_epochs"] = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    args = transform_dict_to_named_tuple(args_dict)

    if accelerator.is_main_process:
        if config.push_to_hub:
            #repo = init_git_repo(config, at_init=True)
            pass
        accelerator.init_trackers("train_example")

        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("********** Running training **********")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    '''
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    '''
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        #unet.train()
        text_encoder.train()

        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                noise = noise.to(dtype=unet.dtype)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noisy_latents = noisy_latents.to(dtype=unet.dtype)

                # Get the text embedding for conditioning
                # with torch.no_grad():
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

if __name__ == '__main__':

    '''
    
    docker run -d --gpus '"device=0,1,2,3"' \
       --rm -it --name text2image_test \
       --shm-size 12G \
       -v /data/wgs/text2img:/home \
       wgs-torch:3.0 \
       sh -c "python3 -u /home/finetune/finetuneSDM.py $options 1>>/home/log/finetuneSDM.log 2>>/home/log/finetuneSDM.err"
    
    docker run -d --gpus '"device=0,1,2,3"' \
       --rm -it --name text2image_test \
       --shm-size 12G \
       -v /data/wgs/text2img:/home \
       wgs-torch:3.0 \
       sh -c "accelerate launch /home/finetune/finetuneSDM.py $options 1>>/home/log/finetuneSDM.log 2>>/home/log/finetuneSDM.err"
    
    docker run --gpus '"device=0,1,2,3"' \
       --rm -it --name text2image_test \
       --shm-size 12G \
       -v /data/wgs/text2img:/home \
       wgs-torch:3.0 \
       bash
    '''

    print('********** Fine-tuning starts **********')
    start_time = time.time()

    '''   Loading pre-trained stable diffusion  '''
    tokenizer, text_encoder, vae, unet = load_pre_model()

    '''   Initialize the optimizer   '''
    optimizer_cls, optimizer, noise_scheduler = initialize()

    '''   Build the dataset   '''
    dataset, image_column, caption_column = build_dataset()

    '''   Preprocessing the datasets   '''
    train_dataset, train_dataloader = preprocessing_datasets()

    '''   fine-tune train   '''
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    ####args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    args_ = (args, unet, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    # notebook_launcher(train_loop, args_, num_processes=4, mixed_precision='fp16')
    # notebook_launcher(train_loop, args_, num_processes=4)
    train_loop(args, unet, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

    '''   save   '''
    save_path = args.output_dir + 'TaiYi-SD-Finetune'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
    text_encoder.save_pretrained(os.path.join(save_path, "text_encoder"))
    vae.save_pretrained(os.path.join(save_path, "vae"))
    unet.save_pretrained(os.path.join(save_path, "unet"))
    print('Saved model successfully')

    end_time = time.time()
    print('Fine-Tuning Run Time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print('********** Fine-tuning ok **********')
    print()
    print()
    print()









