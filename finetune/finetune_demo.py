# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/11/30 3:04 下午
# @File: finetune2.py
'''
finetune Taiyi Stable Diffusion zh
相关原理：https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzk0MzIzODM5MA==&action=getalbum&album_id=2664504810297114628#wechat_redirect
'''
import os, time, sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
# from fengshen.data.universal_datamodule import UniversalDataModule
from universal_datamodule import UniversalDataModule
from fengshen.models.model_utils import add_module_args, configure_optimizers, get_total_steps
# from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from universal_checkpoint import UniversalCheckpoint
from transformers import BertTokenizer, BertModel
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler
from torch.nn import functional as F
# from fengshen.data.taiyi_stable_diffusion_datasets.taiyi_datasets import add_data_args, load_data
from taiyi_datasets import add_data_args, load_data
from fengshen.utils.utils import report_memory
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class StableDiffusion(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Taiyi Stable Diffusion Module')
        parser.add_argument('--train_whole_model', action='store_true', default=False)
        return parent_parser

    def __init__(self, args):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
        self.text_encoder = BertModel.from_pretrained(args.model_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")

        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        # self.noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        self.save_hyperparameters(args)

        self.print_report_memory = args.print_report_memory

        self.current_loss = []
        self.best_loss = 1000
        self.best_models = []

        self.save_path = save_path

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        model_params = [{'params': self.text_encoder.parameters()}]
        if self.hparams.train_whole_model:
            model_params.append({'params': self.unet.parameters()})
        return configure_optimizers(self, model_params=model_params)

    def training_step(self, batch, batch_idx):
        self.text_encoder.train()
        self.unet.train()

        print('---0---', batch["pixel_values"].shape)

        with torch.no_grad():
            latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()

            print('---1--- ', latents.shape)

            latents = latents * 0.18215

        # 添加 Sample latents noise
        noise = torch.randn(latents.shape).to(latents.device)
        noise = noise.to(dtype=self.unet.dtype)
        bsz = latents.shape[0]
        # 为每个图像采样一个随机的时间步 Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # forward diffusion process

        print('---2--- ', timesteps.shape)

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(dtype=self.unet.dtype)

        print('---3--- ', noisy_latents.shape)

        # Get the text embedding for conditioning
        # with torch.no_grad():
        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

        print('---4--- ', encoder_hidden_states.shape)

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        print('---5--- ', noise_pred.shape)

        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
        self.log("train_loss", loss.item(),  on_epoch=False, prog_bar=True, logger=True)

        self.current_loss.append(loss.item())

        if self.trainer.global_rank == 0 and self.global_step == 10:
            report_memory('Stable Diffusion')   # 打印显存占用

        exit()

        return {"loss": loss}

    def on_train_epoch_end(self):

        if self.current_loss[-1] <= self.best_loss:
            self.best_loss = self.current_loss[-1]

            self.best_models = [self.text_encoder, self.vae, self.unet]

            text_encoder, vae, unet = self.best_models
            text_encoder.save_pretrained(os.path.join(self.save_path, "text_encoder"))
            vae.save_pretrained(os.path.join(self.save_path, "vae"))
            unet.save_pretrained(os.path.join(self.save_path, "unet"))

            print('current best loss is {}, save ok'.format(self.best_loss))

        # # 3W epoch、3 GPU，72 imgs、total steps = 3w * 72 // 3 = 720000
        # if self.trainer.current_epoch + 1 == 2000 \
        #         or self.trainer.current_epoch + 1 == 6000 \
        #         or self.trainer.current_epoch + 1 == 10000 \
        #         or self.trainer.current_epoch + 1 == 15000 \
        #         or self.trainer.current_epoch + 1 == 20000:
        #     # tmp_path = self.save_path + '{}/'.format(self.tracurrent best lossiner.current_epoch + 1)
        #     # if not os.path.exists(tmp_path):
        #     #     os.makedirs(tmp_path)
        #     text_encoder, vae, unet = self.best_models
        #     text_encoder.save_pretrained(os.path.join(self.save_path, "text_encoder"))
        #     vae.save_pretrained(os.path.join(self.save_path, "vae"))
        #     unet.save_pretrained(os.path.join(self.save_path, "unet"))

    def on_load_checkpoint(self, checkpoint) -> None:
        # 兼容低版本lightning，低版本lightning从ckpt起来时steps数会被重置为0
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset

def plt_fig(loss, save_path=None):
    # plt.plot(range(1, len(loss) + 1), loss, 'bo--')
    # x = range(0, len(loss) - 100, 100)
    x = range(0, len(loss) - 2, 2)
    plt.plot(x, [loss[i] for i in x], 'bo--')
    plt.title('train_loss')
    plt.xlabel("step_num")
    plt.ylabel('train_loss')
    plt.legend(["train_loss"])

    if save_path is not None:
        plt.savefig(save_path)

if __name__ == '__main__':

    # save_path = '/home/pre_models/Taiyi-SD-finetune/'
    save_path = '/home/pre_models/sd_finetune_app/'

    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = add_data_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = StableDiffusion.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser.add_argument('--print_report_memory', default=3, type=int)
    args = args_parser.parse_args()

    model = StableDiffusion(args)
    tokenizer = model.tokenizer
    datasets = load_data(args, tokenizer=tokenizer)   # <class 'torch.utils.data.dataset.ConcatDataset'>

    print('********** Loading Success **********')

    datamoule = UniversalDataModule(tokenizer=tokenizer, args=args, datasets=datasets)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    # print(args)

    trainer = Trainer.from_argparse_args(args, callbacks=[lr_monitor, checkpoint_callback])

    trainer.fit(model, datamodule=datamoule, ckpt_path=args.load_ckpt_path)

    plt_fig(model.current_loss, save_path=save_path + 'train_loss.png')

    exit()

    '''   save   '''
    text_encoder, vae, unet = model.best_models
    text_encoder.save_pretrained(os.path.join(save_path, "text_encoder"))
    vae.save_pretrained(os.path.join(save_path, "vae"))
    unet.save_pretrained(os.path.join(save_path, "unet"))

    print('********** current best model saving ok **********')
    print()
    print()
    print()

