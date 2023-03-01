# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/11/28 10:39 上午
# @File: get_args.py
'''
get the args need for fine-tuning
'''
import argparse
import pandas as pd
from collections import namedtuple

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # 预训练模型路径
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str, default=None, required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--dataset_name", type=str, default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )

    parser.add_argument(
        "--dataset_config_name", type=str, default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )

    parser.add_argument(
        "--train_data_dir", type=str, default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )

    parser.add_argument("--image_column", type=str, default="image", help="The column of the dataset containing an image.")

    parser.add_argument("--caption_column", type=str, default="text", help="The column of the dataset containing a caption or a list of captions.")

    parser.add_argument("--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution (if not set, random crop will be used)",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    #args = parser.parse_args()
    return parser
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args

def parse_parser_add_arg(parser, as_named_tuple=False):
    args_df = pd.DataFrame(
    pd.Series(parser.__dict__["_actions"]).map(lambda x: x.__dict__).values.tolist())
    args_df = args_df.explode("option_strings")
    args_df["option_strings"] = args_df["option_strings"].map(lambda x: x[2:] if x.startswith("--") else x)\
        .map(lambda x: x[1:] if x.startswith("-") else x)
    args_df = args_df[["option_strings", "default"]]
    args = dict(args_df.values.tolist())
    if as_named_tuple:
        args_parser_namedtuple = namedtuple("args_config", args)
        return args_parser_namedtuple(**args)
    return args_df

def transform_named_tuple_to_dict(N_tuple):
    return dict(map(lambda x: (x, getattr(N_tuple, x)), filter(lambda x: not x.startswith("_"), dir(N_tuple))))

def transform_dict_to_named_tuple(dict_, name="NamedTuple"):
    args_parser_namedtuple = namedtuple(name, dict_)
    return args_parser_namedtuple(**dict_)

args = parse_args()
args = parse_parser_add_arg(args, as_named_tuple=True)

args_dict = transform_named_tuple_to_dict(args)
args_dict["pretrained_model_name_or_path"] = "/home/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/"
# args_dict["pretrained_model_name_or_path"] = "/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"
# args_dict["dataset_name"] = "svjack/pokemon-blip-captions-en-zh"
# args_dict["dataset_name"] = "/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/data/demo_dataset2"
args_dict["dataset_name"] = "/home/data/demo_dataset2"
args_dict["use_ema"] = True
#args_dict["use_ema"] = False
args_dict["resolution"] = 512
args_dict["center_crop"] = True
args_dict["random_flip"] = True
args_dict["train_batch_size"] = 1
args_dict["gradient_accumulation_steps"] = 4
args_dict["gradient_checkpointing"] = True
#### to 15000
args_dict["num_train_epochs"] = 3
# args_dict["max_train_steps"] = 50000
args_dict["max_train_steps"] = 5
# args_dict["learning_rate"] = 1e-05
args_dict["learning_rate"] = 1e-02
args_dict["max_grad_norm"] = 1
args_dict["lr_scheduler"] = "constant"
args_dict["lr_warmup_steps"] = 0
args_dict["logging_dir"] = "/home/log/fintunelogs"
args_dict["output_dir"] = "/home/pre_models/sd_finetune_test"
# args_dict["output_dir"] = "/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/finetune_test"
args_dict["caption_column"] = "text"
args_dict["mixed_precision"] = "no"
args = transform_dict_to_named_tuple(args_dict)

# dataset_name_mapping = {
#     "svjack/pokemon-blip-captions-en-zh": ("image", "zh_text"),
# }
dataset_name_mapping = {
    args.dataset_name: ("image", "text"),
}

'''
export MODEL_NAME="stable-diffusion-v1-4/"
export dataset_name="lambdalabs/pokemon-blip-captions"

accelerate launch train_text_to_image_ori.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=32 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-pokemon-model"
'''

if __name__ == '__main__':
    print()

    print(args)

