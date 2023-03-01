#!/bin/bash

cd /home

MODEL_NAME=Taiyi-Stable-Diffusion-1B-Chinese-v0.1

MODEL_ROOT_DIR=/home/pre_models/${MODEL_NAME}

NNODES=1
GPUS_PER_NODE=3

MICRO_BATCH_SIZE=1

## 如果你不用Deepspeed的话 下面的一段话都可以删掉 Begin
#CONFIG_JSON="$MODEL_ROOT_DIR/${MODEL_NAME}.ds_config.json"
#ZERO_STAGE=1
## Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
#cat <<EOT > $CONFIG_JSON
#{
#    "zero_optimization": {
#        "stage": ${ZERO_STAGE}
#    },
#    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE
#}
#EOT
#export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON
#### End

DATA_ARGS="\
        --dataloader_workers 3 \
        --train_batchsize $MICRO_BATCH_SIZE  \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        --datasets_path ./data \
        --datasets_type txt \
        --resolution 512 \
        "

MODEL_ARGS="\
        --model_path ./pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1 \
        --learning_rate 1e-4 \
        --weight_decay 1e-1 \
        --warmup_ratio 0.01 \
        "

MODEL_CHECKPOINT_ARGS="\
        --save_last \
        --save_ckpt_path ${MODEL_ROOT_DIR}/ckpt \
        --load_ckpt_path ${MODEL_ROOT_DIR}/ckpt/last.ckpt \
        "

TRAINER_ARGS="\
        --max_epoch 10 \
        --accelerator=gpu \
        --strategy=ddp \
        --gpus $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --log_every_n_steps 100 \
        --precision 32 \
        --default_root_dir ${MODEL_ROOT_DIR} \
        --replace_sampler_ddp False \
        --num_sanity_val_steps 0 \
        --limit_val_batches 0 \
        "
# ********* --accelerator=gpu \
# --strategy=ddp \ dp
# --strategy deepspeed_stage_${ZERO_STAGE} \

# num_sanity_val_steps， limit_val_batches 通过这俩参数把 validation 关了

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

python3 -u /home/finetune.py $options 1>>/home/log/finetune.log 2>>/home/log/finetune.err

