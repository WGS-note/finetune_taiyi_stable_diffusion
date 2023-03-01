#!/bin/bash
#SBATCH --job-name=finetune_taiyi # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

cd /data/wgs/text2img

MODEL_NAME=Taiyi-Stable-Diffusion-1B-Chinese-v0.1

MODEL_ROOT_DIR=/data/wgs/text2img/pre_models/${MODEL_NAME}
#if [ ! -d ${MODEL_ROOT_DIR} ];then
#  mkdir ${MODEL_ROOT_DIR}
#fi

NNODES=1
GPUS_PER_NODE=4

MICRO_BATCH_SIZE=1

# 如果你不用Deepspeed的话 下面的一段话都可以删掉 Begin
CONFIG_JSON="$MODEL_ROOT_DIR/${MODEL_NAME}.ds_config.json"
ZERO_STAGE=1
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $CONFIG_JSON
{
    "zero_optimization": {
        "stage": ${ZERO_STAGE}
    },
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON
### End

DATA_ARGS="\
        --dataloader_workers 3 \
        --train_batchsize $MICRO_BATCH_SIZE  \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        --datasets_path ./data \
        --datasets_type txt \
        --resolution 512 \
        "
# --resolution 512 \

MODEL_ARGS="\
        --model_path ./pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1 \
        --learning_rate 1e-2 \
        --weight_decay 1e-1 \
        --warmup_ratio 0.01 \
        "
# --learning_rate 1e-4 \

MODEL_CHECKPOINT_ARGS="\
        --save_last\
        --verbose True\
        --save_ckpt_path ./pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/ckpt/ \
        --load_ckpt_path ./pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/ckpt/last.ckpt \
        "
# --monitor val_acc  (val_loss)
# mode max
# --save_ckpt_path ${MODEL_ROOT_DIR}/ckpt/ \
# --load_ckpt_path ${MODEL_ROOT_DIR}/ckpt/last.ckpt \
# --every_n_train_steps 1\

TRAINER_ARGS="\
        --max_epoch 5 \
        --accelerator=gpu \
        --strategy=ddp \
        --gpus $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 10 \
        --precision 16 \
        --default_root_dir ${MODEL_ROOT_DIR} \
        --replace_sampler_ddp False \
        --num_sanity_val_steps 0 \
        --limit_val_batches 0 \
        "
# --accelerator=gpu \
# --strategy=ddp \ dp
# --precision 32 \
# --log_every_n_steps 100 \
# --strategy deepspeed_stage_${ZERO_STAGE} \

# num_sanity_val_steps， limit_val_batches 通过这俩参数把 validation 关了

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

#srun -N $NNODES --gres=gpu:$GPUS_PER_NODE --ntasks-per-node=$GPUS_PER_NODE --cpus-per-task=20 python3 pretrain_deberta.py $options

docker run -d --gpus '"device=0,1,2,3"' \
       --rm -it --name text2image_test \
       --shm-size 12G \
       -v /data/wgs/text2img:/home \
       wgs-torch:3.0 \
       sh -c "python3 -u /home/finetune.py $options 1>>/home/log/finetune.log 2>>/home/log/finetune.err"

# nohup sh /data/wgs/text2img/dk/finetune.sh &
# sh /data/wgs/text2img/dk/finetune.sh
# docker run --gpus '"device=0,1,2,3"' --shm-size 8G --rm -it -v /data/wgs/text2img:/home wgs-torch:3.0 bash


