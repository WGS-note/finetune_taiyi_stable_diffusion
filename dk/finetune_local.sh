#!/bin/bash
#SBATCH --job-name=finetune_taiyi # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

cd /Users/wangguisen/Documents/markdowns/AI-note/元宇宙

MODEL_NAME=Taiyi-Stable-Diffusion-1B-Chinese-v0.1

MODEL_ROOT_DIR=/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/${MODEL_NAME}
#if [ ! -d ${MODEL_ROOT_DIR} ];then
#  mkdir ${MODEL_ROOT_DIR}
#fi

NNODES=1
GPUS_PER_NODE=1

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
        --dataloader_workers 2 \
        --train_batchsize $MICRO_BATCH_SIZE  \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        --datasets_path /Users/wangguisen/Documents/markdowns/AI-note/元宇宙/finetune_taiyi_stable_diffusion/demo_dataset \
        --datasets_type txt \
        --resolution 512 \
        "

MODEL_ARGS="\
        --model_path /Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1 \
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
        --gpus $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 100 \
        --precision 32 \
        --default_root_dir ${MODEL_ROOT_DIR} \
        --replace_sampler_ddp False \
        --num_sanity_val_steps 0 \
        --limit_val_batches 0 \
        "
# num_sanity_val_steps， limit_val_batches 通过这俩参数把 validation 关了

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

#python3 finetune.py $options
#srun -N $NNODES --gres=gpu:$GPUS_PER_NODE --ntasks-per-node=$GPUS_PER_NODE --cpus-per-task=20 python3 pretrain_deberta.py $options

#docker run -d --gpus '"device=0,1,2,3"' \
#       --rm -it --name text2image_test \
#       -v /data/wgs/text2img:/home \
#       wgs-torch:2.0 \
#       sh -c "python -u /home/finetune.py $options 1>>/home/log/finetune.log 2>>/home/log/finetune.err"

# nohup sh /data/wgs/text2img/dk/finetune.sh &

cd ./finetune_taiyi_stable_diffusion
python -u ./finetune.py $options 1>>./finetune.log 2>>./finetune.err

# sh /Users/wangguisen/Documents/markdowns/AI-note/元宇宙/finetune_taiyi_stable_diffusion/finetune_local.sh


