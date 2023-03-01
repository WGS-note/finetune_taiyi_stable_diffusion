#!/bin/bash
#SBATCH --job-name=randeng_t5_77M
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8               # number of gpus
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -o %x-%j.log
#SBATCH -e %x-%j.err

set -x -e

echo "START TIME: $(date)"
MICRO_BATCH_SIZE=64
ROOT_DIR=/cognitive_comp/ganruyi/experiments/randeng_t5_77M/

ZERO_STAGE=1

config_json="$ROOT_DIR/ds_config.t5_cn_small_pretrain.$SLURM_JOBID.json"
export MASTER_PORT=$[RANDOM%10000+30000]

cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": ${MICRO_BATCH_SIZE},
  "steps_per_print": 100,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "contiguous_gradients": false,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 500000000
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-4,
      "weight_decay": 1e-2
    }
  },
  "scheduler": {
    "params": {
      "warmup_max_lr": 1e-04,
      "warmup_min_lr": 1e-05,
      "total_num_steps": 100000,
      "warmup_num_steps" : 10000
    },
    "type": "WarmupDecayLR"  
  },
  "zero_allow_untested_optimizer": false,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "activation_checkpointing": {
    "partition_activations": false,
    "contiguous_memory_optimization": false
  },
  "wall_clock_breakdown": false
}
EOT

export PL_DEEPSPEED_CONFIG_PATH=$config_json
export TORCH_EXTENSIONS_DIR=/cognitive_comp/ganruyi/tmp/torch_extendsions
# strategy=ddp
strategy=deepspeed_stage_1

TRAINER_ARGS="
    --max_epochs 1 \
    --gpus 8 \
    --num_nodes 1 \
    --strategy ${strategy} \
    --default_root_dir $ROOT_DIR \
    --dirpath $ROOT_DIR/ckpt \
    --save_top_k 3 \
    --every_n_train_steps 50000 \
    --monitor train_loss \
    --mode min \
    --save_last \
    --val_check_interval 0.01 \
    --preprocessing_num_workers 20 \
"
# --accumulate_grad_batches 8 \
DATA_DIR=wudao_180g_t5_tokenized_512

DATA_ARGS="
    --train_batchsize $MICRO_BATCH_SIZE \
    --valid_batchsize $MICRO_BATCH_SIZE \
    --train_data ${DATA_DIR} \
    --train_split_size 0.999 \
    --max_seq_length 512 \
"

MODEL_ARGS="
    --pretrained_model_path /cognitive_comp/ganruyi/hf_models/google/mt5-small \
    --new_vocab_path /cognitive_comp/ganruyi/hf_models/t5_cn_small/sentencepiece_cn.model \
    --keep_tokens_path /cognitive_comp/ganruyi/hf_models/t5_cn_small/sentencepiece_cn_keep_tokens.json \
"
SCRIPTS_PATH=/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/examples/pretrain_t5/pretrain_t5.py

export CMD=" \
    $SCRIPTS_PATH \
    $TRAINER_ARGS \
    $MODEL_ARGS \
    $DATA_ARGS \
    "

echo $CMD
# source activate base
# python $CMD
# srun --nodes=1 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=30 --jobid=171866 -e %x-%j.err -o %x-%j.log python $CMD

SINGULARITY_PATH=/cognitive_comp/ganruyi/pytorch21_06_py3_docker_image_v2.sif
srun --jobid=171866 --job-name=randeng_t5_77M --nodes=1 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=30 -e %x-%j.err -o %x-%j.log singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c '/home/ganruyi/anaconda3/bin/python $CMD'


# to debug - add echo (it exits and prints what it would have launched)
#run_cmd="$PY_LAUNCHER $CMD"
# salloc --nodes=1 --gres=gpu:2 --cpus-per-gpu=20 -t 24:00:00
# clear; srun singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c '/home/ganruyi/anaconda3/bin/python $CMD'
# clear; srun singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH bash -c '/home/ganruyi/anaconda3/bin/python -u -m debugpy --listen 192.168.190.2:53005 --wait-for-client $CMD'