#!/bin/bash
today=$(date -d "now" +%Y-%m-%d)
yesterday=$(date -d "yesterday" +%Y-%m-%d)

cd /Users/wangguisen/Documents/markdowns/AI-note/元宇宙/text2img/

MICRO_BATCH_SIZE=1

DATA_ARGS="\
        --dataloader_workers 3 \
        --train_batchsize $MICRO_BATCH_SIZE  \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        --datasets_path ./data/svg_data/ \
        --datasets_type txt \
        --resolution 200 \
        "

# --datasets_path ./data/naer_test/ \
# --datasets_path ./data/app_data/ \
# --resolution 512 \

sh -c "python3 -u ./finetune/demo1.py $DATA_ARGS 1>>./log/demo.log 2>>./log/demo.err"


