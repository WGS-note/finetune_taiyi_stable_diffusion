#!/bin/bash
today=$(date -d "now" +%Y-%m-%d)
yesterday=$(date -d "yesterday" +%Y-%m-%d)

cd /data/wgs/text2img/pre_models

git lfs clone https://huggingface.co/IDEA-CCNL/Randeng-TransformerXL-5B-Deduction-Chinese

git lfs clone https://huggingface.co/IDEA-CCNL/Randeng-TransformerXL-5B-Abduction-Chinese

# nohup sh /data/wgs/text2img/dk/clone_xl.sh &>/data/wgs/text2img/log/clone_xl.log &