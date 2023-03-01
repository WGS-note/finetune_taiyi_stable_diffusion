#!/bin/bash
today=$(date -d "now" +%Y-%m-%d)
yesterday=$(date -d "yesterday" +%Y-%m-%d)

cd /data/wgs/text2img

docker run -d --gpus '"device=0,1,2,3"' \
           --rm -it --name text2image_test \
           -v /data/wgs/text2img:/home \
           wgs-torch:2.0 \
           sh -c "python -u /home/run_zh.py 1>>/home/log/run.log 2>>/home/log/run.err"

docker run --gpus '"device=0,1,2,3"' \
           --rm -it --name text2image_test \
           -v /data/wgs/text2img:/home \
           wgs-torch:2.0 \
           bash

# taiyi sd zh
docker run -d --gpus '"device=0,1,2,3"' \
           --rm -it --name text2image_test \
           -v /data/wgs/text2img:/home \
           wgs-torch:2.0 \
           sh -c "python -u /home/run_taiyi_sd_zh.py 1>>/home/log/run_taiyi_sd_zh.log 2>>/home/log/run_taiyi_sd_zh.err"

# taiyi sd zh-en
docker run -d --gpus '"device=0,1,2,3"' \
           --rm -it --name text2image_test \
           -v /data/wgs/text2img:/home \
           wgs-torch:2.0 \
           sh -c "python -u /home/run_taiyi_sd_zh_en.py 1>>/home/log/run_taiyi_sd_zh_en.log 2>>/home/log/run_taiyi_sd_zh_en.err"

# waifu-diffusion en
docker run -d --gpus '"device=0,1,2,3"' \
           --rm -it --name text2image_test \
           -v /data/wgs/text2img:/home \
           wgs-torch:2.0 \
           sh -c "python -u /home/run_wd_en.py 1>>/home/log/run_wd_en.log 2>>/home/log/run_wd_en.err"

# run
docker run -d --gpus '"device=0,1,2,3"' \
           --rm -it --name text2image_test \
           -v /data/wgs/text2img:/home \
           wgs-torch:2.0 \
           sh -c "python -u /home/run.py 1>>/home/log/run.log 2>>/home/log/run.err"


# sh /data/wgs/text2img/dk/run.sh
# nohup sh /data/wgs/text2img/dk/run.sh &
# python -u /home/run_zh.py 1>>/home/log/run.log 2>>/home/log/run.err