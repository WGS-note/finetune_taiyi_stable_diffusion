+ finetune.py：微调太乙stable diffusion；
+ 数据位置：./data/naerdataset



```shell
docker run -d --gpus '"device=0,1,3"' \
       --rm -it --name text2image_test \
       --shm-size 12G \
       -v /data/wgs/text2img:/home \
       wgs-torch:3.0 \
       sh -c "python3 -u /home/finetune/finetune.py $options 1>>/home/log/finetune_test.log 2>>/home/log/finetune_test.err"
```

