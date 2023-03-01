prjPath=${PWD}

# 模型的缓存地址
HF_HOME=/data/wgs/notebook/model/.cache/huggingface

# hugging face的token
HUGGING_FACE_HUB_TOKEN=XXX


docker run --name jackoneflow --rm \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ${HF_HOME}:${HF_HOME} \
  -v ${prjPath}:${prjPath} \
  -w ${prjPath} \
  -e HF_HOME=${HF_HOME} \
  -e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} \
  -it oneflowinc/oneflow-sd:cu112 bash -c "pip install -i https://mirrors.aliyun.com/pypi/simple/ accelerate && python -u main.py"