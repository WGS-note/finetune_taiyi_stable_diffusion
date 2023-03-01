# 133 GPU

#FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

#RUN echo "" > /etc/apt/sources.list.d/cuda.list
#RUN sed -i "s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list
#RUN sed -i "s@/security.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list
#RUN apt-get update --fix-missing && apt-get install -y fontconfig --fix-missing
#RUN apt-get install -y cmake
#RUN apt-get install -y python3.7 python3-pip
#
#RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone && \
#    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy matplotlib seaborn h5py sklearn numpy==1.20.3 pandas==1.3.5
#
##RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchtext==0.11.0
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ftfy==6.1.1 jieba==0.42.1
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboardX==2.5 tensorboard==2.2.2
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple datasets==2.6.1 protobuf==3.19.0
#
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple huggingface_hub==0.10.1 transformers==4.23.1
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple diffusers==0.6.0
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple accelerate==0.11.0 modelcards==0.1.6
#
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytorch_lightning==1.8.1
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lightning==1.8.1
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple deepspeed==0.7.5
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx==1.12.0 onnxruntime-gpu==1.12
#
#WORKDIR /home
# cd /data/wgs/text2img
# docker build -t wgs-torch:3.0 -f ./dk/Dockerfile .
# docker run --gpus '"device=0,1,2,3"' --rm -it --name test wgs-torch:3.0 bash

# ------------------------------

#FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
#
#RUN echo "" > /etc/apt/sources.list.d/cuda.list
#RUN sed -i "s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list
#RUN sed -i "s@/security.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list
#RUN apt-get update --fix-missing && apt-get install -y fontconfig --fix-missing
#RUN apt-get install -y cmake
#RUN apt-get install -y vim
##RUN apt-get install -y git
#
#RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone && \
#    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scipy matplotlib seaborn h5py sklearn numpy==1.20.3 pandas==1.3.5
#
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ftfy==6.1.1 jieba==0.42.1
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorboardX==2.5.1 tensorboard==2.11.0
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple datasets==2.7.1 protobuf==3.20.1
#
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple huggingface_hub==0.11.1 transformers==4.25.1
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple diffusers==0.9.0
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple accelerate==0.15.0 modelcards==0.1.6
#
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytorch_lightning==1.8.1
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lightning==1.8.1
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple deepspeed==0.7.6
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnx==1.12.0 onnxruntime-gpu==1.13.1
#
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple markupsafe==2.0.1
#
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple SentencePiece
#
#WORKDIR /home
#
## docker build -t wgs-torch:6.1 -f ./dk/Dockerfile .

# ------------------------------

#FROM wgs-torch:6.1
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple SentencePiece
#WORKDIR /home
#
## docker build -t wgs-torch:6.2 -f ./dk/Dockerfile .


# ------------------------------

#FROM wgs-torch:6.1
#RUN pip install rouge
#WORKDIR /home
#
## docker build -t wgs-torch:6.2 -f ./dk/Dockerfile .

# ------------------------------

#FROM wgs-torch:3.0
#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade transformers diffusers
## ==4.26.1
#WORKDIR /home
#
## docker build -t wgs-torch:3.1 -f ./dk/Dockerfile .

# ------------------------------

FROM wgs-torch:3.0

#RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
RUN apt-get install libgl1-mesa-glx -y

WORKDIR /home

# docker build -t wgs-torch:3.2 -f ./dk/Dockerfile .
# 改名 wgs-torch:3.0




