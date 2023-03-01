FROM oneflowinc/oneflow-sd:cu112

ENV LANG C.UTF-8
ENV TZ Asia/Shanghai
RUN echo "" > /etc/apt/sources.list.d/cuda.list \
&&  sed -i "s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list \
&&  sed -i "s@/security.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list \
&&  apt-get update \
&&  DEBIAN_FRONTEND=noninteractive apt-get install tzdata \
&&  ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
&&  echo $TZ > /etc/timezone
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 

# 去除默认配置
RUN echo ""> /opt/conda/pip.conf && \
    echo ""> /root/.pip/pip.conf  && \
    echo "" > /root/.config/pip/pip.conf  && \
    echo "" > /etc/pip.conf  && \
    echo "" > /etc/xdg/pip/pip.conf

# 这里结尾加个正常退出，会报错protobuf版本不兼容，但验证uwsgi和diffusers可用
RUN python3.8 -m pip install -i http://10.16.0.32:6000/offline4pip3/simple/ --trusted-host 10.16.0.32 \
    flask-cors==3.0.10 pyarrow==10.0.0 \
    requests==2.26.0 Flask==2.0.1 \
    ; exit 0

RUN python3.8 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
    pymongo==3.11.3 oss2==2.16.0 kafka-python==1.4.7 \
    datasets==2.6.1 accelerate==0.11.0 tensorboard==2.2.2 modelcards==0.1.6 ftfy==6.1.1 redis==4.3.0 \
&&  apt-get clean

RUN conda install -c conda-forge uwsgi -y

ADD src /txtdraw
WORKDIR /txtdraw

#CMD uwsgi --http 0.0.0.0:6030 --ini uwsgi.ini --procname txtdraw
CMD ["sh", "launch.sh"]


# 以下基础镜像可能已经包含，如果缺少自行添加
# protobuf==3.19.0 pandas==1.3.5 jieba==0.42.1 numpy==1.23.4 uwsgi==2.0.18 torch==1.9.0 torchvision==0.10.0 diffusers==0.6.0  huggingface_hub==0.10.1 transformers==4.23.1 