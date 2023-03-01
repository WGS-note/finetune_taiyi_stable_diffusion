docker stop jacktest
docker rm jacktest

prjPath=/data/renwanxin/txtdraw2022-develop
HF_HOME=/data/renwanxin/notebook/model/.cache/huggingface
HUGGING_FACE_HUB_TOKEN=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 # 有条件不要用我的token，\(•◡•)/


docker run --rm --name jacktest \
	-v ${prjPath}:/data \
	--ulimit memlock=-1 --ulimit stack=67108864 \
	-p 8001:6030 \
	-v ${HF_HOME}:${HF_HOME} \
	-e HF_HOME=${HF_HOME} \
	-e HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN} \
    -it jackoneflow:1.0 sh -c "cd /data && python -u /data/draw.py"
