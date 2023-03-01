# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2023/2/2 15:36
# @File: run_qa.py
'''
中文生成式问答
https://huggingface.co/IDEA-CCNL/Randeng-T5-784M-QA-Chinese
'''







if __name__ == '__main__':
    '''
    docker run -d --gpus '"device=0,1,2,3"' \
               --rm -it --name nlg_text \
               -v /data/wgs/text2img:/home \
               wgs-torch:6.2 \
               sh -c "python -u /home/nlg_text/run.py 1>>/home/log/nlg_text.log 2>>/home/log/nlg_text.err"
    '''