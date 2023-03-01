# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2023/2/1 17:31
# @File: run.py
'''
文本生成
https://huggingface.co/IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese
https://huggingface.co/IDEA-CCNL/Wenzhong-GPT2-110M
'''
import torch, time
from transformers import GPT2Tokenizer, GPT2Model
from transformers import pipeline, set_seed

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

model_path = "/home/pre_models/Wenzhong2.0-GPT2-3.5B-chinese"

# tokenizer = GPT2Tokenizer.from_pretrained(model_path)
# model = GPT2Model.from_pretrained(model_path)
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)

set_seed(55)
generator = pipeline('text-generation', model=model_path, device=device)
print('----------')

def text_generator(text):
    start_time = time.time()

    res = generator(text, max_length=50, num_return_sequences=1)

    end_time = time.time()
    print(res)
    print('inference run time: {:.0f}分 {:.4f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

if __name__ == '__main__':
    '''
    docker run -d --gpus '"device=0,1,2,3"' \
               --rm -it --name nlg_text \
               -v /data/wgs/text2img:/home \
               wgs-torch:6.2 \
               sh -c "python -u /home/nlg_text/run.py 1>>/home/log/nlg_text.log 2>>/home/log/nlg_text.err"
    '''

    text = "北京位于"
    text_generator(text)

    text = "今天天气真好"
    text_generator(text)

    text = "今天天气怎么样？"
    text_generator(text)
