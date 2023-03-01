# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/12/21 10:08
# @File: causal.py
'''
基于燃灯系列的因果推理测试
演绎推理: https://huggingface.co/IDEA-CCNL/Randeng-TransformerXL-5B-Deduction-Chinese
反绎推理: https://huggingface.co/IDEA-CCNL/Randeng-TransformerXL-5B-Abduction-Chinese
'''

import os, time, sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import warnings
from typing import List, Union
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer
from fengshen.models.transfo_xl_reasoning import TransfoXLModel
from fengshen.models.transfo_xl_reasoning import abduction_generate, deduction_generate
from fengshen.utils import sample_sequence_batch

warnings.filterwarnings("ignore")

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
device_ids = [0, 1, 2, 3]

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

def en_to_zh(sentence:str):
    en_pun = u",.!?[]()<>\"\"''"
    zh_pun = u"，。！？【】（）《》“”‘’"
    table = {
        ord(f): ord(t) for f,t in zip(en_pun, zh_pun)
    }
    return sentence.translate(table)

class Causal():

    __support_models__ = {
        'deduction': '/home/pre_models/Randeng-TransformerXL-5B-Deduction-Chinese',
        'abduction': '/home/pre_models/Randeng-TransformerXL-5B-Abduction-Chinese',
    }

    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tokenizer = self.load()

    def load(self):
        if self.model_name == 'deduction':
            self.generate_flag = 'deduction'
        elif self.model_name == 'abduction':
            self.generate_flag = 'abduction'
        else:
            raise ValueError('unexpected parameter')

        start_time = time.time()

        self.model = TransfoXLModel.from_pretrained(self.__support_models__[self.model_name])
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.__support_models__[self.model_name],
            eos_token='<|endoftext|>',
            pad_token='<|endoftext|>',
            extra_ids=0
        )
        self.tokenizer.add_special_tokens({'bos_token': '<bos>'})

        end_time = time.time()
        print('Loading run time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))

        return self.model, self.tokenizer

    def generate(self, input_text: Union[str, List[str]], batch_size=1, device='cpu'):
        start_time = time.time()

        if self.generate_flag == 'deduction':
            res = deduction_generate(self.model, self.tokenizer, input_text, batch_size=batch_size, device=device)
        elif self.generate_flag == 'abduction':
            res = abduction_generate(self.model, self.tokenizer, input_text, batch_size=batch_size, device=device)
        else:
            raise ValueError('unexpected parameter')

        print(type(res))
        print(res)

        end_time = time.time()
        print('inference run time: {:.0f}分 {:.0f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
        print()


if __name__ == '__main__':

    '''
    docker run --rm -it -v /data/wgs/text2img:/home wgs-torch:6.1 bash
    apt-get install -y git git-lfs
    
    docker run -d --gpus '"device=0,1,2,3"' \
           --rm -it --name causal \
           -v /data/wgs/text2img:/home \
           wgs-torch:6.2 \
           sh -c "python -u /home/causal_deduction/causal.py 1>>/home/log/causal.log 2>>/home/log/causal.err"
    '''

    input_text = "玉米价格持续上涨"
    # input_texts = ["机器人统治世界", "玉米价格持续上涨"]

    causal = Causal(model_name='deduction')
    causal.generate(input_text=input_text, batch_size=1, device=device)



    


