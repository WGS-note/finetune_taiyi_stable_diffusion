# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2023/2/1 14:42
# @File: run.py
'''
语义纠错
https://huggingface.co/IDEA-CCNL/Randeng-Transformer-1.1B-Denoise
'''
import torch, time
from fengshen.models.transfo_xl_denoise.tokenization_transfo_xl_denoise import TransfoXLDenoiseTokenizer
from fengshen.models.transfo_xl_denoise.modeling_transfo_xl_denoise import TransfoXLDenoiseModel
from fengshen.models.transfo_xl_denoise.generate import denoise_generate

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

model_path = "/home/pre_models/Randeng-Transformer-1.1B-Denoise"
# model_path = "/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/Randeng-Transformer-1.1B-Denoise"

tokenizer = TransfoXLDenoiseTokenizer.from_pretrained(model_path)
model = TransfoXLDenoiseModel.from_pretrained(model_path)

# input_text = "凡是有成就的人, 都很严肃地对待生命自己的"
# # "有成就的人都很严肃地对待自己的生命。"

def senm_correct(input_text, device):
    start_time = time.time()

    res = denoise_generate(model, tokenizer, input_text, device=device)

    end_time = time.time()

    print(res)
    print('inference run time: {:.0f}分 {:.4f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))
    print()

    return res

if __name__ == '__main__':
    '''
    docker run -d --gpus '"device=0,1,2,3"' \
               --rm -it --name summarization \
               -v /data/wgs/text2img:/home \
               wgs-torch:6.2 \
               sh -c "python -u /home/text_summarization/run.py 1>>/home/log/text_summarization.log 2>>/home/log/text_summarization.err"
    '''

    text1 = "凡是有成就的人, 都很严肃地对待生命自己的"
    senm_correct(text1, device)

