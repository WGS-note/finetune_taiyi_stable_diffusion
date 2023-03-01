# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/11/3 5:17 下午
# @File: get_params.py
'''
不用传输入的模型参数量计算
'''
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, CLIPTokenizer, BertForTokenClassification
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *
import torchkeras.summary as summary

def model_structure(model):

    netLayerCount = 0
    blank = ' '
    # print('-' * 90)
    # print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
    #       + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
    #       + ' ' * 3 + 'number' + ' ' * 3 + '|')
    # print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):

        splay = key.split('.')
        if ('bias' not in splay) and ('key' not in splay) and ('value' not in splay) and ('to_k' not in splay) and ('to_v' not in splay):
            netLayerCount += 1

        # if len(key) <= 30:
        #     key = key + (30 - len(key)) * blank
        # shape = str(w_variable.shape)
        # if len(shape) <= 40:
        #     shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        # print('| {} | {} | {} |'.format(key, shape, str_num))

    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

    # B：10亿
    _b = num_para * type_size / 1000000000

    return num_para, num_para * type_size / 1000 / 1000, _b, netLayerCount

if __name__ == '__main__':
    print()

    text_encoder = BertForTokenClassification.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/text_encoder')
    _, _, _, nl1 = model_structure(text_encoder)
    vae = AutoencoderKL.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/', subfolder="vae")
    _, _, _, nl2 = model_structure(vae)
    unet = UNet2DConditionModel.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/', subfolder="unet")
    _, _, _, nl3 = model_structure(unet)
    sdsc = StableDiffusionSafetyChecker.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/', subfolder='safety_checker')
    _, _, _, nl4 = model_structure(sdsc)
    print(nl1 + nl2 + nl3 + nl4)

    # tokenizer = CLIPTokenizer.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/waifu-diffusion/tokenizer', return_dict=False)
    # out = tokenizer('The Summer Palace is so beautiful in autumn')
    # print(type(out))
    # print(out)


    '''   waifu-diffusion   '''
    # text_encoder = BertForTokenClassification.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/waifu-diffusion/text_encoder')
    # vae = AutoencoderKL.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/waifu-diffusion/vae', subfolder="vae")
    # unet = UNet2DConditionModel.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/waifu-diffusion/unet', subfolder="unet")
    # sdsc = StableDiffusionSafetyChecker.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/waifu-diffusion/safety_checker')
    # # cfe = CLIPFeatureExtractor.from_pretrained('')
    #
    # bert_tolparas, bert_M, bert_b = model_structure(text_encoder)
    # vae_tolparas, vae_M, vae_b = model_structure(vae)
    # unet_tolparas, unet_M, unet_b = model_structure(unet)
    # sdsc_tolparas, sdsc_M, sdsc_b = model_structure(sdsc)
    #
    # print('总参数量：', bert_tolparas + vae_tolparas + unet_tolparas + sdsc_tolparas)
    # print('单位B：', bert_b + vae_b + unet_b + sdsc_b)
    '''
    总参数量： 1370219969
    单位B： 1.370219969 B
    '''

    '''   Taiyi-Stable-Diffusion 中文  '''
    # text_encoder = BertForTokenClassification.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/text_encoder')
    # vae = AutoencoderKL.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/vae', subfolder="vae")
    # unet = UNet2DConditionModel.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/unet', subfolder="unet")
    # sdsc = StableDiffusionSafetyChecker.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/Taiyi-Stable-Diffusion-1B-Chinese-v0.1/safety_checker')
    #
    # bert_tolparas, bert_M, bert_b = model_structure(text_encoder)
    # vae_tolparas, vae_M, vae_b = model_structure(vae)
    # unet_tolparas, unet_M, unet_b = model_structure(unet)
    # sdsc_tolparas, sdsc_M, sdsc_b = model_structure(sdsc)
    #
    # print('总参数量：', bert_tolparas + vae_tolparas + unet_tolparas + sdsc_tolparas)
    # print('单位B：', bert_b + vae_b + unet_b + sdsc_b)
    '''
    总参数量： 1348835009
    单位B： 1.348835009 B
    '''

    '''   Taiyi-Stable-Diffusion 中英混合   '''
    # text_encoder = BertForTokenClassification.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/text_encoder')
    # vae = AutoencoderKL.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/vae', subfolder="vae")
    # unet = UNet2DConditionModel.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/unet', subfolder="unet")
    # sdsc = StableDiffusionSafetyChecker.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/Taiyi-Stable-Diffusion-1B-Chinese-EN-v0.1/safety_checker')
    #
    # bert_tolparas, bert_M, bert_b = model_structure(text_encoder)
    # vae_tolparas, vae_M, vae_b = model_structure(vae)
    # unet_tolparas, unet_M, unet_b = model_structure(unet)
    # sdsc_tolparas, sdsc_M, sdsc_b = model_structure(sdsc)
    #
    # print('总参数量：', bert_tolparas + vae_tolparas + unet_tolparas + sdsc_tolparas)
    # print('单位B：', bert_b + vae_b + unet_b + sdsc_b)
    '''
    总参数量： 1370219969
    单位B： 1.370219969 B
    '''

    '''   svjack-Stable-Diffusion 中文   '''
    # text_encoder = BertForTokenClassification.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/Stable-Diffusion-Pokemon-zh/text_encoder')
    # vae = AutoencoderKL.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/Stable-Diffusion-Pokemon-zh/vae', subfolder="vae")
    # unet = UNet2DConditionModel.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/Stable-Diffusion-Pokemon-zh/unet', subfolder="unet")
    # sdsc = StableDiffusionSafetyChecker.from_pretrained("/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/stable-diffusion-safety-checker")
    # # cfe = CLIPFeatureExtractor.from_pretrained("/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/clip-vit-base-patch32")
    #
    # bert_tolparas, bert_M, bert_b = model_structure(text_encoder)
    # vae_tolparas, vae_M, vae_b = model_structure(vae)
    # unet_tolparas, unet_M, unet_b = model_structure(unet)
    # sdsc_tolparas, sdsc_M, sdsc_b = model_structure(sdsc)
    # # cfe_tolparas, cfe_M, cfe_b = model_structure(cfe)
    #
    # print('总参数量：', bert_tolparas + vae_tolparas + unet_tolparas + sdsc_tolparas)
    # print('单位B：', bert_b + vae_b + unet_b + sdsc_b)
    '''
    总参数量： 1349227199
    单位B： 1.3492271989999999 B
    '''

    '''   Runwayml-Stable_Diffusion-v1-5   '''
    # text_encoder = BertForTokenClassification.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/stable-diffusion-v1-5/text_encoder')
    # vae = AutoencoderKL.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/stable-diffusion-v1-5/vae', subfolder="vae")
    # unet = UNet2DConditionModel.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/stable-diffusion-v1-5/unet', subfolder="unet")
    # sdsc = StableDiffusionSafetyChecker.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/stable-diffusion-v1-5/safety_checker')
    #
    # bert_tolparas, bert_M, bert_b = model_structure(text_encoder)
    # vae_tolparas, vae_M, vae_b = model_structure(vae)
    # unet_tolparas, unet_M, unet_b = model_structure(unet)
    # sdsc_tolparas, sdsc_M, sdsc_b = model_structure(sdsc)
    #
    # print('总参数量：', bert_tolparas + vae_tolparas + unet_tolparas + sdsc_tolparas)
    # print('单位B：', bert_b + vae_b + unet_b + sdsc_b)
    '''
    总参数量： 1370219969
    单位B： 1.370219969
    '''

    '''   Nitrosocke-Mo_Di_Diffusion   '''
    # text_encoder = BertForTokenClassification.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/mo-di-diffusion/text_encoder')
    # vae = AutoencoderKL.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/mo-di-diffusion/vae', subfolder="vae")
    # unet = UNet2DConditionModel.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/mo-di-diffusion/unet', subfolder="unet")
    # sdsc = StableDiffusionSafetyChecker.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/mo-di-diffusion/safety_checker')
    #
    # bert_tolparas, bert_M, bert_b = model_structure(text_encoder)
    # vae_tolparas, vae_M, vae_b = model_structure(vae)
    # unet_tolparas, unet_M, unet_b = model_structure(unet)
    # sdsc_tolparas, sdsc_M, sdsc_b = model_structure(sdsc)
    #
    # print('总参数量：', bert_tolparas + vae_tolparas + unet_tolparas + sdsc_tolparas)
    # print('单位B：', bert_b + vae_b + unet_b + sdsc_b)
    '''
    总参数量： 1370219969
    单位B： 1.370219969
    '''

    '''   CompVis-ldm_text2im_large   '''
    # text_encoder = BertForTokenClassification.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/ldm-text2im-large-256/bert')
    # vae = AutoencoderKL.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/ldm-text2im-large-256/vqvae', subfolder="vae")
    # unet = UNet2DConditionModel.from_pretrained('/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/ldm-text2im-large-256/unet', subfolder="unet")
    #
    # bert_tolparas, bert_M, bert_b = model_structure(text_encoder)
    # vae_tolparas, vae_M, vae_b = model_structure(vae)
    # unet_tolparas, unet_M, unet_b = model_structure(unet)
    #
    # print('总参数量：', bert_tolparas + vae_tolparas + unet_tolparas)
    # print('单位B：', bert_b + vae_b + unet_b)
    '''
    总参数量： 1206270893
    单位B： 1.2062708930000001
    '''



