# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2023/2/1 15:02
# @File: run.py
'''
生成式文本摘要
https://huggingface.co/IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese
https://huggingface.co/IDEA-CCNL/Randeng_Pegasus_523M/tree/main
https://huggingface.co/IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1
'''
from transformers import PegasusForConditionalGeneration
import time
from tokenizers_pegasus import PegasusTokenizer

modle_path = '/home/pre_models/Randeng-Pegasus-238M-Summary-Chinese'
# modle_path = '/Users/wangguisen/Documents/markdowns/AI-note/元宇宙/pre_models/Randeng-Pegasus-238M-Summary-Chinese'

model = PegasusForConditionalGeneration.from_pretrained(modle_path)
tokenizer = PegasusTokenizer.from_pretrained(modle_path)

def summari_generate(text):
    start_time = time.time()

    inputs = tokenizer(text, max_length=1024, return_tensors="pt")

    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"])
    output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    end_time = time.time()
    print('inference run time: {:.0f}分 {:.4f}秒'.format((end_time - start_time) // 60, (end_time - start_time) % 60))

    return output

if __name__ == '__main__':

    '''
    docker run -d --gpus '"device=0,1,2,3"' \
               --rm -it --name summarization \
               -v /data/wgs/text2img:/home \
               wgs-torch:6.2 \
               sh -c "python -u /home/text_summarization/run.py 1>>/home/log/text_summarization.log 2>>/home/log/text_summarization.err"
    '''

    text1 = "在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！"
    print('out1 ---------- \n', summari_generate(text1), '\n')

    text2 = "ChatGPT的横空出世，在人工智能领域掀起了重要变革，这一智能工具因其解放人类生产力的潜力，从使用者到投资者，引起了各界的广泛关注。"
    print('out2 ---------- \n', summari_generate(text2), '\n')

    text3 = "2月1日，涉事的无锡动物园负责人表示，这只猴子是和别的猴子打架输了，出于求生的目的跳过电网出逃。猴子打架看起来，还挺有趣的！在工作人员捉猴子过程中，猴子跑到了狮子园外场，中间还隔着电网和一条小河，不存在被狮子攻击的情况，最后将猴子成功抓回了猴山。"
    print('out3 ---------- \n', summari_generate(text3), '\n')

    text4 = "ChatGPT的论文尚未放出，也不知道会不会有论文放出，但是根据公开资料显示，其训练方式，跟OpenAI之前的一个工作InstructGPT基本无异，主要是训练数据上有小的差异，因此我们可以从InstructGPT的论文中，窥探ChatGPT强大的秘密。"
    print('out4 ---------- \n', summari_generate(text4), '\n')

    text5 = "针对传统的流量分类管理系统存在不稳定、结果反馈不及时、分类结果显示不直观等问题,设计一个基于web的在线的流量分类管理系统.该系统采用流中前5个包(排除3次握手包)所含信息作为特征值计算资源,集成一种或多种分类算法用于在线网络流量分类,应用数据可视化技术处理分类结果。"
    print('out5 ---------- \n', summari_generate(text5), '\n')

    text6 = "山东省郯城县一名男子与他人在饭店饮酒，连续喝了两场，结果因饮酒过量乙醇中毒，不幸死亡。郯城县人民法院依法判决6名共饮的“酒友”承担部分责任，共同赔偿死者亲属死亡赔偿金等损失279501元，“酒友”们不服上诉。日前，临沂市中级人民法院作出二审裁判，驳回上诉，维持原判。"
    print('out6 ---------- \n', summari_generate(text6), '\n')

    text7 = "镇江市市场监管局认为，自2022年12月开始，鱼跃医疗利用市场供需紧张状况，在血氧仪生产入库平均成本环比上涨47%的情况下，大幅度提高该产品销售价格，平均销售价格环比上涨了131.78%，销售价格上涨幅度明显高于成本增长幅度，推动了血氧仪市场价格过快、过高上涨，扰乱市场价格秩序。"
    print('out7 ---------- \n', summari_generate(text7), '\n')

    text8 = "1月31日，广东一女子私自坐上警察摩托车拍照，被多次警告仍然无动于衷，视频中女子大声辩解只是坐一下而已，警察警告称妨碍正常执法是有权告你的，女子依旧我行我素，还对警察说：喜欢你，警察无奈将车钥匙拔下再次警告她，女子不悔改还不断撒娇卖萌。最终对该女子采取强制措施，依法将其带离现场。网友：别拿无知当可爱。"
    print('out8 ---------- \n', summari_generate(text8), '\n')

    text9 = "68岁老人开车撞上路边的奥迪，老伴下车后，直接秒晕倒！这一波操作，估计娱乐圈的演员都要自叹不如，娱乐圈的人不得不感慨自愧不如啊！"
    print('out9 ---------- \n', summari_generate(text9), '\n')

    text10 = "韩国总统尹锡悦1月11日首次明确宣称“韩国可能拥有自己的核武器”后，韩国社会对于“独立开发核武器”的议论持续升温。韩国《朝鲜日报》1月31日称，民调显示，76%的韩国民众表示“韩国需要独立开发核武器”。韩国对于“拥核”的支持声高涨也引起美国高度关注，一些媒体对韩国是否有能力独立研制核武器进行了详细评估。"
    print('out10 ---------- \n', summari_generate(text10), '\n')













