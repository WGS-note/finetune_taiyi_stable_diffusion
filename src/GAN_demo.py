# coding:utf-8
# @Email: wangguisen@infinities.com.cn
# @Time: 2022/11/9 3:44 下午
# @File: GAN_demo.py
'''
基于手写数字识别数据的 GAN demo
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms

'''
使用手写数字识别为例
现将数据归一化到（-1,1）
  其和GAN的训练技巧有关，对于生成器最后使用tanh激活，tanh的取值范围就是（-1,1），
  为了方便生成的图片和输入噪声取值范围相同，所以将输入归一化到（-1,1）
'''
# 对数据归一化（-1，1）
transform = transforms.Compose([
    transforms.ToTensor(),            # （0，1），(channel, h, w)
    transforms.Normalize(0.5, 0.5)    # 均值、方差为0.5后，将（0，,1）变为（-1，,1）？？？
])

train_ds = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)

dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

# imgs, labels = next(iter(dataloader))
# print(imgs.shape)

'''   定义生成器   '''
'''
基于这个例子，输入为长度100的正态分布噪声
输出维度为(1, 28, 28)
'''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linears = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        # x 为长度为100的noise
        out = self.linears(x)
        out = out.view(-1, 28, 28, 1)
        return out

'''   定义判别器   '''
'''
输入维度为（1，,28，,28）
'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linears = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.linears(x)

'''   初始化模型、优化器、损失   '''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gen = Generator().to(device)
dis = Discriminator().to(device)

g_optim = torch.optim.Adam(gen.parameters(), lr=0.0001)
d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)

loss_fn = nn.BCELoss()


'''   绘图函数  '''
def gen_img_plot(net, test_input):
    prediction = np.squeeze(net(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow((prediction[i] + 1) / 2)
        plt.axis('off')
    plt.show()

test_input = torch.randn(16, 100, device=device)

'''   训练   '''
D_loss = []
G_loss = []
for epoch in range(20):
    d_epoch_loss = 0
    g_epoch_loss = 0
    count = len(dataloader)
    for step, (img, label) in enumerate(dataloader):
        img = img.to(device)
        size = img.size(0)
        random_noise = torch.randn(size, 100, device=device)

        '''判别器优化'''
        d_optim.zero_grad()
        # 判别器输入真实的图片，得到对真实图片的预测结果
        real_output = dis(img)
        # 判别器在真实图片上的损失
        d_real_loss = loss_fn(real_output, torch.ones_like(real_output))    # 希望判别器将真实的数据判别为全1
        d_real_loss.backward()

        # 判别器输入生成的图片，得到判别器在生成图像上的损失
        gen_img = gen(random_noise)
        fake_output = dis(gen_img.detach())   # 对于生成图片产生的损失，我们的优化目标是判别器，希望fake_output被判定为0，来优化判别器，所以要截断梯度，detach会得到一个没有tensor的梯度
        d_fake_loss = loss_fn(fake_output, torch.zeros_like(fake_output))   # 希望判别器将生成的数据判别为全0
        d_fake_loss.backward()

        # 判别器总损失
        d_loss = d_real_loss + d_fake_loss
        d_optim.step()

        '''生成器优化'''
        g_optim.zero_grad()
        fake_output = dis(gen_img)   # 优化生成器，所以不用截断 detach
        # 对于生成器，希望生成的图片判定为1
        g_loss = loss_fn(fake_output, torch.ones_like(fake_output))   # 生成器的损失
        g_loss.backward()
        g_optim.step()

        with torch.no_grad():
            d_epoch_loss += d_loss
            g_epoch_loss += g_loss

    with torch.no_grad():
        d_epoch_loss /= count
        g_epoch_loss /= count
        D_loss.append(d_epoch_loss)
        G_loss.append(g_epoch_loss)
        print('Epoch: ', epoch)
        gen_img_plot(net=gen, test_input=test_input)



