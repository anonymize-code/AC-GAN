import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
import seaborn as sn
import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from torchvision import datasets, transforms
from glob import glob
# import ipdb

torch.set_default_tensor_type(torch.DoubleTensor)
from torchvision.utils import save_image
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler(feature_range=(-1, 1))



class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    epoch = 800
    alpha = 3e-5
    print_per_step = 100  # 控制输出
    z_dim = 32
    load_path="../nk_test/data/clean_bvp_all/bvp_train_1_mm.csv"
    save_path="results/wgan/UBFC/signal_img_1/base_wgan"




# 定义判别器  #####Discriminator######使用多层网络来作为判别器
# 将图片利用LeNet网络进行二分类，判断图片是真实的还是生成的
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.f1 = nn.Sequential(
            nn.Linear(192, 512),
            nn.LeakyReLU(inplace=True)
            # nn.Dropout(0.5)
        )
        self.f2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(inplace=True)
            # nn.Dropout(0.5)
        )
        self.f3 = nn.Sequential(
            nn.Linear(1024, 1280),
            nn.LeakyReLU(inplace=True)
            # nn.Dropout(0.5)
        )
        self.f4 = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.LeakyReLU(inplace=True)
            # nn.Dropout(0.5)
        )
        self.f5 = nn.Sequential(
            nn.Linear(1280, 256),
            nn.LeakyReLU(inplace=True)
            # nn.Dropout(0.5)
        )
        self.out = nn.Sequential(
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.out(x)
        return x


# 定义判别器  #####Generator######使用多层网络来作为判别器
# 输入一个100维的0～1之间的高斯分布，多层映射到784维

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(32,192),
            nn.LeakyReLU(inplace=True),
            nn.Linear(192, 256),  # 用线性变换将输入映射到256维
            nn.LeakyReLU(inplace=True),  # relu激活
            nn.Linear(256, 512),  # 线性变换
            nn.LeakyReLU(inplace=True),  # relu激活
            nn.Linear(512, 1024),  # 线性变换
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 256),  # 线性变换
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),  # 线性变换
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 192),  # 线性变换
            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间，因为输入的真实数据的经过transforms之后也是这个分布
        )

    def forward(self, x):
        x = self.gen(x)
        return x


class dataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, file_path):
        xy = np.loadtxt(file_path, delimiter=',')  # 使用numpy读取数据
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        # self.x_data.to(torch.double)
        # self.y_data.to(torch.double)
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        # return self.x_data[index]

    def __len__(self):
        return self.len


def gradient_penalty(model, real_images, fake_images):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0), 1, 1, 1))
    # Get random interpolation between real and fake data
    alpha = alpha.cuda()
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
    interpolates = interpolates

    model_interpolates = model(interpolates)
    grad_outputs = torch.ones(model_interpolates.size(), requires_grad=False).cuda()

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty


class TrainProcess:
    def __init__(self):
        self.data = self.load_data()
        self.D = discriminator().to(Config.device)
        self.G = generator().to(Config.device)
        # self.D.load_state_dict(torch.load("results/wgan/signal_img_1/test3/D_parameter119.pkl"))
        # self.G.load_state_dict(torch.load("results/wgan/signal_img_1/test3/G_parameter119.pkl"))
        self.criterion = nn.BCELoss()  # 定义损失函数
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=1e-3, betas=(0.5, 0.9))
        self.d_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer, gamma=0.99)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-3, betas=(0.5, 0.9))
        self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, gamma=0.99)

    @staticmethod
    def load_data():
        train_dataset = dataset(Config.load_path)

        # 返回一个数据迭代器
        # shuffle：是否打乱顺序
        data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                  batch_size=Config.batch_size,
                                                  shuffle=True,
                                                  drop_last=True)

        return data_loader

    def train_step(self):
        dloss_list = []
        gloss_list = []
        one = torch.tensor(1).cuda()
        loss_fn = torch.nn.MSELoss()
        mone = one * -1
        # ##########################进入训练##判别器的判断过程#####################
        for epoch in range(Config.epoch):  # 进行多个epoch的训练
            for i, (signal, _) in enumerate(self.data):
                num_signal = signal.size(0)

                for p in self.D.parameters():
                    p.requires_grad_ = True
                for iter_d in range(5):
                    signal = signal.view(num_signal, -1)
                    real_signal = Variable(signal)
                    real_signal.requires_grad_(True)
                    if (Config.device != 'cpu'):
                        real_signal = real_signal.cuda()
                    self.D.zero_grad()
                    real_out = self.D(real_signal)
                    real_scores = real_out
                    d_loss_real = torch.mean(real_out)
                    # d_loss_real.backward(mone)

                    z = Variable(torch.randn(num_signal, Config.z_dim))
                    if (Config.device != 'cpu'):
                        z = z.cuda()
                    # 随机生成一些噪声
                    fake_signal = self.G(z)
                    fake_out = self.D(fake_signal)
                    fake_scores = fake_out
                    d_loss_fake = torch.mean(fake_out)
                    # d_loss_fake.backward(one)

                    gp = gradient_penalty(self.D, real_signal, fake_signal)
                    gp.requires_grad_(True)
                    # gp.backward()

                    d_loss = d_loss_fake - d_loss_real + gp
                    d_loss.backward()
                    self.d_optimizer.step()  # 更新参数
                    # self.d_scheduler.step()

                for p in self.D.parameters():
                    p.requires_grad_ = False
                self.G.zero_grad()
                z = Variable(torch.randn(num_signal, Config.z_dim))  # 得到随机噪声
                if (Config.device != 'cpu'):
                    z = z.cuda()
                fake_signal = self.G(z)  # 随机噪声输入到生成器中，得到一副假的图片
                output = self.D(fake_signal)  # 经过判别器得到的结果
                output = torch.mean(output)
                gloss=-output
                gloss.backward()
                # output.backward(mone)
                g_loss = -(torch.mean(output))  # 得到的假的图片与真实的图片的label的loss
                self.g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数
                # self.g_scheduler.step()

            if epoch == 0:
                # real_images = self.to_signal(real_signal.cpu().data)
                # a=real_images[1]
                a = real_signal.detach().cpu().numpy()
                signals = pd.DataFrame({
                    "bvp": a[0]
                })
                signals.plot()
                plt.savefig(Config.save_path+'/real_images.png')
                plt.close()
            fake_signal_cpu = fake_signal.cpu()
            a = fake_signal_cpu.detach().numpy()
            signals = pd.DataFrame({
                "bvp": a[0]
            })
            signals.plot()
            plt.savefig(Config.save_path+'/fake_images-{}.png'.format(epoch))
            plt.close()
            # if epoch % 10 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                    'D real: {:.6f},D fake: {:.6f},GP:{:.6f}'.format(
                epoch, Config.epoch, d_loss.data.item(), g_loss.data.item(),
                real_scores.data.mean(), fake_scores.data.mean(), gp  # 打印的是真实图片的损失均值
            ))
            dloss_list.append(d_loss.data.item())
            gloss_list.append(g_loss.data.item())

            # save_image(fake_signal, './signal_img/fake_images-{}.png'.format(epoch + 1))
            if ((epoch + 1) % 40 == 0):
                torch.save(self.G.state_dict(), Config.save_path+"/G_parameter" + str(epoch) + ".pkl")
                torch.save(self.D.state_dict(), Config.save_path+"/D_parameter" + str(epoch) + ".pkl")
                loss_plot = pd.DataFrame({
                    "dloss": dloss_list,
                    "gloss": gloss_list
                })
                loss_plot.plot()
                plt.savefig(Config.save_path+'/loss' + str(epoch) + '.png')


def create_signal(model_path, epoch):
    generate = generator().to(Config.device)
    generate.load_state_dict(torch.load(model_path))
    for i in range(epoch):
        Z = Variable(torch.randn(Config.z_dim, 1)).cuda()

        fake_signal = generate(Z.T).cpu().detach().numpy().T
        fake_signal = fake_signal.reshape(-1, 1)
        fake_signal = mm.fit_transform(fake_signal)
        fake_signal = fake_signal.reshape(1, -1)
        # signals = pd.DataFrame({
        #     "bvp": fake_signal[0,:]
        # })
        # signals.plot()
        # plt.show(block=False)
        # plt.close()
        if i == 0:
            res = fake_signal
        else:
            res = np.r_[res, fake_signal]
        if (i % 100 == 0):
            print('epoch=' + str(i))

    np.savetxt('./results/generate/UBFC_WGAN_test1_3000_state_1.csv', res, delimiter=',')


if __name__ == "__main__":
    # 创建文件夹
    # create_signal('results/wgan/UBFC/signal_img_1/WGAN_test1/G_parameter799.pkl', 3000)
    p = TrainProcess()
    p.train_step()
