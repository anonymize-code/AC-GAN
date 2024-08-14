import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
torch.set_default_tensor_type(torch.DoubleTensor)
from sklearn.preprocessing import MinMaxScaler
import torch.distributions as dist

mm = MinMaxScaler(feature_range=(-1, 1))


class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = 177 # CLAS 256 WESAD 537 UBFC 177
    batch_size = 128
    epoch = 800
    alpha = 3e-5
    print_per_step = 100  # 控制输出
    z_dim = 32
    dataSet = 'UBFC'
    model_path = "results/BVPTests/bvp_ode"
    model_id = 6
    model_epoch = 450
    load_path = "datasets/bvp_data/" + dataSet + "_bvp_train_1_mm.csv"  # UBFC,WESAD,CLAS
    save_path = "results/wgan/" + dataSet + "/signal_img_1/ODE_mmd_test1"


def min_max_scaler(tensor, feature_range=(0, 1)):
    # 移动tensor到CUDA设备
    tensor = tensor.cuda()

    # 找到最小值和最大值
    min_val = tensor.min(dim=0, keepdim=True)[0]
    max_val = tensor.max(dim=0, keepdim=True)[0]

    # 缩放到 [0, 1]
    tensor_norm = (tensor - min_val) / (max_val - min_val)

    # 调整到 [feature_range[0], feature_range[1]]
    min_range, max_range = feature_range
    tensor_scaled = tensor_norm * (max_range - min_range) + min_range

    return tensor_scaled


def gaussian_kernel_matrix(x, y, sigma=1.0):
    """
    计算高斯核矩阵
    x, y 的形状应为 (batch_size, features, dim_features)
    """
    beta = 1. / (2. * sigma ** 2)
    dist_sq = torch.cdist(x.reshape(x.size(0), -1), y.reshape(y.size(0), -1), p=2).pow(2)
    return torch.exp(-beta * dist_sq)


def compute_mmd(x, y, sigma=1.0):
    """
    计算两个批次之间的MMD
    """
    # Compute the kernel matrices
    K_xx = gaussian_kernel_matrix(x, x, sigma)
    K_yy = gaussian_kernel_matrix(y, y, sigma)
    K_xy = gaussian_kernel_matrix(x, y, sigma)
    K_yx = gaussian_kernel_matrix(y, x, sigma)

    # Compute MMD
    mmd = K_xx.mean() + K_yy.mean() - K_xy.mean() - K_yx.mean()
    return mmd


from bptt import models


def load_model(model_id, epoch):
    model = models.Model()
    model.init_from_model_path(model_id, epoch=epoch)
    model.eval()
    return model


def is_model_id(path):
    """Check if path ends with a three digit, e.g. save/test/001 """
    run_nr = path.split('\\')[-1]
    three_digit_numbers = {str(digit).zfill(3) for digit in set(range(1000))}
    return run_nr in three_digit_numbers


def get_model_ids(path):
    """
    Get model ids from a directory by recursively searching all subdirectories for files ending with a number
    """
    assert os.path.exists(path)
    if is_model_id(path):
        model_ids = [path]
    else:
        all_subfolders = glob(os.path.join(path, '**/*'), recursive=True)
        model_ids = [file for file in all_subfolders if is_model_id(file)]
    assert model_ids, 'could not load from path: {}'.format(path)
    return model_ids


# 定义判别器  #####Discriminator######使用多层网络来作为判别器
# 将图片利用LeNet网络进行二分类，判断图片是真实的还是生成的
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.f1 = nn.Sequential(
            nn.Linear(192, 512),
            nn.LeakyReLU(inplace=True)
            # nn.Dropout(0.3)
        )
        self.f2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(inplace=True)
            # nn.Dropout(0.3)
        )
        self.f3 = nn.Sequential(
            nn.Linear(1024, 1280),
            nn.LeakyReLU(inplace=True)
            # nn.Dropout(0.3)
        )
        self.f4 = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.LeakyReLU(inplace=True)
            # nn.Dropout(0.3)
        )
        self.f5 = nn.Sequential(
            nn.Linear(1280, 256),
            nn.LeakyReLU(inplace=True)
            # nn.Dropout(0.3)
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
            nn.Linear(32, 192),
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
        xy = np.loadtxt(file_path, delimiter=',') [Config.start:,:] # 使用numpy读取数据
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



def gradient_penalty_3(model, real_images, fake_images, att_images):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.tensor([1.0, 1.0, 1.0])
    dirichlet_dist = dist.Dirichlet(alpha)
    samples = dirichlet_dist.sample([real_images.size(0)])
    eta_1, eta_2, eta_3 = samples[:, 0].unsqueeze(1).cuda(), samples[:, 1].unsqueeze(1).cuda(), samples[:, 2].unsqueeze(1).cuda()

    interpolated = eta_1 * real_images + eta_2 * fake_images + eta_3 * att_images

    interpolated = interpolated.cuda()

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = model(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(
                                        prob_interpolated.size()).cuda(),
                                    create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty


def gradient_penalty(model, real_images, fake_images):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake data
    eta = torch.FloatTensor(real_images.size(0), 1).uniform_(0, 1)
    # eta = torch.FloatTensor(real_images.size(0)).uniform_(0, 1)

    eta = eta.cuda()

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    interpolated = interpolated.cuda()

    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = model(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(
                                        prob_interpolated.size()).cuda(),
                                    create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return grad_penalty


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
        for p in model.parameters():
            p.requires_grad_ = False
        # ##########################进入训练##判别器的判断过程#####################
        for epoch in range(Config.epoch):  # 进行多个epoch的训练
            for i, (signal, _) in enumerate(self.data):
                num_signal = signal.size(0)
                signal = signal.view(num_signal, -1)
                real_signal = Variable(signal)
                real_signal.requires_grad_(True)
                if (Config.device != 'cpu'):
                    real_signal = real_signal.cuda()
                model.eval()
                with torch.no_grad():
                    att_real_signal, att_real_latent = model.generate_batch_free_trajectory(Variable(real_signal.double()).cuda(), 192)
                for p in self.D.parameters():
                    p.requires_grad_ = True
                for iter_d in range(5):
                    self.D.zero_grad()
                    real_out = self.D(real_signal)
                    real_scores = real_out
                    d_loss_real = torch.mean(real_out)

                    z = Variable(torch.randn(num_signal, Config.z_dim))
                    if (Config.device != 'cpu'):
                        z = z.cuda()
                    # 随机生成一些噪声
                    fake_signal = self.G(z)
                    fake_out = self.D(fake_signal)
                    fake_scores = fake_out
                    d_loss_fake = torch.mean(fake_out)

                    gp_3 = gradient_penalty_3(self.D, real_signal, fake_signal, att_real_signal)
                    # gp = gradient_penalty(self.D, real_signal, fake_signal)

                    gp_3.requires_grad_(True)
                    # gp.backward()

                    d_loss = d_loss_fake - d_loss_real + gp_3
                    d_loss.backward()
                    self.d_optimizer.step()  # 更新参数
                    # self.d_scheduler.step()

                # ==================训练生成器============================
                # ###############################生成网络的训练###############################
                # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
                # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
                # 反向传播更新的参数是生成网络里面的参数，
                # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
                # 这样就达到了对抗的目的
                # 计算假的图片的损失
                for p in self.D.parameters():
                    p.requires_grad_ = False
                signal = signal.view(num_signal, -1)
                real_signal = Variable(signal)
                self.G.zero_grad()
                z = Variable(torch.randn(num_signal, Config.z_dim))  # 得到随机噪声
                if (Config.device != 'cpu'):
                    z = z.cuda()
                fake_signal = self.G(z)  # 随机噪声输入到生成器中，得到fake_signal

                model.eval()
                with torch.no_grad():
                    att_fake_signal, att_fake_latent = model.generate_batch_free_trajectory(Variable(fake_signal.double()).cuda(), 192)
                    fake_latent = model.get_batch_latent_trajectory(Variable(fake_signal.double()).cuda())

                att_fake_latent = min_max_scaler(att_fake_latent, feature_range=(0, 1))
                fake_latent = min_max_scaler(fake_latent, feature_range=(0, 1))
                mmd_loss = compute_mmd(att_fake_latent, fake_latent)
                g_loss2 = 20*mmd_loss
                output = self.D(fake_signal)  # 经过判别器得到的结果
                g_loss1 = torch.mean(output)
                g_loss = g_loss2 - g_loss1
                g_loss.backward()
                self.g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数


            if epoch == 0:

                a = real_signal.detach().cpu().numpy()
                signals = pd.DataFrame({
                    "bvp": a[0]
                })
                signals.plot()
                plt.savefig(Config.save_path + '/real_images.png')
                plt.close()

            fake_signal_cpu = fake_signal.cpu()
            a = fake_signal_cpu.detach().numpy()
            signals = pd.DataFrame({
                "bvp": a[0]
            })
            signals.plot()
            plt.savefig(Config.save_path + '/fake_images-{}.png'.format(epoch))
            plt.close()
            # if epoch % 10 == 0:
            print('Epoch[{}/{}],d_loss:{:.6f},g_loss1:{:.6f},,g_loss2:{:.6f} '
                  'D real: {:.6f},D fake: {:.6f},GP:{:.6f}'.format(
                epoch, Config.epoch, d_loss.data.item(), g_loss1.data.item(), g_loss2.data.item(),
                real_scores.data.mean(), fake_scores.data.mean(), gp_3  # 打印的是真实图片的损失均值
            ))
            dloss_list.append(d_loss.data.item())
            gloss_list.append(g_loss.data.item())

            # save_image(fake_signal, './signal_img/fake_images-{}.png'.format(epoch + 1))
            if ((epoch + 1) % 40 == 0):
                torch.save(self.G.state_dict(), Config.save_path + "/G_parameter" + str(epoch) + ".pkl")
                torch.save(self.D.state_dict(), Config.save_path + "/D_parameter" + str(epoch) + ".pkl")
                loss_plot = pd.DataFrame({
                    "dloss": dloss_list,
                    "gloss": gloss_list
                })
                loss_plot.plot()
                plt.savefig(Config.save_path + '/loss' + str(epoch) + '.png')


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

    np.savetxt('./results/generate/UBFC_ODELSTM_test1_3000_state_1.csv', res, delimiter=',')


if __name__ == "__main__":
    # 创建文件夹
    if not os.path.exists(Config.save_path):
        os.mkdir(Config.save_path)
    model_path = Config.model_path
    model_ids = [get_model_ids(model_path)[Config.model_id]]
    print(model_ids)
    model = load_model(*model_ids, Config.model_epoch).to(Config.device)
    # create_signal('results/wgan/UBFS/signal_img_1/ODE_test1/G_parameter799.pkl', 3000)
    p = TrainProcess()
    p.train_step()
