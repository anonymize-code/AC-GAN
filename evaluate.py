import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from minepy import MINE

mm = MinMaxScaler(feature_range=(-1, 1))


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算


def eva_mmd(data1, data2, data3):
    mmd_sum_1, mmd_sum_2 = 0, 0
    len = 2000
    size = 40
    epoch = int(len / size)
    for i in range(epoch):
        x = torch.tensor(data1[i * size:(i + 1) * size, :])
        y = torch.tensor(data2[i * size:(i + 1) * size, :])
        z = torch.tensor(data3[i * size:(i + 1) * size, :])
        mmd_1 = mmd_rbf(x, y)
        mmd_2 = mmd_rbf(x, z)
        # print(mmd_1,mmd_2)
        mmd_sum_1 += mmd_1
        mmd_sum_2 += mmd_2

    print("ave_mmd:", mmd_sum_1 / epoch)
    print("ave_mmd:", mmd_sum_2 / epoch)
    return mmd_sum_2 / epoch


def eva_mse(data1, data2, data3):
    mse_sum_1, mse_sum_2 = 0, 0
    len = 2000
    size = 1
    epoch = int(len / size)
    for i in range(epoch):
        x = torch.tensor(data1[i * size:(i + 1) * size, :])
        y = torch.tensor(data2[i * size:(i + 1) * size, :])
        z = torch.tensor(data3[i * size:(i + 1) * size, :])
        mse_1 = F.mse_loss(x, y)
        mse_2 = F.mse_loss(x, z)
        # print(mmd_1,mmd_2)
        mse_sum_1 += mse_1
        mse_sum_2 += mse_2

    print("ave_mse:", mse_sum_1 / epoch)
    print("ave_mse:", mse_sum_2 / epoch)


def eva_dtw(data1, data2, data3):
    dtw_sum_1, dtw_sum_2 = 0, 0
    len = 3000
    size = 1
    epoch = int(len / size)
    for i in range(epoch):
        x = torch.tensor(data1[i * size:(i + 1) * size, :])
        y = torch.tensor(data2[i * size:(i + 1) * size, :])
        z = torch.tensor(data3[i * size:(i + 1) * size, :])
        dtw_1 = fastdtw(x, y, dist=euclidean)[0]
        dtw_2 = fastdtw(x, z, dist=euclidean)[0]
        # print(mmd_1,mmd_2)
        dtw_sum_1 += dtw_1
        dtw_sum_2 += dtw_2

    print("ave_dtw:", dtw_sum_1 / epoch)
    print("ave_dtw:", dtw_sum_2 / epoch)


def eva_pc(data1, data2, data3):
    pc_sum_1, pc_sum_2 = 0, 0
    len = 3000
    size = 1
    epoch = int(len / size)
    for i in range(epoch):
        x = torch.tensor(data1[i * size:(i + 1) * size, :])
        y = torch.tensor(data2[i * size:(i + 1) * size, :])
        z = torch.tensor(data3[i * size:(i + 1) * size, :])
        pc_1 = abs(np.corrcoef(x, y))[0, 1]
        pc_2 = abs(np.corrcoef(x, z))[0, 1]
        # print(mmd_1,mmd_2)
        pc_sum_1 += pc_1
        pc_sum_2 += pc_2

    print("ave_pc:", pc_sum_1 / epoch)
    print("ave_pc:", pc_sum_2 / epoch)


def eva_mic(data1, data2, data3):
    mic_sum_1, mic_sum_2 = 0, 0
    len = 2000
    size = 1
    epoch = int(len / size)
    mine = MINE(alpha=0.8, c=15)
    for i in range(epoch):
        x = data1[i * size:(i + 1) * size, :]
        y = data2[i * size:(i + 1) * size, :]
        z = data3[i * size:(i + 1) * size, :]
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        mine.compute_score(x, y)
        mic_1 = mine.mic()
        mine.compute_score(x, z)
        mic_2 = mine.mic()
        # print(mmd_1,mmd_2)
        mic_sum_1 += mic_1
        mic_sum_2 += mic_2
        print(i)
        # print("mic_1:", mic_1)
        # print("mic_2:", mic_2)

    print("ave_mic:", mic_sum_1 / epoch)
    print("ave_mic:", mic_sum_2 / epoch)


def find_mmd(data1, data2):
    mmd_sum_1, mmd_sum_2 = 0, 0
    len = 2000
    size = 40
    epoch = int(len / size)
    for i in range(epoch):
        x = torch.tensor(data1[i * size:(i + 1) * size, :])
        y = torch.tensor(data2[i * size:(i + 1) * size, :])
        mmd_1 = mmd_rbf(x, y)
        mmd_sum_1 += mmd_1

    print("ave_mmd:", mmd_sum_1 / epoch)
    return mmd_sum_1 / epoch


data1 = np.loadtxt('datasets/bvp_data/UBFC_bvp_train_1_mm.csv', delimiter=',')[:2000, :192]  # 7
data2 = np.loadtxt('results/generate/UBFC_WGAN_test1_3000_state_1.csv', delimiter=',')[:2000, :]
data3 = np.loadtxt('results/generate/UBFC_ODELSTM_test1_3000_state_1.csv', delimiter=',')[: 2000, :]  #

eva_mic(data1, data2, data3)
eva_mmd(data1, data2, data3)
eva_mse(data1, data2, data3)
eva_dtw(data1, data2, data3)
