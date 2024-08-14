import nonLinearFeatureExtract as NonLinFea
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def real_feature_extract():
    mm = MinMaxScaler(feature_range=(-1, 1))
    type = 'bvp'
    filepath = 'data/clean_' + type + '_all'
    i = 1
    trail_len = 6  # 6s一个trail
    trail_step = 2  # 每个trail向前移动步长为2s
    trail_num = int((180 - trail_len) / trail_step + 1)  # 一条时长为180s的数据可以划分的trail数
    nonLinearFeature1 = []
    nonLinearFeature2 = []
    nonLinearFeature3 = []
    for num in glob.glob(filepath + '/s*'):
        # num1 = num.replace('eda', 'ppg')

        data_path1 = num + '/clean_' + type + '_s' + str(i) + '_T1.csv'
        data_path2 = num + '/clean_' + type + '_s' + str(i) + '_T2.csv'
        data_path3 = num + '/clean_' + type + '_s' + str(i) + '_T3.csv'
        data_all_1 = np.loadtxt(data_path1)
        data_all_2 = np.loadtxt(data_path2)
        data_all_3 = np.loadtxt(data_path3)

        # if i == 5:
        for n in range(trail_num):  # 32HZ的采样频率,6s为时间窗，一条数据时长3分钟，共截出30个trail
            m = n * 32 * trail_step  # 32*6=192 6s的数据

            data1 = data_all_1[m:m + trail_len * 32]
            data1 = data1.reshape((len(data1), 1))
            data1 = mm.fit_transform(data1)  # 标准化 归一化
            data1 = data1.reshape((len(data1),))
            data1_EN = NonLinFea.EnFeatureExtract(data1)
            data1_FD = NonLinFea.FractalDimensionExtract(data1)
            data1_nonLinear = list(data1_EN)
            data1_nonLinear.append(data1_FD)

            data2 = data_all_2[m:m + trail_len * 32]
            data2 = data2.reshape((len(data2), 1))
            data2 = mm.fit_transform(data2)  # 标准化 归一化
            data2 = data2.reshape((len(data2),))
            data2_EN = NonLinFea.EnFeatureExtract(data2)
            data2_FD = NonLinFea.FractalDimensionExtract(data2)
            data2_nonLinear = list(data2_EN)
            data2_nonLinear.append(data2_FD)

            data3 = data_all_3[m:m + trail_len * 32]
            data3 = data3.reshape((len(data3), 1))
            data3 = mm.fit_transform(data3)  # 标准化 归一化
            data3 = data3.reshape((len(data3),))
            data3_EN = NonLinFea.EnFeatureExtract(data3)
            data3_FD = NonLinFea.FractalDimensionExtract(data3)
            data3_nonLinear = list(data3_EN)
            data3_nonLinear.append(data3_FD)

            nonLinearFeature1.append(data1_nonLinear)
            nonLinearFeature2.append(data2_nonLinear)
            nonLinearFeature3.append(data3_nonLinear)
        print(i)
        i += 1
    np.array(nonLinearFeature1)
    np.array(nonLinearFeature2)
    np.array(nonLinearFeature3)
    # 存特征
    path = os.getcwd()
    savepath = path + '/nonLinear/real/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    np.savetxt(savepath + 'data_all_1(-1,1).csv', nonLinearFeature1, delimiter=',')
    np.savetxt(savepath + 'data_all_2(-1,1).csv', nonLinearFeature2, delimiter=',')
    np.savetxt(savepath + 'data_all_3(-1,1).csv', nonLinearFeature3, delimiter=',')


def fake_feature_extract():
    mm = MinMaxScaler()

    # method=['base','c-rnn-gan','cGanWithDS','dcgan','time_gan']
    method = ['ODELSTM']
    dataset = "UBFC"
    for meth in method:
        data_all_1 = np.loadtxt("results/generate/" + dataset + "_" + meth + "_test1_3000_state_1.csv", delimiter=",")
        data_all_2 = np.loadtxt("results/generate/" + dataset + "_" + meth + "_test1_3000_state_2.csv", delimiter=",")

        nonLinearFeature1 = []
        nonLinearFeature2 = []
        i = 1
        for n in range(len(data_all_2)):  # 32HZ的采样频率,6s为时间窗，一条数据时长3分钟，共截出30个trail

            data1 = data_all_1[n, :]
            data1 = data1.reshape((len(data1), 1))
            data1 = mm.fit_transform(data1)  # 标准化 归一化
            data1 = data1.reshape((len(data1),))
            data1_EN = NonLinFea.EnFeatureExtract(data1)
            data1_FD = NonLinFea.FractalDimensionExtract(data1)
            data1_nonLinear = list(data1_EN)
            data1_nonLinear.append(data1_FD)

            data2 = data_all_2[n, :]
            data2 = data2.reshape((len(data2), 1))
            data2 = mm.fit_transform(data2)  # 标准化 归一化
            data2 = data2.reshape((len(data2),))
            data2_EN = NonLinFea.EnFeatureExtract(data2)
            data2_FD = NonLinFea.FractalDimensionExtract(data2)
            data2_nonLinear = list(data2_EN)
            data2_nonLinear.append(data2_FD)

            nonLinearFeature1.append(data1_nonLinear)
            nonLinearFeature2.append(data2_nonLinear)
            print(meth +' '+ str(i))
            i += 1
        # 存特征
        np.array(nonLinearFeature1)
        np.array(nonLinearFeature2)
        # 存特征
        path = os.getcwd()
        savepath = path + '/nonLinear/fake/' + dataset + '/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        np.savetxt(savepath + 'data_all_1_' + meth + '(-1,1).csv', nonLinearFeature1, delimiter=',')
        np.savetxt(savepath + 'data_all_2_' + meth + '(-1,1).csv', nonLinearFeature2, delimiter=',')


# reaf_feature_extract()
fake_feature_extract()
