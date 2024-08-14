import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn import decomposition, datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import os
import matplotlib.style as mplstyle

mplstyle.use('fast')

mm = MinMaxScaler(feature_range=(0, 1))
filepath = 'clean_eda_all'
i = 1
dim = 3
tau = 7
result = np.empty((0, 3))


def print_plot(result1, print_method, start, end, type=1, result2=np.empty((0, 3)), i=0):
    xs_1 = result1[:, 0]
    ys_1 = result1[:, 1]
    zs_1 = result1[:, 2]
    xs_2 = result2[:, 0]
    ys_2 = result2[:, 1]
    zs_2 = result2[:, 2]
    xs_i_1 = []
    ys_i_1 = []
    zs_i_1 = []
    xs_i_2 = []
    ys_i_2 = []
    zs_i_2 = []
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.title("TEST " + str(i))
    color2 = 'c'
    if type == 1:
        color = 'r'
    elif type == 2:
        color = 'g'
    else:
        color = 'b'
    if print_method == 0:
        # ax.scatter(xs_1, ys_1, zs_1, c=color, marker='o', s=2)
        # ax.scatter(xs_2, ys_2, zs_2, c=color2, marker='o', s=2)
        ax.plot(xs_1, ys_1, zs_1, c=color,linewidth=0.5)
        ax.plot(xs_2, ys_2, zs_2, c=color2,linewidth=0.5)
    elif print_method == 1:
        for k in range(start, end):
            plt.title("TEST " + str(i))
            ax.cla()
            xs_i_1.append(xs_1[k])
            ys_i_1.append(ys_1[k])
            zs_i_1.append(zs_1[k])
            ax.scatter(xs_i_1, ys_i_1, zs_i_1, c=color, marker='o', s=10)
            if result2.size != 0:
                xs_i_2.append(xs_2[k])
                ys_i_2.append(ys_2[k])
                zs_i_2.append(zs_2[k])
                ax.scatter(xs_i_2, ys_i_2, zs_i_2, c=color2, marker='o', s=10)
            plt.pause(0.01)

    else:
        for k in range(start, end):
            xs_i_1.append(xs_1[k])
            ys_i_1.append(ys_1[k])
            zs_i_1.append(zs_1[k])
            if result2.size != 0:
                xs_i_2.append(xs_2[k])
                ys_i_2.append(ys_2[k])
                zs_i_2.append(zs_2[k])
        ax.scatter(xs_i_1, ys_i_1, zs_i_1, c=color, marker='o', s=5)
        ax.scatter(xs_i_2, ys_i_2, zs_i_2, c=color2, marker='o', s=10)
    # plt.pause(1)
    # plt.close()


def print_plot_2D(result1, i, print_method, start, end, type, result2=np.empty((0, 3))):
    xs_1 = result1[:, 0]
    ys_1 = result1[:, 1]
    xs_2 = result2[:, 0]
    ys_2 = result2[:, 1]
    xs_i_1 = []
    ys_i_1 = []
    xs_i_2 = []
    ys_i_2 = []
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    plt.title("TEST " + str(i))
    color2 = 'c'
    if type == 1:
        color = 'r'
    elif type == 2:
        color = 'g'
    else:
        color = 'b'
    if print_method == 0:
        ax.scatter(xs_1, ys_1, c=color, marker='o', s=2)
        ax.scatter(xs_2, ys_2, c=color2, marker='o', s=2)
    elif print_method == 1:
        for k in range(start, end):
            plt.title("TEST " + str(i))
            ax.cla()
            xs_i_1.append(xs_1[k])
            ys_i_1.append(ys_1[k])
            ax.scatter(xs_i_1, ys_i_1, c=color, marker='o', s=10)
            if result2.size != 0:
                xs_i_2.append(xs_2[k])
                ys_i_2.append(ys_2[k])
                ax.scatter(xs_i_2, ys_i_2, c=color2, marker='o', s=10)
            plt.pause(0.01)

    else:
        for k in range(start, end):
            xs_i_1.append(xs_1[k])
            ys_i_1.append(ys_1[k])
            if result2.size != 0:
                xs_i_2.append(xs_2[k])
                ys_i_2.append(ys_2[k])
        ax.scatter(xs_i_1, ys_i_1, c=color, marker='o', s=10)
        ax.scatter(xs_i_2, ys_i_2, c=color2, marker='o', s=10)
    # plt.pause(1)
    # plt.close()


def print_PSR():
    i = 1
    dim = 7
    tau = 8
    type = 1
    # filepath = 'data/clean_bvp_all'
    filepath = 'data/cut_bvp_wesad'
    print_method = 2  # 0直接输出，1逐个点输出，2输出部分和总体
    for num in glob.glob(filepath + '/s*'):
        # bvp_path = num + '/clean_bvp_s' + str(i) + '_T' + str(type) + '.csv'
        bvp_path = num + '/cut_bvp_stress.csv'

        # bvp_path = num + '/clean_eda_s' + str(i) + '_T' + str(type) + '.csv'
        # bvp_path = num +'/cut_bvp_base.csv'
        ppg = np.loadtxt(bvp_path)
        bvp_data = ppg.reshape((len(ppg), 1))
        bvp_data = mm.fit_transform(bvp_data)  # 标准化 归一化
        bvp_data = bvp_data.reshape((len(bvp_data),))
        ps_vector = np.zeros(((len(ppg) - (dim - 1) * tau), dim))
        for j in range(len(ppg) - (dim - 1) * tau):
            for k in range(dim):
                ps_vector[j][k] = bvp_data[j + k * tau]
        result = ps_vector

        model = decomposition.PCA(n_components=3)
        model.fit(result)
        result = model.fit_transform(result)
        result = mm.fit_transform(result)

        # model=TSNE(n_components=3)
        # result=model.fit_transform(result)

        # model=decomposition.KernelPCA(n_components=3,kernel='rbf')
        # result = model.fit_transform(result)

        if (type == 1):
            # if i == 15:
            start = 3000
            end = 3500
            print_plot(result, i, print_method, start, end, type)
        # print_plot(result, i, 0, start, end, type)
        elif (type == 2):
            start = 1000
            end = 1192
            print_plot(result, i, print_method, start, end, type)
        elif (type == 3):
            start = 1000
            end = 1192
            print_plot(result, i, print_method, start, end, type)
        plt.show(block=False)
        plt.close()
        i += 1


def print_PSR_all(type):
    i = 1
    dim = 7
    tau = 8
    if type == 'ppg':
        signal = 'bvp'
    else:
        signal = 'eda'
    filepath = 'clean_' + type + '_all'
    length = 5760 - (dim - 1) * tau
    result = np.empty((0, dim))
    result1, result2, result3 = np.empty((0, dim)), np.empty((0, dim)), np.empty((0, dim))
    for num in glob.glob(filepath + '/s*'):
        ps_vector1, ps_vector2, ps_vector3 = np.zeros((length, dim)), np.zeros((length, dim)), np.zeros((length, dim))  # N-(dim-1)*tau
        bvp_path_1 = num + '/clean_' + signal + '_s' + str(i) + '_T1.csv'
        bvp_path_2 = num + '/clean_' + signal + '_s' + str(i) + '_T2.csv'
        bvp_path_3 = num + '/clean_' + signal + '_s' + str(i) + '_T3.csv'
        ppg1 = np.loadtxt(bvp_path_1)
        ppg2 = np.loadtxt(bvp_path_2)
        ppg3 = np.loadtxt(bvp_path_3)

        # result1, result2, result3 = np.empty((0, dim)), np.empty((0, dim)), np.empty((0, dim))
        bvp_data_1 = ppg1
        bvp_data_1 = bvp_data_1.reshape((len(bvp_data_1), 1))
        bvp_data_1 = mm.fit_transform(bvp_data_1)  # 标准化 归一化
        bvp_data_1 = bvp_data_1.reshape((len(bvp_data_1),))

        bvp_data_2 = ppg2
        bvp_data_2 = bvp_data_2.reshape((len(bvp_data_2), 1))
        bvp_data_2 = mm.fit_transform(bvp_data_2)  # 标准化 归一化
        bvp_data_2 = bvp_data_2.reshape((len(bvp_data_2),))

        bvp_data_3 = ppg3
        bvp_data_3 = bvp_data_3.reshape((len(bvp_data_3), 1))
        bvp_data_3 = mm.fit_transform(bvp_data_3)  # 标准化 归一化
        bvp_data_3 = bvp_data_3.reshape((len(bvp_data_3),))

        for j in range(5760 - (dim - 1) * tau):
            for k in range(dim):
                ps_vector1[j][k] = bvp_data_1[j + k * tau]
                ps_vector2[j][k] = bvp_data_2[j + k * tau]
                ps_vector3[j][k] = bvp_data_3[j + k * tau]

        result1 = ps_vector1
        result2 = ps_vector2
        result3 = ps_vector3

        model = decomposition.PCA(n_components=3)
        model.fit(result1)
        result1 = model.transform(result1)
        result1 = mm.fit_transform(result1)

        model.fit(result2)
        result2 = model.transform(result2)
        result2 = mm.fit_transform(result2)

        model.fit(result3)
        result3 = model.transform(result3)
        result3 = mm.fit_transform(result3)

        fig = plt.figure(figsize=(16, 8))

        xs = result1[:, 0]
        ys = result1[:, 1]
        zs = result1[:, 2]
        ax = fig.add_subplot(131, projection='3d')
        plt.title("TEST " + str(i))
        ax.scatter(xs, ys, zs, c='r', marker='o', s=2)
        # for i in range(len(xs)):
        #     ax.scatter(xs[i], ys[i], zs[i], c='r', marker='o')
        #     plt.pause(0.01)

        xs = result2[:, 0]
        ys = result2[:, 1]
        zs = result2[:, 2]
        ax = fig.add_subplot(132, projection='3d')
        ax.scatter(xs, ys, zs, c='g', marker='o', s=2)
        # for i in range(len(xs)):
        #     ax.scatter(xs[i], ys[i], zs[i], c='g', marker='o')
        #     plt.pause(0.01)

        xs = result3[:, 0]
        ys = result3[:, 1]
        zs = result3[:, 2]
        ax = fig.add_subplot(133, projection='3d')
        ax.scatter(xs, ys, zs, c='b', marker='o', s=2)
        # for i in range(len(xs)):
        #     ax.scatter(xs[i], ys[i], zs[i], c='b', marker='o')
        #     plt.pause(0.01)

        plt.tight_layout()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.45)
        plt.show(block=False)
        # plt.pause(1)
        plt.close()
        i += 1


def print_twoSingal_PSR():
    i = 1
    dim = 7
    tau = 8
    filepath = 'clean_bvp_all'

    type = 1
    print_method = 2  # 0直接输出，1逐个点输出，2输出部分和总体

    for num in glob.glob(filepath + '/s*'):
        num1 = num.replace('bvp', 'eda')
        bvp_path = num + '/clean_bvp_s' + str(i) + '_T' + str(type) + '.csv'
        eda_path = num1 + '/clean_eda_s' + str(i) + '_T' + str(type) + '.csv'
        ppg = np.loadtxt(bvp_path)
        eda = np.loadtxt(eda_path)
        i += 1
        bvp_data = ppg.reshape((len(ppg), 1))
        bvp_data = mm.fit_transform(bvp_data)  # 标准化 归一化
        bvp_data = bvp_data.reshape((len(bvp_data),))
        eda_data = eda.reshape((len(eda), 1))
        eda_data = mm.fit_transform(eda_data)  # 标准化 归一化
        eda_data = eda_data.reshape((len(eda_data),))

        bvp_ps_vector = np.zeros(((5760 - (dim - 1) * tau), dim))
        eda_ps_vector = np.zeros(((5760 - (dim - 1) * tau), dim))
        fusion_ps_vector = np.zeros(((5760 - (dim - 1) * tau), dim * 2))
        for j in range(5760 - (dim - 1) * tau):
            for k in range(dim):
                bvp_ps_vector[j][k] = bvp_data[j + k * tau]
                eda_ps_vector[j][k] = eda_data[j + k * tau]
                fusion_ps_vector[j][k] = bvp_data[j + k * tau]
                fusion_ps_vector[j][k + dim] = eda_data[j + k * tau]
        bvp_result = bvp_ps_vector
        eda_result = eda_ps_vector
        model = decomposition.PCA(n_components=3)
        model.fit(bvp_result)
        model.fit(eda_result)
        model.fit(fusion_ps_vector)
        bvp_result = model.fit_transform(bvp_result)
        bvp_result = mm.fit_transform(bvp_result)

        eda_result = model.fit_transform(eda_result)
        eda_result = mm.fit_transform(eda_result)

        fusion_result = model.fit_transform(fusion_ps_vector)
        fusion_result = mm.fit_transform(fusion_result)

        result = np.r_[eda_result, bvp_result]
        if (type == 1):
            start = 1000
            end = 1200
            # print_plot(bvp_result, i, print_method, start, end, type, result2=eda_result)
            print_plot(eda_result, i, print_method, start, end, type)
        elif (type == 2):
            start = 3000
            end = 4100
            # print_plot(bvp_result, i, print_method, start, end, type, result2=eda_result)
            print_plot(eda_result, i, print_method, start, end, type)
            # print_plot(fusion_result, i, print_method, start, end, type,result2=bvp_result)
        elif (type == 3):
            start = 3000
            end = 3200
            # print_plot(bvp_result, i, 0, start, end, type)
            print_plot(fusion_result, i, print_method, start, end, type, result2=eda_result)
        plt.show(block=False)
        plt.close()

# print_PSR()
# print_PSR_all('eda')
# print_twoSingal_PSR()
