import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, friedmanchisquare
from scipy.stats import norm
from math import ceil

plt.rc('font', family='Times New Roman')
color_list = ['purple','crimson', 'chocolate', 'orange', 'blue','lightseagreen', 'green','purple','cyan']


def cd_diagram(data, labels, cd):
    # apply Friedman test
    friedman_pvalue = friedmanchisquare(*data)[1]

    # compute ranks
    ranks = rankdata(-data.T, axis=1)

    # compute average ranks
    ranks_mean = np.mean(ranks, axis=0)

    # calculate critical difference

    n_datasets = len(data[0])
    n_methods = len(data)
    q_alpha = 2.85  # from the Nemenyi table
    cd_val = q_alpha * np.sqrt((n_methods * (n_methods + 1)) / (6 * n_datasets))

    # 绘图
    plt.figure(figsize=(7, 6))
    # plt.plot([0, 0], [2, 5], linestyle='-', color='black', linewidth=1)
    plt.plot([0, 0], [0, 10], linestyle='-', color='black', linewidth=1)

    # plt.show()
    for i in range(n_methods):
        plt.plot([0, 0], [ranks_mean[i], ranks_mean[i]], marker='o',
                 c=color_list[len(color_list) - i - 1])
        plt.plot([0, i + 1], [ranks_mean[i], ranks_mean[i]], linestyle='dotted', linewidth=1,
                 c=color_list[len(color_list) - i - 1])
        plt.plot([i + 1, i + 1], [ranks_mean[i] - cd_val / 2, ranks_mean[i] + cd_val / 2], linewidth=5, c=color_list[len(color_list) - i - 1])
        plt.plot([0, i + 1], [ranks_mean[i] - cd_val / 2, ranks_mean[i] - cd_val / 2], linestyle='dotted', linewidth=1, c=color_list[len(color_list) - i - 1])
        plt.plot([0, i + 1], [ranks_mean[i] + cd_val / 2, ranks_mean[i] + cd_val / 2], linestyle='dotted', linewidth=1, c=color_list[len(color_list) - i - 1])
    plt.text(0 + 1, ranks_mean[0] + 0.4, "%s:%.2f" % (labels[0], round(ranks_mean[0], 2)), ha='center', va='center',
             fontsize=12.5)
    plt.text(1 + 1, ranks_mean[1] + 0.4, "%s:%.2f" % (labels[1], round(ranks_mean[1], 2)), ha='center', va='center',
             fontsize=12.5)
    plt.text(2 + 1, ranks_mean[2] + 0.4, "%s:%.2f" % (labels[2], round(ranks_mean[2], 2)), ha='center', va='center',
             fontsize=12.5)
    plt.text(3 + 1, ranks_mean[3] + 0.4, "%s:%.2f" % (labels[3], round(ranks_mean[3], 2)), ha='center', va='center',
             fontsize=12.5)
    plt.text(4 + 1, ranks_mean[4] + 0.45, "%s:%.2f" % (labels[4], round(ranks_mean[4], 2)), ha='center', va='center',
             fontsize=12.5)
    plt.text(5 + 1, ranks_mean[5] + 0.4, "%s:%.2f" % (labels[5], round(ranks_mean[5], 2)), ha='center', va='center',
             fontsize=12.5)

    # plt.text(6 + 1, ranks_mean[6] + 0.55, "%s:%.2f" % (labels[6], round(ranks_mean[6], 2)), ha='center', va='center',
    #          fontsize=12.5)
    # plt.text(7 + 1, ranks_mean[7] + 0.65, "%s:%.2f" % (labels[7], round(ranks_mean[7], 2)), ha='center', va='center',
    #          fontsize=12.5)
    # plt.text(8 + 1, ranks_mean[8] + 0.55, "%s:%.2f" % (labels[8], round(ranks_mean[8], 2)), ha='center', va='center',
    #          fontsize=12.5)

    # plt.text(0 + 1, ranks_mean[0] + 1.3, "%s:%.2f" % (labels[0], round(ranks_mean[0], 2)), ha='center', va='center',
    #          fontsize=12.5)
    # plt.text(1 + 1, ranks_mean[1] + 1.3, "%s:%.2f" % (labels[1], round(ranks_mean[1], 2)), ha='center', va='center',
    #          fontsize=12.5)
    # plt.text(2 + 1, ranks_mean[2] + 1.3, "%s:%.2f" % (labels[2], round(ranks_mean[2], 2)), ha='center', va='center',
    #          fontsize=12.5)
    # plt.text(3 + 1, ranks_mean[3] + 1.3, "%s:%.2f" % (labels[3], round(ranks_mean[3], 2)), ha='center', va='center',
    #          fontsize=12.5)
    # plt.text(4 + 1, ranks_mean[4] + 1.4, "%s:%.2f" % (labels[4], round(ranks_mean[4], 2)), ha='center', va='center',
    #          fontsize=12.5)
    # plt.text(5 + 1, ranks_mean[5] + 1.5, "%s:%.2f" % (labels[5], round(ranks_mean[5], 2)), ha='center', va='center',
    #          fontsize=12.5)

    # plt.text(6 + 1, ranks_mean[6] + 1.3, "%s:%.2f" % (labels[6], round(ranks_mean[6], 2)), ha='center', va='center',
    #          fontsize=12.5)
    # plt.text(7 + 1, ranks_mean[7] + 1.3, "%s:%.2f" % (labels[7], round(ranks_mean[7], 2)), ha='center', va='center',
    #          fontsize=12.5)
    # plt.text(8 + 1, ranks_mean[8] + 1.3, "%s:%.2f" % (labels[8], round(ranks_mean[8], 2)), ha='center', va='center',
    #          fontsize=12.5)

    plt.xlim([-1, ceil(n_methods + 1)])
    plt.ylim([2, 5])
    # plt.ylim([0, 8])
    plt.xlabel('Method', fontsize=15)
    plt.ylabel('Average Rank', fontsize=15)
    plt.title('Nemenyi test: p-value=0.05, CD=%.4f' % (cd_val), fontsize=15)
    plt.tight_layout()
    plt.savefig('result/Acc/acc3.png', dpi=500)
    # plt.savefig('result/correlation/cor3.png', dpi=500)
    # plt.show()


# set the data
#
# result1 = [[76.3, 63, 63.5], [73.2, 47.4, 54.1], [73.5, 47.4, 54.1], [63.2, 54.7, 61.3], [72.1, 47.4, 54.1], [69.7, 54.9, 62.5]]  # acc
# result2 = [[76.3, 62.7, 59.2], [71.1, 43, 52.9], [71.5, 43, 50.4], [55.6, 56.2, 56.4], [68.3, 43, 52.9], [66.3, 53.6, 60.4]]  # F1
# result3 = [[76.3, 63, 63.5, 76.3, 62.7, 59.2], [73.2, 47.4, 54.1, 71.1, 43, 52.9], [73.5, 47.4, 54.1, 71.5, 43, 50.4], [63.2, 54.7, 61.3, 55.6, 56.2, 56.4], [72.1, 47.4, 54.1, 68.3, 43, 52.9],
#            [69.7, 54.9, 62.5, 66.3, 53.6, 60.4]]
def acc_plot():
    methods = ["ODELSTM", "mymethod","base","WGAN2023","cGanWithDS", "time_gan", "c-rnn-gan", "LogGAN","dcgan"]
    selected_indices = [1, 2, 4, 5, 6, 8]  # 你想要选择的索引列表
    methods = [methods[i] for i in selected_indices]  # 使用列表推导式来选择数据
    result_acc = [[] for _ in range(len(selected_indices))]
    i = 0
    for method in methods:
        result_acc[i].append(np.loadtxt("result/Acc/" + method + "_acc_1.csv", delimiter=',')[0])
        i += 1
    result_acc = np.array(result_acc).reshape(len(selected_indices), -1)
    data = np.array(result_acc)
    # labels = ["mymethod", "cGAN+DS", "WGAN", "Time-GAN", "C-RNN-GAN", "DCGAN"]
    labels = ["CAC-GAN", "AC-GAN","WGAN","PPG-WGAN" ,"cGanWithDS", "TimeGAN", "C-RNN-GAN", "LSM-GAN","DCGAN"]
    labels = [labels[i] for i in selected_indices]
    cd = 0.5
    cd_diagram(data, labels, cd)


def cor_plot():
    result_cor = np.genfromtxt("result/correlation/cor1.csv", delimiter=',')
    data = np.array(result_cor).T
    data=data[[1,2,4,5,6,8]]
    labels = ["CAC-GAN", "AC-GAN","WGAN","PPG-WGAN" ,"cGanWithDS", "TimeGAN", "C-RNN-GAN", "LSM-GAN","DCGAN"]
    selected_indices = [1, 2, 4, 5, 6, 8]  # 你想要选择的索引列表
    labels = [labels[i] for i in selected_indices]  # 使用列表推导式来选择数据
    cd = 0.5
    cd_diagram(data, labels, cd)


# cor_plot()
acc_plot()
