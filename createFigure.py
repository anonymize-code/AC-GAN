import os
import torch as tc
from sklearn import decomposition
import phase_space_reconstruction as psr
import numpy as np
import utils
from glob import glob
from bptt import models
import matplotlib.pyplot as plt
import matplotlib


def load_model(model_id, epoch):
    model = models.Model()
    model.init_from_model_path(model_id, epoch=epoch)
    model.eval()
    return model


#
#
def is_model_id(path):
    """Check if path ends with a three digit, e.g. save/test/001 """
    run_nr = path.split('\\')[-1]
    three_digit_numbers = {str(digit).zfill(3) for digit in set(range(1000))}
    return run_nr in three_digit_numbers


#
#
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


matplotlib.rc('text', usetex=True)

data_path = 'datasets/Lorenz/lorenz_data_chaos.npy'


data = tc.tensor(utils.read_data(data_path))


model_path = 'results/GradientTests/Lorenz'

# 模型选择
model_ids = [get_model_ids(model_path)[4]]
model = load_model(*model_ids, 300)

N = 10000
# 生成信号点的个数
ts, z = model.generate_free_trajectory(data, 15000)
ts = ts.detach().cpu()
z = z.detach().cpu()
data = data.cpu()
# ts=ts[::3]
#



# 信号图形
# N = 300
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.plot(data[:N, 0], linewidth=1)
# ax.plot(ts[:N, 0], linewidth=1)
# plt.show()

# # 信号吸引子图形
# N = 500
# model = decomposition.PCA(n_components=3)
# z_3 = model.fit_transform(z.detach().numpy())
# np.savetxt("results/BVPTests/bvp_ode/extract_attractor_27.csv",z_3,delimiter=',')
# psr.print_plot(z_3, 0, 0, N)
# # psr.print_plot(data_1, 0, 0, N)
# plt.show()

# 洛伦兹图形
N = 5000
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)
ax.plot(data[:N, 0], data[:N, 1], data[:N, 2], linewidth=0.5)
plt.savefig(model_ids[0] + "\lorenz0.jpg", dpi=600, format="jpg", bbox_inches='tight', pad_inches=0.1)
plt.show()
