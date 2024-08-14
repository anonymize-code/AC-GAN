import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
mm = MinMaxScaler(feature_range=(-1, 1))

# CLAS 256 WESAD 1137 UBFC 177
bvp_data=np.loadtxt('datasets/bvp_data/UBFC_bvp_train_1_mm.csv',delimiter=',')[:,:192]
data_list=[]
for i in range(bvp_data.shape[0]):
    data_list.append(bvp_data[i].tolist())
data_np=np.reshape(data_list, -1)
data_np=np.reshape(data_np, (len(data_np),1))
np.save('datasets/bvp_data/CLAS_bvp_all_3_train_L6.npy', data_np)