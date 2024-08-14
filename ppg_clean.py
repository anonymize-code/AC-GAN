import neurokit2 as nk
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

def save_clean_ppg():
    filepath = 'E:\\data\\56\\34-A Multimodal Database For Psychophysiological Studies Of Social Stress'
    i = 1
    for num in glob.glob(filepath + '/s*'):
        for num_next in glob.glob(num + '/s*'):
            bvp_path1 = num_next + '/bvp_s' + str(i) + '_T1.csv'
            bvp_path2 = num_next + '/bvp_s' + str(i) + '_T2.csv'
            bvp_path3 = num_next + '/bvp_s' + str(i) + '_T3.csv'
            ppg1 = np.loadtxt(bvp_path1)
            ppg_downsample1 = nk.signal_resample(ppg1, method='interpolation', sampling_rate=64,
                                                 desired_sampling_rate=32)
            ppg2 = np.loadtxt(bvp_path2)
            ppg_downsample2 = nk.signal_resample(ppg2, method='interpolation', sampling_rate=64,
                                                 desired_sampling_rate=32)
            ppg3 = np.loadtxt(bvp_path3)
            ppg_downsample3 = nk.signal_resample(ppg3, method='interpolation', sampling_rate=64,
                                                 desired_sampling_rate=32)
            ppg1_clean = nk.ppg_clean(ppg_downsample1, method='elgendi', sampling_rate=32)
            ppg2_clean = nk.ppg_clean(ppg_downsample2, method='elgendi', sampling_rate=32)
            ppg3_clean = nk.ppg_clean(ppg_downsample3, method='elgendi', sampling_rate=32)
            path = os.getcwd() + '/clean_ppg_all/s' + str(i)
            if not os.path.exists(path):
                os.makedirs(path)
            np.savetxt(path + '/clean_bvp_s' + str(i) + '_T1.csv', ppg1_clean, delimiter=',')
            np.savetxt(path + '/clean_bvp_s' + str(i) + '_T2.csv', ppg2_clean, delimiter=',')
            np.savetxt(path + '/clean_bvp_s' + str(i) + '_T3.csv', ppg3_clean, delimiter=',')
        i += 1


save_clean_ppg()

