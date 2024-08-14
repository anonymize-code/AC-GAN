import EntropyHub as EH
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import hfd


def EnFeatureExtract(data):
    ApEn = EH.ApEn(data)[0][-1]
    SampEn = EH.SampEn(data)[0][-1]
    FuzzEn = EH.FuzzEn(data)[0][-1]
    PermEn = EH.PermEn(data)[0][-1]
    K2En = EH.K2En(data)[0][-1]
    return ApEn, SampEn, FuzzEn, PermEn, K2En


def PoincarePlotFeaturnExtract(data):
    info = nk.ppg_findpeaks(data, 32)
    peaks = nk.signal_formatpeaks(info, len(data))
    hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=32, show=True)
    return hrv_nonlinear["HRV_SD1"][0], hrv_nonlinear["HRV_SD2"][0], hrv_nonlinear["HRV_SD1SD2"][0]


def FractalDimensionExtract(data):
    return hfd.hfd(data)
