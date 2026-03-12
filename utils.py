import numpy as np
from config import *

def get_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def dbm_to_watt(dbm):
    return 10 ** (dbm / 10.0) / 1000.0

def watt_to_dbm(watt):
    if watt <= 1e-20: return -100.0
    return 10.0 * np.log10(watt * 1000.0)

def calculate_path_loss(dist):
    if dist <= 1.0:
        return REFERENCE_LOSS
    # 【修复4】加上参考损耗，这才是真实的无线电传播模型
    return REFERENCE_LOSS + 10 * PATH_LOSS_EXPONENT * np.log10(dist)

def calculate_sinr(signal_watt, interference_watt):
    noise_watt = dbm_to_watt(NOISE_FLOOR_DBM)
    interference_watt = max(0.0, interference_watt)
    sinr_linear = signal_watt / (interference_watt + noise_watt)
    if sinr_linear <= 1e-20: return -100.0
    return 10 * np.log10(sinr_linear)

def quantize_rssi(rssi_watt):
    rssi_dbm = watt_to_dbm(rssi_watt)
    norm = (rssi_dbm - NOISE_FLOOR_DBM) / 50.0
    return np.clip(norm, 0.0, 1.0)