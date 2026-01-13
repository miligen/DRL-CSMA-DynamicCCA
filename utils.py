# utils.py
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
    if dist <= 1.0: return 0.0
    return 10 * PATH_LOSS_EXPONENT * np.log10(dist)


def calculate_sinr(signal_watt, interference_watt):
    noise_watt = dbm_to_watt(NOISE_FLOOR_DBM)
    # 【修复】强制干扰为非负
    interference_watt = max(0.0, interference_watt)

    sinr_linear = signal_watt / (interference_watt + noise_watt)

    # 【修复】防止对极小值取对数
    if sinr_linear <= 1e-20:
        return -100.0

    sinr_db = 10 * np.log10(sinr_linear)
    return sinr_db


def quantize_rssi(rssi_watt):
    """归一化 RSSI 到 0-1"""
    rssi_dbm = watt_to_dbm(rssi_watt)
    norm = (rssi_dbm - NOISE_FLOOR_DBM) / 50.0
    return np.clip(norm, 0.0, 1.0)

if __name__ == '__main__':
    print(calculate_path_loss(get_distance((0,0), (0,50))))