import numpy as np
from config import *

def get_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def normalize_interference_dist(d_min):
    """
    【核心重构】将最近干扰源的距离，映射为神经网络的 [0, 1] 状态输入。
    距离越近 (d_min趋于0) -> 干扰越大，映射值越接近 1.0
    距离越远 (d_min > MAX_SENSE_RANGE) -> 无干扰，映射值为 0.0
    这样完美平替了之前 quantize_rssi 的功能，无需修改神经网络结构。
    """
    if d_min >= MAX_SENSE_RANGE:
        return 0.0
    return np.clip(1.0 - (d_min / MAX_SENSE_RANGE), 0.0, 1.0)