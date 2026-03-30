import torch
import numpy as np

# ===========================
# 1. 基础设置
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ===========================
# 2. 网络拓扑与物理模型 (距离布尔模型)
# ===========================
AREA_SIZE = 200.0
LAMBDA_U = 0.004
COMMUNICATION_RANGE = 30.0   # 基础通信距离 (Rc)

# 【核心重构】基于距离的感知范围 (Rs) 设定
# 最小感知距离 (极度激进，忽视近处邻居，用于克服暴露终端)
MIN_SENSE_RANGE = 0.125 * COMMUNICATION_RANGE  # 20.0m
# 最大感知距离 (极度保守，听得极远，用于发现隐藏终端)
MAX_SENSE_RANGE = 2.2 * COMMUNICATION_RANGE  # 88.0m

# 动作空间：Agent 现在的任务是从这些距离半径中选一个作为自己的 CCA 侦听圈
RS_SET = list(np.linspace(MIN_SENSE_RANGE, MAX_SENSE_RANGE, 15))

# ===========================
# 3. 时隙结构
# ===========================
MICRO_SLOT_TIME = 20e-6
DATA_TIME = 1e-3

FIXED_CW = 15
T_MAX_SENSE = FIXED_CW * MICRO_SLOT_TIME
SLOT_DURATION = T_MAX_SENSE + DATA_TIME

# ===========================
# 4. 状态与动作维度
# ===========================
K_SENSE_HISTORY = 10
MAX_NEIGHBORS = 10

STATE_DIM = K_SENSE_HISTORY + MAX_NEIGHBORS
ACTION_DIM = MAX_NEIGHBORS * len(RS_SET)

# ===========================
# 5. 训练与统计参数
# ===========================
TOTAL_SLOTS = 100000
STATS_INTERVAL = 20

LR = 3e-4
GAMMA_RL = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 5000

BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 100

REWARD_SUCCESS = 1.0
REWARD_FAIL = -1.0

EMA_ALPHA = 0.8