import torch
import numpy as np

# ===========================
# 1. 基础设置
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ===========================
# 2. 网络拓扑与泊松点过程 (PPP)
# ===========================
AREA_SIZE = 100.0
LAMBDA_U = 0.002
COMMUNICATION_RANGE = 40.0

# ===========================
# 3. 物理层 & 时隙结构
# ===========================
MICRO_SLOT_TIME = 20e-6
DATA_TIME = 1e-3

FIXED_CW = 15
TH_SET = list(np.linspace(-82.0, -10.0, 15))

T_MAX_SENSE = FIXED_CW * MICRO_SLOT_TIME
SLOT_DURATION = T_MAX_SENSE + DATA_TIME

TX_POWER_DBM = 15.0
NOISE_FLOOR_DBM = -90.0
SINR_THRESHOLD_DB = 3.0
PATH_LOSS_EXPONENT = 3.0
REFERENCE_LOSS = 40.0

# ===========================
# 4. 状态与动作维度
# ===========================
K_SENSE_HISTORY = 10
MAX_NEIGHBORS = 5

# 【核心修改】状态维度扩维：干扰历史 (K) + 目标链路质量矩阵 (MAX_NEIGHBORS)
STATE_DIM = K_SENSE_HISTORY + MAX_NEIGHBORS
ACTION_DIM = MAX_NEIGHBORS * len(TH_SET)

# ===========================
# 5. 训练与统计参数
# ===========================
TOTAL_SLOTS = 100000
STATS_INTERVAL = 20       # 每 20 个时隙统计一次利用率和奖励

LR = 3e-4
GAMMA_RL = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 5000      # 衰减步数，基于 Agent 做出决策的次数

BATCH_SIZE = 32
MEMORY_SIZE = 10000       # 扩大容量，防止灾难性遗忘
TARGET_UPDATE = 100

REWARD_SUCCESS = 1.0
REWARD_FAIL = -1.0

# 【新增】链路质量的 EMA 遗忘因子 (0.8表示侧重历史，0.2响应当前)
EMA_ALPHA = 0.8