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
AREA_SIZE = 100.0          # 区域大小 100m x 100m
LAMBDA_U = 0.002           # 用户密度 (预计 20 个节点)
COMMUNICATION_RANGE = 40.0 # 最大物理通信距离

# ===========================
# 3. 物理层 & 时隙结构
# ===========================
MICRO_SLOT_TIME = 20e-6   # 20us
DATA_TIME = 1e-3          # 1ms

FIXED_CW = 15

# 【修改1】扩大 CCA 阈值范围：-82 到 -10，均匀切分为 30 个值
TH_SET = list(np.linspace(-82.0, -10.0, 15))

T_MAX_SENSE = FIXED_CW * MICRO_SLOT_TIME
SLOT_DURATION = T_MAX_SENSE + DATA_TIME

TX_POWER_DBM = 15.0
NOISE_FLOOR_DBM = -90.0
SINR_THRESHOLD_DB = 3.0
# 【修复2】修改路径损耗指数为典型的城市场景 3.0
PATH_LOSS_EXPONENT = 3.0
# 【修复3】增加 1米处的参考损耗 (Reference Loss)
REFERENCE_LOSS = 40.0

# ===========================
# 4. 状态与动作维度
# ===========================
K_SENSE_HISTORY = 10
MAX_NEIGHBORS = 5

STATE_DIM = K_SENSE_HISTORY
ACTION_DIM = MAX_NEIGHBORS * len(TH_SET)

# ===========================
# 5. 训练与统计参数
# ===========================
# 【修改3】不再使用 NUM_FRAMES，改为总时隙数和统计间隔
TOTAL_SLOTS = 20000       # 总运行时隙数 (相当于原来的 3000帧 * 20)
STATS_INTERVAL = 20       # 每 20 个时隙统计一次利用率和奖励

LR = 3e-4
GAMMA_RL = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 5000      # 衰减步数，基于 Agent 做出决策的次数

BATCH_SIZE = 32
MEMORY_SIZE = int(TOTAL_SLOTS / 6)
TARGET_UPDATE = 100       # 每 100 个统计间隔 (即2000个时隙) 更新一次目标网络

REWARD_SUCCESS = 1.0
REWARD_FAIL = -1.0