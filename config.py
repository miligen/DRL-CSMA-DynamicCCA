# config.py
import torch

# ===========================
# 1. 基础设置
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# ===========================
# 2. 网络拓扑
# ===========================
NUM_NODES = 4
DISTANCE = 50.0
INTERFERENCE_RANGE = 120.0

# ===========================
# 3. 物理层 & 时隙结构
# ===========================
# 时间单位: 秒
MICRO_SLOT_TIME = 20e-6   # 20us
DATA_TIME = 1e-3          # 1ms
ACK_TIME = 100e-6         # 100us
GUARD_TIME = 10e-6

# 动作空间参数
# 竞争窗口集合 (CW_min ... CW_max)
CW_SET = [4, 8, 16, 32]   # 对应动作中的 W
# CCA 阈值集合 (dBm) fixme: CCA阈值设置
TH_LOW = -85.0    # 敏感 (Hidden Terminal)
TH_MID = -75.0    # 默认
TH_HIGH = -65.0   # 激进 (Exposed Terminal)
TH_SET = [TH_LOW, TH_MID, TH_HIGH]

# 最大的物理侦听时间限制
W_MAX_PHYSICAL = max(CW_SET)
T_MAX_SENSE = W_MAX_PHYSICAL * MICRO_SLOT_TIME

SLOT_DURATION = T_MAX_SENSE + DATA_TIME + GUARD_TIME + ACK_TIME

# 信号参数
TX_POWER_DBM = 20.0
NOISE_FLOOR_DBM = -90.0 # 底噪
SINR_THRESHOLD_DB = 10.0
PATH_LOSS_EXPONENT = 3.5

# ===========================
# 4. 流量与训练
# ===========================
LAMBDA_POISSON = 0.2
MAX_QUEUE_SIZE = 50
NUM_FRAMES = 500
SLOTS_PER_FRAME = 20
BATCH_SIZE = 64
LR = 5e-4
GAMMA_RL = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 5000
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

# ===========================
# 5. 奖励参数 (Risk-Sensitive)
# ===========================
ALPHA = 1.0       # 发送基础分
LAMBDA_RS = 2.0   # 暴露终端激励
DELTA_PENALTY = 2.0 # 鲁莽惩罚
BETA = 0.5        # 信道质量激励 (1 - P_coll) 的系数
GAMMA_COOP = 0.5  # 每侦听到一个 ACK 的奖励

# 状态维度: Queue(3) + History(3) + I_obs(1) + P_coll_obs(1) = 8
MAX_NEIGHBORS = 3
STATE_DIM = MAX_NEIGHBORS + MAX_NEIGHBORS + 2
# 动作维度: Target(3) * W(4) * Th(3) = 36
ACTION_DIM = MAX_NEIGHBORS * len(CW_SET) * len(TH_SET)