import numpy as np
from collections import deque
from config import *
from models import DQNAgent


class Node:
    def __init__(self, node_id, pos):
        self.id = node_id
        self.pos = pos
        self.neighbors = []
        self.agent = DQNAgent(node_id)

        self.status = 'IDLE'

        self.backoff_counter = 0
        self.chosen_th_watt = 0.0
        self.current_action_idx = 0
        self.decision_state = None
        self.target_id = None

        # 【新增】记录当前动作选中的邻居索引，用于结算时更新特定的链路
        self.target_n_idx = 0

        self.sense_history = deque([0.0] * K_SENSE_HISTORY, maxlen=K_SENSE_HISTORY)

        # 【核心新增】目标链路质量矩阵 H_i，初始化为 1.0 (乐观初始，鼓励探索)
        self.link_quality = [1.0] * MAX_NEIGHBORS

    def init_neighbors(self, all_nodes):
        from utils import get_distance
        distances = []
        for other in all_nodes:
            if other.id != self.id:
                dist = get_distance(self.pos, other.pos)
                if dist <= COMMUNICATION_RANGE:
                    distances.append((dist, other.id))

        distances.sort()
        self.neighbors = [n_id for d, n_id in distances[:MAX_NEIGHBORS]]

    def get_state_vector(self):
        # 【核心修改】将干扰历史和链路质量矩阵拼接为完整状态
        state_sense = np.array(self.sense_history, dtype=np.float32)
        state_link = np.array(self.link_quality, dtype=np.float32)
        return np.concatenate((state_sense, state_link))

    def update_link_quality(self, n_idx, is_success):
        """【新增】使用指数移动平均 (EMA) 更新目标链路的历史成功率"""
        if n_idx >= len(self.link_quality):
            return
        # 成功为1.0，失败为0.0
        reward_val = 1.0 if is_success else 0.0
        self.link_quality[n_idx] = EMA_ALPHA * self.link_quality[n_idx] + (1.0 - EMA_ALPHA) * reward_val

    def reset_for_new_frame(self):
        self.status = 'IDLE'
        self.backoff_counter = 0
        self.current_action_idx = 0
        self.decision_state = None
        self.target_id = None
        self.target_n_idx = 0
        self.chosen_th_watt = 0.0
        self.sense_history.extend([0.0] * K_SENSE_HISTORY)
        # 注意：不重置 link_quality，保留长期的拓扑记忆