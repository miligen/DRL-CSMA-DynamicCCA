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
        self.chosen_rs = 0.0 # 【重构】由 chosen_th_watt 改为 chosen_rs (选择的感知半径)
        self.current_action_idx = 0
        self.decision_state = None
        self.target_id = None
        self.target_n_idx = 0

        # 【新增】利他主义侦听计数器：记录从决策到发送期间，侦听到的邻居成功 ACK 数量
        self.overheard_acks = 0

        self.sense_history = deque([0.0] * K_SENSE_HISTORY, maxlen=K_SENSE_HISTORY)
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
        state_sense = np.array(self.sense_history, dtype=np.float32)
        state_link = np.array(self.link_quality, dtype=np.float32)
        return np.concatenate((state_sense, state_link))

    def update_link_quality(self, n_idx, is_success):
        if n_idx >= len(self.link_quality):
            return
        reward_val = 1.0 if is_success else 0.0
        self.link_quality[n_idx] = EMA_ALPHA * self.link_quality[n_idx] + (1.0 - EMA_ALPHA) * reward_val

    def reset_for_new_frame(self):
        self.status = 'IDLE'
        self.backoff_counter = 0
        self.current_action_idx = 0
        self.decision_state = None
        self.target_id = None
        self.target_n_idx = 0
        self.chosen_rs = 0.0
        self.overheard_acks = 0  # 【新增】状态重置时清零
        self.sense_history.extend([0.0] * K_SENSE_HISTORY)