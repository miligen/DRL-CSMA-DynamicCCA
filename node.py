import numpy as np
from collections import deque
from config import *
from models import DQNAgent


class Node:
    def __init__(self, node_id, pos):
        self.id = node_id
        self.pos = pos
        self.neighbors = []  # 实际可选的邻居 (最多 MAX_NEIGHBORS 个)
        self.agent = DQNAgent(node_id)

        self.status = 'IDLE'

        self.backoff_counter = 0
        self.chosen_th_watt = 0.0
        self.current_action_idx = 0
        self.decision_state = None
        self.target_id = None

        self.sense_history = deque([0.0] * K_SENSE_HISTORY, maxlen=K_SENSE_HISTORY)

    def init_neighbors(self, all_nodes):
        from utils import get_distance
        distances = []
        for other in all_nodes:
            if other.id != self.id:
                dist = get_distance(self.pos, other.pos)
                if dist <= COMMUNICATION_RANGE:
                    distances.append((dist, other.id))

        # 按距离排序，选取最近的 MAX_NEIGHBORS 个作为可选通信目标
        distances.sort()
        self.neighbors = [n_id for d, n_id in distances[:MAX_NEIGHBORS]]

    def get_state_vector(self):
        return np.array(self.sense_history, dtype=np.float32)

    def reset_for_new_frame(self):
        self.status = 'IDLE'
        self.backoff_counter = 0
        self.current_action_idx = 0
        self.decision_state = None
        self.target_id = None
        self.chosen_th_watt = 0.0
        self.sense_history.extend([0.0] * K_SENSE_HISTORY)