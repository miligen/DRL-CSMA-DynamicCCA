# node.py
import numpy as np
from config import *
from models import DQNAgent


class Node:
    def __init__(self, node_id, pos):
        self.id = node_id
        self.pos = pos
        self.neighbors = []

        # 每个节点拥有自己独立的 Agent
        self.agent = DQNAgent(node_id)

        # 队列与历史
        self.local_queues = {}
        self.history_status = {}

        # 状态机变量
        self.status = 'IDLE'  # IDLE, BACKOFF, TX

        # 退避相关
        self.backoff_counter = 0
        self.chosen_th_watt = 0.0
        self.current_action_idx = 0
        self.decision_state = None
        self.target_id = None

        # 统计观测
        self.obs_accum_interference = 0.0
        self.obs_busy_slots = 0
        self.obs_total_slots = 0
        self.obs_overheard_acks = 0 # 统计侦听到的 ACK 数量

        # 结果标记
        self.tx_start_time = -1
        self.ack_received = False

    def init_neighbors(self, all_nodes):
        from utils import get_distance
        for other in all_nodes:
            if other.id != self.id:
                dist = get_distance(self.pos, other.pos)
                if dist <= DISTANCE * 1.5:
                    self.neighbors.append(other.id)
                    self.local_queues[other.id] = 0
                    self.history_status[other.id] = 0

    def generate_traffic(self):
        for n_id in self.neighbors:
            new_pkts = np.random.poisson(LAMBDA_POISSON)
            self.local_queues[n_id] = min(self.local_queues[n_id] + new_pkts, MAX_QUEUE_SIZE)
            self.local_queues[n_id] = 50 # fixme:先不考虑流量生成进行排查已有逻辑

    def get_state_vector(self):
        # 1. 队列 (归一化)
        q_vec = [self.local_queues.get(n, 0) / MAX_QUEUE_SIZE for n in self.neighbors]
        q_vec += [0] * (MAX_NEIGHBORS - len(q_vec))

        # 2. 历史状态 (+1, 0, -1) 无数据、或者初始时、或不存在邻居状态为0
        h_vec = [self.history_status.get(n, 0) for n in self.neighbors]
        h_vec += [0] * (MAX_NEIGHBORS - len(h_vec))

        # 3. 退避过程中的平均干扰强度 4.退避过程中有干扰的时隙占比（冲突概率）
        if self.obs_total_slots > 0:
            avg_i = self.obs_accum_interference / self.obs_total_slots
            p_coll = self.obs_busy_slots / self.obs_total_slots
        else:
            avg_i = 0.0
            p_coll = 0.0

        from utils import quantize_rssi
        stat_vec = [quantize_rssi(avg_i), p_coll]

        return np.array(q_vec + h_vec + stat_vec, dtype=np.float32)

    def reset_stats_for_new_action(self):
        self.obs_accum_interference = 0.0
        self.obs_busy_slots = 0
        self.obs_total_slots = 0
        self.obs_overheard_acks = 0