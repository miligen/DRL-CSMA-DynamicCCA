import numpy as np
from config import *
from utils import *
from node import Node


class AdHocEnv:
    def __init__(self):
        expected_nodes = LAMBDA_U * AREA_SIZE * AREA_SIZE
        self.num_nodes = np.random.poisson(expected_nodes)
        self.num_nodes = max(self.num_nodes, 2)

        xs = np.random.uniform(0, AREA_SIZE, self.num_nodes)
        ys = np.random.uniform(0, AREA_SIZE, self.num_nodes)

        self.nodes = [Node(i, (xs[i], ys[i])) for i in range(self.num_nodes)]
        for node in self.nodes:
            node.init_neighbors(self.nodes)

    def get_valid_mask(self, node):
        mask = [False] * ACTION_DIM
        num_rs = len(RS_SET)
        for i, n_id in enumerate(node.neighbors):
            start = i * num_rs
            end = start + num_rs
            for k in range(start, end): mask[k] = True
        return mask

    def decode_action(self, node, action_idx):
        num_rs = len(RS_SET)
        rs_idx = action_idx % num_rs
        n_idx = action_idx // num_rs

        if n_idx >= len(node.neighbors):
            n_idx = 0

        target_id = node.neighbors[n_idx]
        rs_val = RS_SET[rs_idx]
        return target_id, rs_val, n_idx

    def get_min_tx_distance(self, target_node, tx_nodes):
        """【核心重构】计算距离目标节点最近的发送源距离"""
        min_dist = float('inf')
        for tx in tx_nodes:
            if tx.id == target_node.id: continue
            d = get_distance(tx.pos, target_node.pos)
            if d < min_dist:
                min_dist = d
        return min_dist

    def run_slot(self):
        # === 1. 决策阶段 ===
        for node in self.nodes:
            if node.status == 'IDLE':
                mask = self.get_valid_mask(node)
                if not any(mask): continue

                state = node.get_state_vector()
                action_idx = node.agent.select_action(state, mask)
                target, rs_val, n_idx = self.decode_action(node, action_idx)

                node.decision_state = state
                node.current_action_idx = action_idx
                node.target_id = target
                node.target_n_idx = n_idx
                node.chosen_rs = rs_val  # 设定感知半径

                node.backoff_counter = np.random.randint(0, FIXED_CW + 1)
                node.status = 'BACKOFF'

        # === 2. 微时隙步进 (布尔侦听) ===
        active_tx_nodes = []
        for t in range(FIXED_CW + 1):
            snapshot_emitters = [n for n in self.nodes if n.status == 'TX']
            nodes_starting_tx = []

            for node in self.nodes:
                if node.status == 'BACKOFF':
                    d_min = self.get_min_tx_distance(node, snapshot_emitters)
                    # 将距离转化为神经网络认识的 [0,1] 干扰强度
                    node.sense_history.append(normalize_interference_dist(d_min))

                    # 【布尔判定】如果最近的发送者在我的感知半径内，信道忙碌
                    is_busy = d_min < node.chosen_rs
                    if not is_busy:
                        node.backoff_counter -= 1

                    if node.backoff_counter < 0:
                        nodes_starting_tx.append(node)

            for node in nodes_starting_tx:
                node.status = 'TX'
                active_tx_nodes.append(node)

        # === 3. 上帝视角结算与学习 (布尔干涉模型) ===
        tx_nodes = [n for n in self.nodes if n.status == 'TX']
        total_success = 0
        total_step_reward = 0.0
        num_updated_nodes = 0

        for tx in tx_nodes:
            rx = self.nodes[tx.target_id]

            # ==========================================
            # 【修复 BUG】剔除合法的发送者，只计算"其他"发送源带来的干扰
            # ==========================================
            interfering_tx_nodes = [n for n in tx_nodes if n.id != tx.id]
            rx_d_min = self.get_min_tx_distance(rx, interfering_tx_nodes)

            # 如果接收方没在发数据，且其他干扰源都在通信距离 Rc 之外，则接收成功！
            is_success = (rx.status != 'TX') and (rx_d_min > COMMUNICATION_RANGE)

            # ==========================================
            # 【等效重构】基于距离的暴露终端奖励/惩罚机制
            # ==========================================
            tx_d_min = self.get_min_tx_distance(tx, tx_nodes)  # 这里不用改，因为函数内部排除了自身

            # 如果距离最近的活跃节点在 2*Rc 以内，说明 tx 顶着干扰强行发送了
            k_aggressiveness = 0.0
            if tx_d_min < 2.0 * COMMUNICATION_RANGE:
                # 距离越近，激进指数越大 (最大为 1.0)
                clamped_d = max(tx_d_min, MIN_SENSE_RANGE)
                k_aggressiveness = (2.0 * COMMUNICATION_RANGE - clamped_d) / (
                            2.0 * COMMUNICATION_RANGE - MIN_SENSE_RANGE)

            ALPHA_BONUS = 1.0
            BETA_PENALTY = 1.0

            if is_success:
                reward = REWARD_SUCCESS + ALPHA_BONUS * k_aggressiveness
                total_success += 1
            else:
                reward = REWARD_FAIL - BETA_PENALTY * k_aggressiveness
            # ==========================================

            total_step_reward += reward
            num_updated_nodes += 1

            tx.update_link_quality(tx.target_n_idx, is_success)

            next_state = tx.get_state_vector()
            tx.agent.memory.push(tx.decision_state, tx.current_action_idx,
                                 reward, next_state, False)
            tx.agent.update()

            tx.status = 'IDLE'

        # avg_reward = total_step_reward / num_updated_nodes if num_updated_nodes > 0 else 0.0 # 把计算平均值的工作统一交给 main.py 即可
        return total_success, total_step_reward, num_updated_nodes