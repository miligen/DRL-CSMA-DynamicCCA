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
                node.chosen_rs = rs_val

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
                    node.sense_history.append(normalize_interference_dist(d_min))

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

        # ==========================================
        # 步骤 3.1: 提前评估所有当前 TX 节点的成功状态
        # ==========================================
        success_status = {}
        successful_rx_ids = set()

        for tx in tx_nodes:
            rx = self.nodes[tx.target_id]
            interfering_tx_nodes = [n for n in tx_nodes if n.id != tx.id]
            rx_d_min = self.get_min_tx_distance(rx, interfering_tx_nodes)

            is_success = (rx.status != 'TX') and (rx_d_min > COMMUNICATION_RANGE)
            success_status[tx.id] = is_success

            if is_success:
                successful_rx_ids.add(rx.id)

        # ==========================================
        # 步骤 3.2: 模拟侦听 ACK (累加利他避让奖励)
        # ==========================================
        # 对于所有正处于 BACKOFF 或本回合刚 TX 完毕的节点：
        # 如果在它们"活跃"期间，有邻居接收成功了（且不是自己的目标），说明它的策略成功保护了邻居。
        for node in self.nodes:
            if node.status in ['BACKOFF', 'TX']:
                for rx_id in successful_rx_ids:
                    # 如果成功的接收者是我的邻居，且不是我当前正在发的目标
                    if rx_id in node.neighbors and rx_id != node.target_id:
                        node.overheard_acks += 1

        # ==========================================
        # 步骤 3.3: 发放综合奖励与 DQN 学习更新
        # ==========================================
        for tx in tx_nodes:
            is_success = success_status[tx.id]

            if is_success:
                total_success += 1

            # --- A. 计算基础奖惩与激进发送激励 ---
            tx_d_min = self.get_min_tx_distance(tx, tx_nodes)
            k_aggressiveness = 0.0
            if tx_d_min < 2.0 * COMMUNICATION_RANGE:
                clamped_d = max(tx_d_min, MIN_SENSE_RANGE)
                k_aggressiveness = (2.0 * COMMUNICATION_RANGE - clamped_d) / (
                            2.0 * COMMUNICATION_RANGE - MIN_SENSE_RANGE)

            ALPHA_BONUS = 1.0
            BETA_PENALTY = 1.0

            # --- B. 定义利他主义保护奖励系数 ---
            ALTRUISTIC_BONUS = 0.5  # 每次侦听到邻居 ACK 获得的额外奖励

            if is_success:
                reward = REWARD_SUCCESS + ALPHA_BONUS * k_aggressiveness
            else:
                reward = REWARD_FAIL - BETA_PENALTY * k_aggressiveness

            # --- C. 结算这段漫长等待期内累积的侦听 ACK 奖励 ---
            reward += ALTRUISTIC_BONUS * tx.overheard_acks

            total_step_reward += reward
            num_updated_nodes += 1

            # 状态与经验池更新
            tx.update_link_quality(tx.target_n_idx, is_success)
            next_state = tx.get_state_vector()
            tx.agent.memory.push(tx.decision_state, tx.current_action_idx,
                                 reward, next_state, False)
            tx.agent.update()

            # 彻底完成当前回合，重置节点状态与利他计数器
            tx.status = 'IDLE'
            tx.overheard_acks = 0

        return total_success, total_step_reward, num_updated_nodes