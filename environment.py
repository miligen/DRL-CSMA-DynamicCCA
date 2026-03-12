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

        self.gain_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    dist = get_distance(self.nodes[i].pos, self.nodes[j].pos)
                    self.gain_matrix[i, j] = -calculate_path_loss(dist)


    def get_valid_mask(self, node):
        mask = [False] * ACTION_DIM
        num_th = len(TH_SET)
        for i, n_id in enumerate(node.neighbors):
            start = i * num_th
            end = start + num_th
            for k in range(start, end): mask[k] = True
        return mask

    def decode_action(self, node, action_idx):
        num_th = len(TH_SET)
        th_idx = action_idx % num_th
        n_idx = action_idx // num_th

        if n_idx >= len(node.neighbors):
            n_idx = 0

        target_id = node.neighbors[n_idx]
        th_val = TH_SET[th_idx]
        return target_id, th_val

    def calculate_interference(self, rx_node, tx_nodes):
        i_watt = 0.0
        for tx in tx_nodes:
            if tx.id == rx_node.id: continue
            p_rx = dbm_to_watt(TX_POWER_DBM + self.gain_matrix[tx.id, rx_node.id])
            i_watt += p_rx
        return i_watt

    # 去除了 def reset(self)，因为时间轴是连续的，不需要定期重置

    def run_slot(self):
        # === 1. 决策阶段 ===
        for node in self.nodes:
            if node.status == 'IDLE':
                mask = self.get_valid_mask(node)
                if not any(mask): continue

                state = node.get_state_vector()
                action_idx = node.agent.select_action(state, mask)
                target, th_dbm = self.decode_action(node, action_idx)

                node.decision_state = state
                node.current_action_idx = action_idx
                node.target_id = target
                node.chosen_th_watt = dbm_to_watt(th_dbm)

                node.backoff_counter = np.random.randint(0, FIXED_CW + 1)
                node.status = 'BACKOFF'

        # === 2. 微时隙步进 ===
        active_tx_nodes = []
        for t in range(FIXED_CW + 1):
            snapshot_emitters = [n for n in self.nodes if n.status == 'TX']
            nodes_starting_tx = []

            for node in self.nodes:
                if node.status == 'BACKOFF':
                    i_watt = self.calculate_interference(node, snapshot_emitters)
                    node.sense_history.append(quantize_rssi(i_watt))

                    is_busy = i_watt >= node.chosen_th_watt
                    if not is_busy:
                        node.backoff_counter -= 1

                    if node.backoff_counter < 0:
                        nodes_starting_tx.append(node)

            for node in nodes_starting_tx:
                node.status = 'TX'
                active_tx_nodes.append(node)

        # === 3. 上帝视角结算与学习 ===
        tx_nodes = [n for n in self.nodes if n.status == 'TX']
        total_success = 0
        total_step_reward = 0.0
        num_updated_nodes = 0

        for tx in tx_nodes:
            rx = self.nodes[tx.target_id]

            sig = dbm_to_watt(TX_POWER_DBM + self.gain_matrix[tx.id, rx.id])
            intf = self.calculate_interference(rx, tx_nodes)
            intf = max(0.0, intf - sig)

            is_success = False
            sinr = -100.0  # 初始化为一个极小值用于打印

            # 接收端半双工约束
            if rx.status != 'TX':
                sinr = calculate_sinr(sig, intf)
                if sinr >= SINR_THRESHOLD_DB:
                    is_success = True

            # === 4. 奖励与统计更新 ===
            if is_success:
                reward = REWARD_SUCCESS
                total_success += 1  # 【关键修复】加上这一行，吞吐量才会统计！
            else:
                reward = REWARD_FAIL

            total_step_reward += reward
            num_updated_nodes += 1

            # ==========================================
            # 【新增：核心日志打印】
            # 取消注释下一行，可以观察底层物理参数的博弈过程
            # print(f"[物理层日志] Node {tx.id} -> Node {rx.id} | 动作 CCA_Th: {watt_to_dbm(tx.chosen_th_watt):.1f}dBm | 接收端干扰: {watt_to_dbm(intf):.1f}dBm | 接收端 SINR: {sinr:.2f}dB | 结果: {'成功(+1)' if is_success else '失败(-1)'}")
            # ==========================================

            next_state = tx.get_state_vector()
            tx.agent.memory.push(tx.decision_state, tx.current_action_idx,
                                 reward, next_state, False)
            tx.agent.update()

            tx.status = 'IDLE'

        avg_reward = total_step_reward / num_updated_nodes if num_updated_nodes > 0 else 0.0
        return total_success, avg_reward