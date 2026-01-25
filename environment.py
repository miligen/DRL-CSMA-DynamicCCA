# environment.py

import numpy as np
from config import *
from utils import *
from node import Node


class AdHocEnv:
    def __init__(self):
        self.nodes = [Node(i, (i * DISTANCE, 0)) for i in range(NUM_NODES)]
        for node in self.nodes:
            node.init_neighbors(self.nodes)

        # 预计算增益矩阵
        self.gain_matrix = np.zeros((NUM_NODES, NUM_NODES))
        for i in range(NUM_NODES):
            for j in range(NUM_NODES):
                if i != j:
                    dist = get_distance(self.nodes[i].pos, self.nodes[j].pos)
                    self.gain_matrix[i, j] = -calculate_path_loss(dist)

    def get_valid_mask(self, node):
        # 有数据发送的节点对应动作掩码置为True
        mask = [False] * ACTION_DIM
        num_w = len(CW_SET)
        num_th = len(TH_SET)
        block_size = num_w * num_th

        for i, n_id in enumerate(node.neighbors):
            if node.local_queues[n_id] > 0:
                start = i * block_size
                end = start + block_size
                for k in range(start, end): mask[k] = True
        return mask

    def decode_action(self, node, action_idx):
        # TODO: action_idx所有节点数量一致，但是对于不同节点是否可以用不同的state数量，使得后续更处理更简单一些（比如掩码这些）？
        num_w = len(CW_SET)
        num_th = len(TH_SET)
        tmp = action_idx
        th_idx = tmp % num_th
        tmp //= num_th
        w_idx = tmp % num_w
        tmp //= num_w
        n_idx = tmp

        target_id = node.neighbors[n_idx]
        w_val = CW_SET[w_idx]
        th_val = TH_SET[th_idx]
        return target_id, w_val, th_val

    def calculate_interference(self, rx_node, tx_nodes):
        i_watt = 0.0
        for tx in tx_nodes:
            if tx.id == rx_node.id: continue
            p_rx = dbm_to_watt(TX_POWER_DBM + self.gain_matrix[tx.id, rx_node.id])
            i_watt += p_rx
        return i_watt

    def reset(self):
        """重置环境中的所有节点到初始状态 (用于新 Frame 开始)"""
        for node in self.nodes:
            node.reset_for_new_frame()

    def run_slot(self, current_frame=0):
        # === 1. 数据包生成 ===
        for node in self.nodes: node.generate_traffic()

        # === 2. 决策阶段 (仅 IDLE 节点)
        for node in self.nodes:
            if node.status == 'IDLE':
                mask = self.get_valid_mask(node)
                if not any(mask): continue

                # 获取状态
                state = node.get_state_vector()
                # agent选择动作
                action_idx = node.agent.select_action(state, mask)
                target, w_max, th_dbm = self.decode_action(node, action_idx)

                if target is None:
                    continue

                # 更新节点状态
                node.decision_state = state
                node.current_action_idx = action_idx
                node.target_id = target
                node.chosen_th_watt = dbm_to_watt(th_dbm)

                node.backoff_counter = np.random.randint(0, w_max + 1) # TODO: 和agent直接输出计数器有什么区别？直接输出是否会更稳定？
                node.reset_stats_for_new_action()
                node.status = 'BACKOFF'

        # === 3. 微时隙步进 (CSMA/CA 过程) ===
        active_tx_nodes = []

        for t in range(max(CW_SET) + 1):
            # A. 【快照】本时刻的干扰源
            snapshot_emitters = [n for n in self.nodes if n.status == 'TX']

            # 用于记录本微时隙即将倒数归零的节点
            nodes_starting_tx = []

            # B. 【决策】每个节点基于侦听到的干扰强度判断是否减小计数器
            for node in self.nodes:
                if node.status == 'BACKOFF':
                    i_watt = self.calculate_interference(node, snapshot_emitters)

                    node.obs_accum_interference += i_watt # TODO: 干扰是否要累计，还是最值？（参考论文）
                    node.obs_total_slots += 1

                    is_busy = i_watt >= node.chosen_th_watt

                    if is_busy:
                        node.obs_busy_slots += 1
                        # 冻结计数器
                    else:
                        node.backoff_counter -= 1

                    # 倒数结束
                    if node.backoff_counter < 0:
                        nodes_starting_tx.append(node)

            # C. 【状态同步】mini-slot结束，统一转变发送状态
            for node in nodes_starting_tx:
                node.status = 'TX'
                node.tx_start_time = t
                node.ack_received = False
                active_tx_nodes.append(node)

        # === 4. 数据传输与 ACK 统计===
        tx_nodes = [n for n in self.nodes if n.status == 'TX']
        successful_pairs = []  # 记录本时隙 Data 成功的 (Tx, Rx) 对

        # --- 4.1 Data接收判定 ---
        for tx in tx_nodes:
            rx = self.nodes[tx.target_id]

            # Data SINR
            sig = dbm_to_watt(TX_POWER_DBM + self.gain_matrix[tx.id, rx.id])
            intf = self.calculate_interference(rx, tx_nodes)
            intf = max(0.0, intf - sig)  # 数值保护

            if rx.status != 'TX':  # 半双工约束
                sinr = calculate_sinr(sig, intf)
                if sinr >= SINR_THRESHOLD_DB:
                    successful_pairs.append((tx, rx))

        # --- 4.2 ACK发送与接收 ---
        if successful_pairs:
            # 这里的 successful_pairs 包含了所有 Data 阶段成功的 (tx, rx)

            # A. Tx 接收 ACK (判定是否发送成功)
            # 逻辑：只有在同一微时隙开始发送 (tx_start_time相同) 的节点，其对应的接收方才会同时回 ACK，从而产生干扰。
            for tx, rx in successful_pairs:
                # 找出当前时刻同时在发 ACK 的干扰源，条件：是成功的接收者 且 对应发送方的开始时间与当前 tx 相同 且 不是 rx 自己
                concurrent_ack_intf_senders = [
                    pair[1] for pair in successful_pairs
                    if pair[0].tx_start_time == tx.tx_start_time and pair[1].id != rx.id
                ]

                ack_sig = dbm_to_watt(TX_POWER_DBM + self.gain_matrix[rx.id, tx.id])
                ack_intf = max(0.0, self.calculate_interference(tx, concurrent_ack_intf_senders))

                if calculate_sinr(ack_sig, ack_intf) >= SINR_THRESHOLD_DB:
                    tx.ack_received = True

            # B. 其他节点侦听 ACK (避让奖励来源)
            for node in self.nodes:
                # 只有处于 BACKOFF (正在避让) 的节点才统计 Overhearing
                if node.status == 'BACKOFF':
                    # 遍历每一个成功的 ACK 事件
                    for tx, rx in successful_pairs:
                        concurrent_ack_intf_senders = [
                            pair[1] for pair in successful_pairs
                            if pair[0].tx_start_time == tx.tx_start_time and pair[1].id != rx.id
                        ]

                        ack_sig = dbm_to_watt(TX_POWER_DBM + self.gain_matrix[rx.id, node.id])
                        ack_intf = max(0.0, self.calculate_interference(node, concurrent_ack_intf_senders))

                        if calculate_sinr(ack_sig, ack_intf) >= SINR_THRESHOLD_DB:
                            node.obs_overheard_acks += 1

        # === 5. 奖励计算与状态更新===
        total_success = 0
        total_step_reward = 0.0  # 记录本时隙的总奖励
        real_success = len(successful_pairs)

        for node in self.nodes:
            if node.status == 'TX':
                # 只有完成 TX 的节点才结算
                p_coll = 0
                i_avg = 0
                if node.obs_total_slots > 0:
                    p_coll = node.obs_busy_slots / node.obs_total_slots
                    i_avg = quantize_rssi(node.obs_accum_interference / node.obs_total_slots)

                is_aggressive = (node.chosen_th_watt >= dbm_to_watt(TH_HIGH))

                # --- 发送奖励 (R_outcome) ---
                if node.ack_received:
                    r_outcome = ALPHA
                    if is_aggressive: r_outcome += LAMBDA_RS * i_avg

                    # 更新状态
                    node.history_status[node.target_id] = 1
                    node.local_queues[node.target_id] = max(0, node.local_queues[node.target_id] - 1)
                    total_success += 1
                else:
                    r_outcome = -ALPHA
                    if is_aggressive: r_outcome -= DELTA_PENALTY
                    node.history_status[node.target_id] = -1

                # --- 计算信道质量奖励 (R_channel) ---
                # 奖励寻找空闲窗口：P_coll 越低，奖励越高
                r_channel = BETA * (1.0 - p_coll)

                # --- 计算协作侦听奖励 ---
                # 奖励在退避期间听到了邻居的成功
                r_coop = GAMMA_COOP * node.obs_overheard_acks

                # 总奖励
                reward = r_outcome + r_channel + r_coop

                # 累加奖励
                total_step_reward += reward

                next_state = node.get_state_vector()

                # agent训练
                node.agent.memory.push(node.decision_state, node.current_action_idx,
                                       reward, next_state, False)
                node.agent.update()

                node.status = 'IDLE'

        return total_success, total_step_reward, real_success