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
        num_w = len(CW_SET)
        num_th = len(TH_SET)
        tmp = action_idx
        th_idx = tmp % num_th
        tmp //= num_th
        w_idx = tmp % num_w
        tmp //= num_w
        n_idx = tmp

        # 【修复】增加安全检查
        # 如果计算出的 n_idx 超出了实际邻居范围 (极少见，但在 mask 全 False 回退时可能发生逻辑边界问题)
        if n_idx >= len(node.neighbors):
            # 强制回退到第一个邻居，避免返回 None
            if len(node.neighbors) > 0:
                n_idx = 0
            else:
                return None, 0, 0  # 孤立节点，无邻居

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

    def run_slot(self):
        # === 1. 数据包生成 ===
        for node in self.nodes: node.generate_traffic()

        # === 2. 决策阶段 (仅 IDLE 节点) ===
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

                node.backoff_counter = np.random.randint(0, w_max + 1)
                node.reset_stats_for_new_action()
                node.status = 'BACKOFF'

        # === 3. 微时隙步进 (CSMA 过程 - 修复版) ===

        # 3.1: 初始化：本时隙开始时，已经在 TX 的节点
        # 注意：跨时隙系统中，上个时隙就在发的节点，这个时隙还在发
        # 我们假设 DATA_TIME = 1ms, SLOT_DURATION > 1ms，
        # 但为简化，我们在 slot 层面认为一次 TX 持续整个 slot 的后半段。
        # 这里 active_tx_nodes 仅用于记录本时隙内 *新触发* 或 *持续* 的发射
        active_tx_nodes = []

        for t in range(max(CW_SET) + 1):
            # A. 【快照】本时刻的干扰源
            # snapshot_emitters 包含：之前就已经在 TX 的 + 本微时隙刚转为 TX 的(下一轮生效)
            snapshot_emitters = [n for n in self.nodes if n.status == 'TX']

            # 用于记录本微时隙即将倒数归零的节点
            nodes_starting_tx = []

            # B. 【决策】每个节点基于快照做判断
            for node in self.nodes:
                if node.status == 'BACKOFF':
                    i_watt = self.calculate_interference(node, snapshot_emitters)

                    node.obs_accum_interference += i_watt
                    node.obs_total_slots += 1

                    is_busy = i_watt >= node.chosen_th_watt

                    if is_busy:
                        node.obs_busy_slots += 1
                        # 冻结
                    else:
                        node.backoff_counter -= 1

                    # 倒数结束
                    if node.backoff_counter < 0:
                        nodes_starting_tx.append(node)

            # C. 【状态同步】统一转变状态，模拟并行性
            for node in nodes_starting_tx:
                node.status = 'TX'
                node.tx_start_time = t
                node.ack_received = False
                active_tx_nodes.append(node)

        # === 4. 数据传输与 ACK 结算 ===
        tx_nodes = [n for n in self.nodes if n.status == 'TX']

        for tx in tx_nodes:
            rx = self.nodes[tx.target_id]

            # Data SINR
            sig = dbm_to_watt(TX_POWER_DBM + self.gain_matrix[tx.id, rx.id])
            intf = self.calculate_interference(rx, tx_nodes)
            intf = max(0.0, intf - sig)  # 数值保护

            success = False
            if rx.status != 'TX':
                sinr = calculate_sinr(sig, intf)
                if sinr >= SINR_THRESHOLD_DB:
                    success = True

            # ACK 阶段 fixme: 记录确认接收时间，根据接收时间是否冲突判断ack是否成功
            if success:
                successful_receivers = [self.nodes[n.target_id] for n in tx_nodes if n.target_id != rx.id]
                ack_sig = dbm_to_watt(TX_POWER_DBM + self.gain_matrix[rx.id, tx.id])
                ack_intf = self.calculate_interference(tx, successful_receivers)
                ack_intf = max(0.0, ack_intf - ack_sig)  # 数值保护

                if calculate_sinr(ack_sig, ack_intf) >= SINR_THRESHOLD_DB:
                    tx.ack_received = True

        # === 5. 奖励计算与状态更新 ===
        total_success = 0

        for node in self.nodes:
            if node.status == 'TX':
                # 只有完成 TX 的节点才结算
                p_coll = 0
                i_avg = 0
                if node.obs_total_slots > 0:
                    p_coll = node.obs_busy_slots / node.obs_total_slots
                    i_avg = quantize_rssi(node.obs_accum_interference / node.obs_total_slots)

                is_aggressive = (node.chosen_th_watt >= dbm_to_watt(TH_HIGH))

                if node.ack_received:
                    r_outcome = ALPHA
                    if is_aggressive: r_outcome += LAMBDA_RS * i_avg
                    node.history_status[node.target_id] = 1
                    node.local_queues[node.target_id] = max(0, node.local_queues[node.target_id] - 1)
                    total_success += 1
                else:
                    r_outcome = -ALPHA
                    if is_aggressive: r_outcome -= DELTA_PENALTY
                    node.history_status[node.target_id] = -1

                r_delay = KAPPA_DELAY * node.obs_total_slots
                reward = r_outcome - r_delay + (BETA * p_coll)  # 加上拥塞奖励

                next_state = node.get_state_vector()

                # agent训练
                node.agent.memory.push(node.decision_state, node.current_action_idx,
                                       reward, next_state, False)
                node.agent.update()

                node.status = 'IDLE'

        return total_success