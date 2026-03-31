import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import random
import json
import os
from datetime import datetime

from config import *
from node import Node
from utils import get_distance, normalize_interference_dist

try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'SimSun']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ThesisEnv:
    def __init__(self, num_nodes, protocol='DQN', cw_size=FIXED_CW):
        self.num_nodes = num_nodes
        self.protocol = protocol
        self.cw_size = cw_size

        xs = np.random.uniform(0, AREA_SIZE, self.num_nodes)
        ys = np.random.uniform(0, AREA_SIZE, self.num_nodes)

        self.nodes = [Node(i, (xs[i], ys[i])) for i in range(self.num_nodes)]
        for node in self.nodes:
            node.init_neighbors(self.nodes)

    def get_min_tx_distance(self, target_node, tx_nodes):
        min_dist = float('inf')
        for tx in tx_nodes:
            if tx.id == target_node.id: continue
            d = get_distance(tx.pos, target_node.pos)
            if d < min_dist: min_dist = d
        return min_dist

    def run_slot(self):
        for node in self.nodes:
            if node.status == 'IDLE':
                if not node.neighbors: continue

                if self.protocol == 'DQN':
                    mask = [False] * ACTION_DIM
                    num_rs = len(RS_SET)
                    for i, n_id in enumerate(node.neighbors):
                        for k in range(i * num_rs, i * num_rs + num_rs): mask[k] = True
                    if not any(mask): continue

                    state = node.get_state_vector()
                    action_idx = node.agent.select_action(state, mask)

                    rs_idx = action_idx % num_rs
                    n_idx = action_idx // num_rs
                    node.target_id = node.neighbors[n_idx]
                    node.chosen_rs = RS_SET[rs_idx]
                    node.decision_state = state
                    node.current_action_idx = action_idx
                    node.target_n_idx = n_idx

                # ==========================================
                # CSMA/CA 的三种固定感知距离 (CCA 阈值) 设定
                # ==========================================
                elif self.protocol == 'CSMA_UltraHighCCA':
                    # 【新增】：极度激进，感知距离仅为通信距离的 0.8 倍
                    node.target_id = random.choice(node.neighbors)
                    node.chosen_rs = 0.8 * COMMUNICATION_RANGE

                elif self.protocol == 'CSMA_HighCCA':
                    # 激进，略大于通信距离
                    node.target_id = random.choice(node.neighbors)
                    node.chosen_rs = 1.1 * COMMUNICATION_RANGE

                elif self.protocol == 'CSMA_LowCCA':
                    # 保守，1.6 倍通信距离
                    node.target_id = random.choice(node.neighbors)
                    node.chosen_rs = 1.6 * COMMUNICATION_RANGE

                node.backoff_counter = np.random.randint(0, self.cw_size + 1)
                node.status = 'BACKOFF'

        active_tx_nodes = []
        for t in range(self.cw_size + 1):
            snapshot_emitters = [n for n in self.nodes if n.status == 'TX']
            nodes_starting_tx = []

            for node in self.nodes:
                if node.status == 'BACKOFF':
                    d_min = self.get_min_tx_distance(node, snapshot_emitters)
                    if self.protocol == 'DQN':
                        node.sense_history.append(normalize_interference_dist(d_min))
                    is_busy = d_min < node.chosen_rs
                    if not is_busy:
                        node.backoff_counter -= 1
                    if node.backoff_counter < 0:
                        nodes_starting_tx.append(node)

            for node in nodes_starting_tx:
                node.status = 'TX'
                active_tx_nodes.append(node)

        tx_nodes = [n for n in self.nodes if n.status == 'TX']
        total_success = 0

        for tx in tx_nodes:
            rx = self.nodes[tx.target_id]
            interfering_tx_nodes = [n for n in tx_nodes if n.id != tx.id]
            rx_d_min = self.get_min_tx_distance(rx, interfering_tx_nodes)
            is_success = (rx.status != 'TX') and (rx_d_min > COMMUNICATION_RANGE)

            if is_success:
                total_success += 1

            if self.protocol == 'DQN':
                tx_d_min = self.get_min_tx_distance(tx, tx_nodes)
                k_aggressiveness = 0.0
                if tx_d_min < 2.0 * COMMUNICATION_RANGE:
                    clamped_d = max(tx_d_min, MIN_SENSE_RANGE)
                    k_aggressiveness = (2.0 * COMMUNICATION_RANGE - clamped_d) / (
                            2.0 * COMMUNICATION_RANGE - MIN_SENSE_RANGE)

                reward = (REWARD_SUCCESS + k_aggressiveness) if is_success else (REWARD_FAIL - k_aggressiveness)
                tx.update_link_quality(tx.target_n_idx, is_success)
                next_state = tx.get_state_vector()
                tx.agent.memory.push(tx.decision_state, tx.current_action_idx, reward, next_state, False)
                tx.agent.update()

            tx.status = 'IDLE'

        return total_success


def plot_and_save_topology(env, filepath):
    plt.figure(figsize=(8, 8))
    for node in env.nodes:
        for neighbor_id in node.neighbors:
            if node.id < neighbor_id:
                neighbor = next(n for n in env.nodes if n.id == neighbor_id)
                plt.plot([node.pos[0], neighbor.pos[0]], [node.pos[1], neighbor.pos[1]],
                         color='gray', linestyle='--', alpha=0.3)
    for node in env.nodes:
        plt.scatter(node.pos[0], node.pos[1], c='dodgerblue', s=120, edgecolors='black', zorder=5)
        plt.text(node.pos[0] + 1.5, node.pos[1] + 1.5, str(node.id), fontsize=10, fontweight='bold', zorder=6)

    plt.title(f"随机生成网络拓扑图 (节点数: {env.num_nodes})", fontsize=16)
    plt.xlabel("X 坐标 (m)", fontsize=14)
    plt.ylabel("Y 坐标 (m)", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(-5, AREA_SIZE + 5)
    plt.ylim(-5, AREA_SIZE + 5)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def run_single_simulation(num_nodes, protocol, total_slots, stats_interval, cw_size):
    print(f"  -> 正在运行 {protocol} ...")
    set_seed(SEED)
    env = ThesisEnv(num_nodes=num_nodes, protocol=protocol, cw_size=cw_size)
    history_throughput = []
    history_eps = []  # 【新增】：记录探索率 epsilon
    acc_success = 0

    for s in range(total_slots):
        succ = env.run_slot()
        acc_success += succ

        if (s + 1) % stats_interval == 0:
            history_throughput.append(acc_success)

            # 仅 DQN 记录真实的 Epsilon 演化
            if protocol == 'DQN':
                agent = env.nodes[0].agent  # 提取 0 号节点的 epsilon 作为代表
                current_eps = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(
                    -1. * agent.steps_done / EPSILON_DECAY)
                history_eps.append(current_eps)

            if (s + 1) % 5000 == 0:
                print(f"      [进度] 时隙 {s + 1:05d} / {total_slots} | 当前吞吐量: {acc_success}")
            acc_success = 0

        if protocol == 'DQN' and (s + 1) % 1000 == 0:
            for node in env.nodes:
                node.agent.target_net.load_state_dict(node.agent.policy_net.state_dict())

    return history_throughput, history_eps


if __name__ == "__main__":
    N_NODES = 40
    T_SLOTS = 100000
    INTERVAL = 100
    CW_SIZE = 15

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"Exp1_Convergence_{N_NODES}Nodes_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"====== 开始实验一：时序收敛与 Epsilon 对比 (结果将保存在 {out_dir}) ======")

    set_seed(SEED)
    dummy_env = ThesisEnv(num_nodes=N_NODES, protocol='DQN', cw_size=CW_SIZE)
    plot_and_save_topology(dummy_env, os.path.join(out_dir, "Exp1_Topology.png"))

    # --- 运行四组协议 ---
    ts_dqn, eps_history = run_single_simulation(N_NODES, 'DQN', T_SLOTS, INTERVAL, CW_SIZE)
    ts_ultra, _ = run_single_simulation(N_NODES, 'CSMA_UltraHighCCA', T_SLOTS, INTERVAL, CW_SIZE)
    ts_high, _ = run_single_simulation(N_NODES, 'CSMA_HighCCA', T_SLOTS, INTERVAL, CW_SIZE)
    ts_low, _ = run_single_simulation(N_NODES, 'CSMA_LowCCA', T_SLOTS, INTERVAL, CW_SIZE)

    # 平滑处理
    window = 10
    smooth_dqn = np.convolve(ts_dqn, np.ones(window) / window, mode='valid').tolist()
    smooth_ultra = np.convolve(ts_ultra, np.ones(window) / window, mode='valid').tolist()
    smooth_high = np.convolve(ts_high, np.ones(window) / window, mode='valid').tolist()
    smooth_low = np.convolve(ts_low, np.ones(window) / window, mode='valid').tolist()

    # Epsilon 不需要平滑，直接截取对应长度
    eps_trimmed = eps_history[:len(smooth_dqn)]

    x_axis = (np.arange(len(smooth_dqn)) * INTERVAL).tolist()

    # ==========================================
    # 绘制双 Y 轴图表
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # --- 左轴：吞吐量 ---
    color_dqn = '#e41a1c'
    color_ultra = '#9467bd'
    color_high = '#1f77b4'
    color_low = '#ff7f0e'

    line1, = ax1.plot(x_axis, smooth_ultra, label=r'CSMA/CA (极高CCA阈值 $R_s=0.8R_c$)', color=color_ultra,
                      linestyle=':')
    line2, = ax1.plot(x_axis, smooth_high, label=r'CSMA/CA (高CCA阈值 $R_s=1.1R_c$)', color=color_high, linestyle='--')
    line3, = ax1.plot(x_axis, smooth_low, label=r'CSMA/CA (低CCA阈值 $R_s=1.6R_c$)', color=color_low, linestyle='-.')
    line4, = ax1.plot(x_axis, smooth_dqn, label='本文 DRL-MAC 协议', color=color_dqn, linewidth=2.5)

    ax1.set_xlabel("时间 (时隙)", fontsize=14)
    ax1.set_ylabel(f"成功发送数据包数 / {INTERVAL} 时隙", fontsize=14, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # --- 右轴：探索率 Epsilon ---
    ax2 = ax1.twinx()
    color_eps = '#2ca02c'
    line5, = ax2.plot(x_axis, eps_trimmed, label=r'DQN 探索率 $\epsilon$ (右轴)', color=color_eps, linewidth=2,
                      linestyle='-')
    ax2.set_ylabel(r"探索率 $\epsilon$ (Epsilon)", fontsize=14, color=color_eps)
    ax2.tick_params(axis='y', labelcolor=color_eps)
    ax2.set_ylim(-0.05, 1.05)  # 探索率固定在 0 到 1 之间

    plt.title(f"吞吐量收敛与 $\epsilon $ 衰减演化对比 (N={N_NODES})", fontsize=16, pad=15)

    # 合并图例
    lines = [line4, line1, line2, line3, line5]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=11, framealpha=0.9)

    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, f"Exp1_Convergence_with_Epsilon.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- 保存 JSON 数据 ---
    raw_data = {
        "config": {"nodes": N_NODES, "total_slots": T_SLOTS, "interval": INTERVAL, "cw": CW_SIZE},
        "x_axis": x_axis,
        "epsilon_history": eps_trimmed,
        "smooth_ts": {
            "DQN": smooth_dqn,
            "UltraHighCCA": smooth_ultra,
            "HighCCA": smooth_high,
            "LowCCA": smooth_low
        }
    }
    with open(os.path.join(out_dir, "exp1_data.json"), 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 实验一完成！双Y轴折线图、拓扑图和JSON数据已完整保存。")