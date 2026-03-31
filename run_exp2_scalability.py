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


# ==========================================
# 环境定义
# ==========================================
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
                # 【新增】加入四种协议的判定，包含极度激进组
                # ==========================================
                elif self.protocol == 'CSMA_UltraHighCCA':
                    node.target_id = random.choice(node.neighbors)
                    node.chosen_rs = 0.8 * COMMUNICATION_RANGE

                elif self.protocol == 'CSMA_HighCCA':
                    node.target_id = random.choice(node.neighbors)
                    node.chosen_rs = 1.1 * COMMUNICATION_RANGE

                elif self.protocol == 'CSMA_LowCCA':
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


# ==========================================
# 辅助与执行函数
# ==========================================
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

    plt.title(f"网络拓扑图 (节点数: {env.num_nodes})", fontsize=16)
    plt.xlabel("X 坐标 (m)", fontsize=14)
    plt.ylabel("Y 坐标 (m)", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(-5, AREA_SIZE + 5)
    plt.ylim(-5, AREA_SIZE + 5)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def run_simulation(num_nodes, protocol, total_slots, cw_size):
    """
    【改动 4 & 3 实现】
    执行仿真并只返回最后一段窗口的均值。
    """
    print(f"    -> 正在运行 {protocol} (时隙数: {total_slots})...")
    set_seed(SEED)  # 确保每次跑的初始拓扑一样
    env = ThesisEnv(num_nodes=num_nodes, protocol=protocol, cw_size=cw_size)

    history = []
    acc_success = 0
    INTERVAL = 100

    # 为了加快效率，如果不需要画时序图，我们只存按 INTERVAL 聚合的数据
    for s in range(total_slots):
        succ = env.run_slot()
        acc_success += succ

        if (s + 1) % INTERVAL == 0:
            history.append(acc_success)
            acc_success = 0

            # 进度打印
            if (s + 1) % 10000 == 0:
                print(f"       [进度] {s + 1:06d} / {total_slots}")

        if protocol == 'DQN' and (s + 1) % 1000 == 0:
            for node in env.nodes:
                node.agent.target_net.load_state_dict(node.agent.policy_net.state_dict())

    # 【核心：统计稳态均值】
    # DQN 取最后 100 个 INTERVAL (即 10,000 时隙) 的均值
    # CSMA 取全部历史的均值 (因为它不需要收敛，截短跑的全部数据都可以算作稳态)
    if protocol == 'DQN':
        steady_state_avg = float(np.mean(history[-100:]))
    else:
        steady_state_avg = float(np.mean(history))

    return steady_state_avg


# ==========================================
# 实验主函数 (带花纹的条形图绘制)
# ==========================================
if __name__ == "__main__":
    # 【改动 5】增加节点数量以展现 CSMA 拥塞崩溃
    NODE_LIST = [20, 40, 60, 80, 100, 120]

    # 【改动 2 & 4】DQN 跑 10万，CSMA 只跑 1万 (提速 10 倍！)
    T_SLOTS_DQN = 100000
    T_SLOTS_CSMA = 10000

    # 强制锁死 CW，绝不动态扩容
    CW_FIXED = 15

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f"Exp2_Scalability_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"====== 开始实验二：稳态吞吐量条形图对比 (结果将保存在 {out_dir}) ======")

    steady_dqn, steady_ultra, steady_high, steady_low = [], [], [], []

    for n in NODE_LIST:
        print(f"\n--- 测试规模: {n} 节点 ---")

        # 【改动 7】保存当前规模的拓扑图
        set_seed(SEED)
        dummy_env = ThesisEnv(num_nodes=n, protocol='DQN', cw_size=CW_FIXED)
        plot_and_save_topology(dummy_env, os.path.join(out_dir, f"Topology_N{n}.png"))

        # 运行四组协议
        avg_dqn = run_simulation(n, 'DQN', T_SLOTS_DQN, CW_FIXED)
        avg_ultra = run_simulation(n, 'CSMA_UltraHighCCA', T_SLOTS_CSMA, CW_FIXED)
        avg_high = run_simulation(n, 'CSMA_HighCCA', T_SLOTS_CSMA, CW_FIXED)
        avg_low = run_simulation(n, 'CSMA_LowCCA', T_SLOTS_CSMA, CW_FIXED)

        steady_dqn.append(avg_dqn)
        steady_ultra.append(avg_ultra)
        steady_high.append(avg_high)
        steady_low.append(avg_low)

    # ==========================================
    # 【改动 6】绘制带花纹的精美条形图 (Bar Chart)
    # ==========================================
    plt.figure(figsize=(12, 7))

    x = np.arange(len(NODE_LIST))  # 节点数标签位置
    width = 0.2  # 柱子宽度

    # 颜色配置与花纹 (Hatch) 配置
    # 采用高对比度颜色，并通过 hatch 确保即使黑白打印也能区分
    bar_ultra = plt.bar(x - 1.5 * width, steady_ultra, width, label=r'CSMA/CA (极高CCA阈值 $0.8R_c$)', color='#9467bd',
                        edgecolor='black', hatch='xx')
    bar_high = plt.bar(x - 0.5 * width, steady_high, width, label=r'CSMA/CA (高CCA阈值 $1.1R_c$)', color='#1f77b4',
                       edgecolor='black', hatch='//')
    bar_low = plt.bar(x + 0.5 * width, steady_low, width, label=r'CSMA/CA (低CCA阈值 $1.6R_c$)', color='#ff7f0e',
                      edgecolor='black', hatch='\\\\')
    bar_dqn = plt.bar(x + 1.5 * width, steady_dqn, width, label='本文 DRL-MAC 协议', color='#d62728', edgecolor='black')

    plt.title("不同网络规模下的稳态吞吐量条形图对比", fontsize=18, pad=15)
    plt.xlabel("网络规模 (节点数)", fontsize=15)
    plt.ylabel("稳态平均吞吐量 (包 / 100时隙)", fontsize=15)
    plt.xticks(x, NODE_LIST, fontsize=13)
    plt.yticks(fontsize=13)

    plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    # 为 DQN 柱子添加数值标签，突出优势
    for i, rect in enumerate(bar_dqn):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.1f}', ha='center', va='bottom', fontsize=11,
                 fontweight='bold', color='#d62728')

    plt.tight_layout()
    chart_filename = os.path.join(out_dir, "Exp2_Steady_BarChart.png")
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # 保存 JSON 数据
    raw_data = {
        "config": {"nodes": NODE_LIST, "t_slots_dqn": T_SLOTS_DQN, "t_slots_csma": T_SLOTS_CSMA, "cw_fixed": CW_FIXED},
        "steady_throughput": {
            "DQN": steady_dqn,
            "UltraHighCCA": steady_ultra,
            "HighCCA": steady_high,
            "LowCCA": steady_low
        }
    }
    with open(os.path.join(out_dir, "exp2_data.json"), 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 实验二完成！条形图、拓扑图和稳态 JSON 数据已全部保存。")