import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import random
from datetime import datetime

from config import *
from environment import AdHocEnv
from utils import get_distance

try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass


# ========================================================
# 直接将评估与画图函数内置，彻底消除 Import / NameError 依赖
# ========================================================
def estimate_optimal_throughput(env, iterations=2000):
    """基于布尔距离模型的理论并发上限估算"""
    all_links = []
    for node in env.nodes:
        for neighbor_id in node.neighbors:
            all_links.append((node.id, neighbor_id))

    max_concurrent = 0
    best_set = []

    for _ in range(iterations):
        np.random.shuffle(all_links)
        active_links = []
        active_nodes = set()

        for tx, rx in all_links:
            if tx in active_nodes or rx in active_nodes:
                continue
            active_links.append((tx, rx))
            valid = True

            # 基于布尔距离的冲突判断
            for active_tx, active_rx in active_links:
                r_node = env.nodes[active_rx]
                d_min = float('inf')
                # 检查所有其他发送者到这个接收者的距离
                for other_tx, _ in active_links:
                    if other_tx != active_tx:
                        d = get_distance(env.nodes[other_tx].pos, r_node.pos)
                        if d < d_min: d_min = d

                # 如果有任何其他发送者进入了通信半径 Rc，则产生干扰，组合失效
                if d_min <= COMMUNICATION_RANGE:
                    valid = False
                    break

            if valid:
                active_nodes.add(tx)
                active_nodes.add(rx)
            else:
                active_links.pop()

        if len(active_links) > max_concurrent:
            max_concurrent = len(active_links)
            best_set = list(active_links)

    return max_concurrent, best_set


def plot_topology(env, run_id, optimal_links=None):
    """绘制拓扑结构图"""
    plt.figure(figsize=(8, 8))
    for node in env.nodes:
        for neighbor_id in node.neighbors:
            neighbor = next(n for n in env.nodes if n.id == neighbor_id)
            plt.plot([node.pos[0], neighbor.pos[0]],
                     [node.pos[1], neighbor.pos[1]],
                     color='gray', linestyle='--', alpha=0.3)
    if optimal_links:
        for idx, (tx, rx) in enumerate(optimal_links):
            tx_node = env.nodes[tx]
            rx_node = env.nodes[rx]
            plt.plot([tx_node.pos[0], rx_node.pos[0]],
                     [tx_node.pos[1], rx_node.pos[1]],
                     color='red', linestyle='-', linewidth=3, alpha=0.8,
                     label='理论最优并发链路' if idx == 0 else "")
            plt.arrow(tx_node.pos[0], tx_node.pos[1],
                      (rx_node.pos[0] - tx_node.pos[0]) * 0.8,
                      (rx_node.pos[1] - tx_node.pos[1]) * 0.8,
                      head_width=1.5, head_length=2, fc='red', ec='red', alpha=0.8)
    for node in env.nodes:
        plt.scatter(node.pos[0], node.pos[1], c='dodgerblue', s=120, edgecolors='black', zorder=5)
        plt.text(node.pos[0] + 1.5, node.pos[1] + 1.5, str(node.id), fontsize=10, fontweight='bold', zorder=6)
    plt.title(f"网络拓扑图 (节点数: {env.num_nodes})")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(-5, AREA_SIZE + 5)
    plt.ylim(-5, AREA_SIZE + 5)
    filename = f"topology_{run_id}.png"
    plt.savefig(filename, dpi=150)
    plt.close()


# ========================================================
# 消融实验主体逻辑
# ========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(use_adaptive_reward, env_name):
    print(f"\n[{env_name}] 初始化布尔距离环境 (自适应奖励: {use_adaptive_reward}) ...")
    set_seed(SEED)
    env = AdHocEnv(use_adaptive_reward=use_adaptive_reward)

    history_util = []
    history_reward = []
    history_collision = []

    acc_success = 0
    acc_reward = 0
    acc_attempts = 0

    print(f"[{env_name}] 开始训练，总时隙数: {TOTAL_SLOTS} ...")
    for s in range(TOTAL_SLOTS):
        succ, rw, attempts = env.run_slot()
        acc_success += succ
        acc_reward += rw
        acc_attempts += attempts

        if (s + 1) % STATS_INTERVAL == 0:
            util = acc_success / STATS_INTERVAL
            avg_rw = acc_reward / (acc_attempts if acc_attempts > 0 else 1)
            col_rate = (acc_attempts - acc_success) / acc_attempts if acc_attempts > 0 else 0.0

            history_util.append(util)
            history_reward.append(avg_rw)
            history_collision.append(col_rate)

            interval_idx = (s + 1) // STATS_INTERVAL

            if interval_idx % 100 == 0:
                print(
                    f"[{env_name}] 时隙 {s + 1:05d} | 吞吐量: {util:.2f} | 奖励: {avg_rw:.2f} | 碰撞率: {col_rate:.2%}")

            if interval_idx % TARGET_UPDATE == 0:
                for node in env.nodes:
                    node.agent.target_net.load_state_dict(node.agent.policy_net.state_dict())

            acc_success = 0
            acc_reward = 0
            acc_attempts = 0

    return history_util, history_reward, history_collision, env


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"====== 开始布尔距离模型消融实验 | 任务ID: {run_id} ======")

    util_base, reward_base, col_base, env_base = run_experiment(use_adaptive_reward=False, env_name="Baseline (Fixed)")
    util_prop, reward_prop, col_prop, env_prop = run_experiment(use_adaptive_reward=True,
                                                                env_name="Proposed (Adaptive)")

    print("\n[分析] 正在估算当前拓扑的理论最大并发传输数...")
    opt_throughput, opt_links = estimate_optimal_throughput(env_base, iterations=5000)
    print(f">>> 理论最优吞吐量 (空间复用上限): {opt_throughput} 包/时隙")
    plot_topology(env_base, run_id, opt_links)

    plt.figure(figsize=(12, 14))
    window_size = max(10, len(util_base) // 50)

    # --- 图 1: 吞吐量 ---
    plt.subplot(3, 1, 1)
    plt.plot(util_base, alpha=0.2, color='royalblue')
    if len(util_base) > window_size:
        plt.plot(np.convolve(util_base, np.ones(window_size) / window_size, mode='valid'), color='blue', linewidth=2,
                 label='Baseline (Fixed Reward)')
    plt.plot(util_prop, alpha=0.2, color='limegreen')
    if len(util_prop) > window_size:
        plt.plot(np.convolve(util_prop, np.ones(window_size) / window_size, mode='valid'), color='green', linewidth=2.5,
                 label='Proposed (Adaptive Reward)')
    plt.axhline(y=opt_throughput, color='red', linestyle='--', linewidth=2,
                label=f'Theoretical Optimal ({opt_throughput})')
    plt.title("System Throughput Comparison (Distance Boolean Model)", fontsize=14)
    plt.ylabel("Packets / Slot")
    plt.legend()
    plt.grid(True, alpha=0.5)

    # --- 图 2: 奖励 ---
    plt.subplot(3, 1, 2)
    if len(reward_base) > window_size:
        plt.plot(np.convolve(reward_base, np.ones(window_size) / window_size, mode='valid'), color='orange',
                 linewidth=2, label='Baseline Reward')
    if len(reward_prop) > window_size:
        plt.plot(np.convolve(reward_prop, np.ones(window_size) / window_size, mode='valid'), color='red', linewidth=2.5,
                 label='Proposed Reward')
    plt.title("Reward Comparison", fontsize=14)
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True, alpha=0.5)

    # --- 图 3: 碰撞率 ---
    plt.subplot(3, 1, 3)
    if len(col_base) > window_size:
        plt.plot(np.convolve(col_base, np.ones(window_size) / window_size, mode='valid'), color='blue', linewidth=2,
                 label='Baseline Collision Rate')
    if len(col_prop) > window_size:
        plt.plot(np.convolve(col_prop, np.ones(window_size) / window_size, mode='valid'), color='green', linewidth=2.5,
                 label='Proposed Collision Rate')
    plt.title("System Collision Rate Comparison", fontsize=14)
    plt.ylabel("Collision Probability")
    plt.xlabel(f"Training Epochs (x{STATS_INTERVAL} Slots)")
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    filename = f"Ablation_DistModel_{run_id}.png"
    plt.savefig(filename, dpi=200)
    print(f"\n====== 消融实验完成！图表已保存为: {filename} ======")
    plt.show()


if __name__ == "__main__":
    main()