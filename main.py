import matplotlib.pyplot as plt
import matplotlib
from environment import AdHocEnv
from config import *
from utils import *
import numpy as np
from datetime import datetime

try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass


def estimate_optimal_throughput(env, iterations=2000):
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
            tx_nodes = [env.nodes[t] for t, _ in active_links]

            for active_tx, active_rx in active_links:
                r_node = env.nodes[active_rx]
                sig = dbm_to_watt(TX_POWER_DBM + env.gain_matrix[active_tx, active_rx])
                intf = env.calculate_interference(r_node, tx_nodes)
                intf = max(0.0, intf - sig)

                if calculate_sinr(sig, intf) < SINR_THRESHOLD_DB:
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
    print(f"拓扑图已保存为: {filename}")
    plt.close()


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{run_id}] 仿真开始 | 总时隙数: {TOTAL_SLOTS} | 统计间隔: {STATS_INTERVAL}")
    env = AdHocEnv()

    print("\n正在估算理论最大并发传输数 (这可能需要几秒钟)...")
    opt_throughput, opt_links = estimate_optimal_throughput(env, iterations=5000)
    print(f">>> 理论最优吞吐量 (空间复用上限): {opt_throughput} 包/时隙")
    print(f">>> 最优并发链路组合: {opt_links}")

    plot_topology(env, run_id, opt_links)

    print("\n=== 拓扑结构信息 ===")
    for node in env.nodes:
        print(f"Node {node.id}: 邻居列表 {node.neighbors}")
    print("==================\n")

    history_util = []
    history_reward = []
    history_collision = []  # 【新增】记录碰撞率

    acc_success = 0
    acc_reward = 0
    acc_attempts = 0  # 【新增】记录尝试发送的总次数

    for s in range(TOTAL_SLOTS):
        # 【修改】解包三个返回值：成功数、奖励、尝试数
        succ, rw, attempts = env.run_slot()
        acc_success += succ
        acc_reward += rw
        acc_attempts += attempts

        if (s + 1) % STATS_INTERVAL == 0:
            util = acc_success / STATS_INTERVAL
            avg_rw = acc_reward / (acc_attempts if acc_attempts > 0 else 1)

            # 【核心新增】计算统计周期内的平均碰撞率
            col_rate = (acc_attempts - acc_success) / acc_attempts if acc_attempts > 0 else 0.0

            history_util.append(util)
            history_reward.append(avg_rw)
            history_collision.append(col_rate)

            interval_idx = (s + 1) // STATS_INTERVAL

            if interval_idx % 10 == 0:
                print(f"时隙 {s + 1} | 吞吐量: {util:.2f} | 平均奖励: {avg_rw:.2f} | 碰撞率: {col_rate:.2%}")

            if interval_idx % TARGET_UPDATE == 0:
                for node in env.nodes:
                    node.agent.target_net.load_state_dict(node.agent.policy_net.state_dict())

            acc_success = 0
            acc_reward = 0
            acc_attempts = 0

    # === 绘图部分 ===
    # 【修改】画布拉长以容纳 3 个子图
    plt.figure(figsize=(10, 12))
    window_size = max(10, len(history_util) // 50)

    # 图 1: 系统吞吐量
    plt.subplot(3, 1, 1)
    plt.plot(history_util, alpha=0.3, color='blue', label='原始数据 (每20时隙)')
    if len(history_util) > window_size:
        smooth_util = np.convolve(history_util, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_util, color='blue', linewidth=2, label='平滑曲线 (Agent表现)')
    plt.axhline(y=opt_throughput, color='red', linestyle='--', linewidth=2, label=f'理论最优上限 ({opt_throughput})')
    plt.title("System Throughput (Throughput)")
    plt.ylabel("成功包数/时隙")
    plt.legend()
    plt.grid(True)

    # 图 2: 训练收敛
    plt.subplot(3, 1, 2)
    plt.plot(history_reward, alpha=0.3, color='orange', label='原始奖励')
    if len(history_reward) > window_size:
        smooth_reward = np.convolve(history_reward, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_reward, color='red', linewidth=2, label='平滑曲线')
    plt.title("Convergence (Reward with Bonus/Penalty)")
    plt.ylabel("平均奖励")
    plt.legend()
    plt.grid(True)

    # 图 3: 系统碰撞率 (Collision Rate)
    plt.subplot(3, 1, 3)
    plt.plot(history_collision, alpha=0.3, color='gray', label='原始碰撞率')
    if len(history_collision) > window_size:
        smooth_col = np.convolve(history_collision, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_col, color='purple', linewidth=2, label='平滑曲线 (碰撞率)')
    plt.title("System Collision Rate")
    plt.ylabel("碰撞概率")
    plt.xlabel(f"运行周期 (x{STATS_INTERVAL} 时隙)")
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    filename = f"result_{run_id}_T{TOTAL_SLOTS}.png"
    plt.savefig(filename, dpi=150)
    print(f"\n仿真结束，结果图已保存为: {filename}")
    plt.show()


if __name__ == "__main__":
    main()