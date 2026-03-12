import matplotlib.pyplot as plt
import matplotlib
from environment import AdHocEnv
from config import *
from utils import *  # 需要引入计算 sinr 相关的工具
import numpy as np
from datetime import datetime

try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass


def estimate_optimal_throughput(env, iterations=2000):
    """
    【新增】通过蒙特卡洛贪心搜索，估算当前拓扑下的理论最大并发传输数（空间复用上限）
    """
    all_links = []
    for node in env.nodes:
        for neighbor_id in node.neighbors:
            all_links.append((node.id, neighbor_id))

    max_concurrent = 0
    best_set = []

    # 随机打乱链路顺序进行贪心尝试，寻找能满足半双工和 SINR>=3dB 的最大链路集合
    for _ in range(iterations):
        np.random.shuffle(all_links)
        active_links = []
        active_nodes = set()

        for tx, rx in all_links:
            # 1. 满足半双工约束：节点不能同时收发，也不能同时与两个节点通信
            if tx in active_nodes or rx in active_nodes:
                continue

            # 尝试加入当前链路
            active_links.append((tx, rx))

            # 2. 检查当前集合中所有链路的 SINR 是否都达标
            valid = True
            tx_nodes = [env.nodes[t] for t, _ in active_links]

            for active_tx, active_rx in active_links:
                r_node = env.nodes[active_rx]
                sig = dbm_to_watt(TX_POWER_DBM + env.gain_matrix[active_tx, active_rx])
                intf = env.calculate_interference(r_node, tx_nodes)
                intf = max(0.0, intf - sig)

                if calculate_sinr(sig, intf) < SINR_THRESHOLD_DB:
                    valid = False
                    break  # 只要有一条链路不达标，当前组合就失效

            if valid:
                active_nodes.add(tx)
                active_nodes.add(rx)
            else:
                active_links.pop()  # 剔除破坏规则的链路

        if len(active_links) > max_concurrent:
            max_concurrent = len(active_links)
            best_set = list(active_links)

    return max_concurrent, best_set


def plot_topology(env, run_id, optimal_links=None):
    """【修改】绘制拓扑，支持传入 run_id 和标出最优链路"""
    plt.figure(figsize=(8, 8))

    # 画线：所有潜在的物理连接（灰色虚线）
    for node in env.nodes:
        for neighbor_id in node.neighbors:
            neighbor = next(n for n in env.nodes if n.id == neighbor_id)
            plt.plot([node.pos[0], neighbor.pos[0]],
                     [node.pos[1], neighbor.pos[1]],
                     color='gray', linestyle='--', alpha=0.3)

    # 【新增】画线：理论最优的并发链路（红色加粗实线）
    if optimal_links:
        for idx, (tx, rx) in enumerate(optimal_links):
            tx_node = env.nodes[tx]
            rx_node = env.nodes[rx]
            plt.plot([tx_node.pos[0], rx_node.pos[0]],
                     [tx_node.pos[1], rx_node.pos[1]],
                     color='red', linestyle='-', linewidth=3, alpha=0.8,
                     label='理论最优并发链路' if idx == 0 else "")
            # 画一个箭头指示传输方向
            plt.arrow(tx_node.pos[0], tx_node.pos[1],
                      (rx_node.pos[0] - tx_node.pos[0]) * 0.8,
                      (rx_node.pos[1] - tx_node.pos[1]) * 0.8,
                      head_width=1.5, head_length=2, fc='red', ec='red', alpha=0.8)

    # 画点：节点位置
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

    # 使用包含时间戳的统一命名
    filename = f"topology_{run_id}.png"
    plt.savefig(filename, dpi=150)
    print(f"拓扑图已保存为: {filename}")
    plt.close()


def main():
    # 【新增】生成本次实验的全局唯一时间戳 ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[{run_id}] 仿真开始 | 总时隙数: {TOTAL_SLOTS} | 统计间隔: {STATS_INTERVAL}")
    env = AdHocEnv()

    # 【新增】计算理论最优吞吐量
    print("\n正在估算理论最大并发传输数 (这可能需要几秒钟)...")
    opt_throughput, opt_links = estimate_optimal_throughput(env, iterations=5000)
    print(f">>> 理论最优吞吐量 (空间复用上限): {opt_throughput} 包/时隙")
    print(f">>> 最优并发链路组合: {opt_links}")

    # 绘制拓扑
    plot_topology(env, run_id, opt_links)

    print("\n=== 拓扑结构信息 ===")
    for node in env.nodes:
        print(f"Node {node.id}: 邻居列表 {node.neighbors}")
    print("==================\n")

    history_util = []
    history_reward = []

    acc_success = 0
    acc_reward = 0

    for s in range(TOTAL_SLOTS):
        succ, rw = env.run_slot()
        acc_success += succ
        acc_reward += rw

        if (s + 1) % STATS_INTERVAL == 0:
            util = acc_success / STATS_INTERVAL
            avg_rw = acc_reward / STATS_INTERVAL

            history_util.append(util)
            history_reward.append(avg_rw)

            interval_idx = (s + 1) // STATS_INTERVAL

            if interval_idx % 10 == 0:
                print(f"时隙 {s + 1} | 吞吐量(包/时隙): {util:.2f} | 平均奖励: {avg_rw:.2f}")

            if interval_idx % TARGET_UPDATE == 0:
                for node in env.nodes:
                    node.agent.target_net.load_state_dict(node.agent.policy_net.state_dict())

            acc_success = 0
            acc_reward = 0

    # === 绘图部分 ===
    plt.figure(figsize=(10, 8))
    window_size = max(10, len(history_util) // 50)

    # 图 1: 系统吞吐量
    plt.subplot(2, 1, 1)
    plt.plot(history_util, alpha=0.3, color='blue', label='原始数据 (每20时隙)')

    if len(history_util) > window_size:
        smooth_util = np.convolve(history_util, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_util, color='blue', linewidth=2, label='平滑曲线 (Agent表现)')

    # 【新增】在吞吐量图中画出理论天花板（红色虚线）
    plt.axhline(y=opt_throughput, color='red', linestyle='--', linewidth=2, label=f'理论最优上限 ({opt_throughput})')

    plt.title("系统吞吐量 (Throughput)")
    plt.ylabel("成功包数/时隙")
    plt.legend()
    plt.grid(True)

    # 图 2: 训练收敛
    plt.subplot(2, 1, 2)
    plt.plot(history_reward, alpha=0.3, color='orange', label='原始奖励')
    if len(history_reward) > window_size:
        smooth_reward = np.convolve(history_reward, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_reward, color='red', linewidth=2, label='平滑曲线')
    plt.title("训练收敛 (Reward)")
    plt.ylabel("平均奖励")
    plt.xlabel(f"运行周期 (x{STATS_INTERVAL} 时隙)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # 【修改】使用时间戳统一命名
    filename = f"result_{run_id}_T{TOTAL_SLOTS}.png"
    plt.savefig(filename, dpi=150)
    print(f"\n仿真结束，结果图已保存为: {filename}")
    plt.show()


if __name__ == "__main__":
    main()