import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import random
from datetime import datetime

from config import *
from environment import AdHocEnv
from main import estimate_optimal_throughput, plot_topology

try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass


def set_seed(seed):
    """【关键】重置随机种子，确保两组实验生成一模一样的物理拓扑！"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_experiment(use_link_quality, env_name):
    """运行单次仿真实验并返回历史数据"""
    print(f"\n[{env_name}] 初始化环境 (启用链路质量矩阵: {use_link_quality}) ...")

    # 在初始化前重置种子，保证两次生成的节点坐标矩阵完全一致
    set_seed(SEED)
    env = AdHocEnv(use_link_quality=use_link_quality)

    history_util = []
    history_reward = []
    acc_success = 0
    acc_reward = 0
    acc_attempts = 0  # 【新增】记录发送尝试总次数
    history_collision = []  # 【新增】记录碰撞率历史

    print(f"[{env_name}] 开始训练，总时隙数: {TOTAL_SLOTS} ...")
    for s in range(TOTAL_SLOTS):
        succ, rw, attempts = env.run_slot()  # 【注意】这里假设 env.run_slot() 返回了 attempts
        acc_success += succ
        acc_reward += rw
        acc_attempts += attempts  # 【新增】

        # 定期统计与记录
        if (s + 1) % STATS_INTERVAL == 0:
            util = acc_success / STATS_INTERVAL
            avg_rw = acc_reward / STATS_INTERVAL
            # 【新增】计算碰撞率
            col_rate = (acc_attempts - acc_success) / acc_attempts if acc_attempts > 0 else 0.0

            history_util.append(util)
            history_reward.append(avg_rw)
            history_collision.append(col_rate)  # 【新增】

            interval_idx = (s + 1) // STATS_INTERVAL

            if interval_idx % 100 == 0:  # 降低打印频率，以免刷屏
                print(f"[{env_name}] 时隙 {s + 1:05d} | 吞吐量: {util:.2f} | 平均奖励: {avg_rw:.2f}")

            # 定期更新 Target Network
            if interval_idx % TARGET_UPDATE == 0:
                for node in env.nodes:
                    node.agent.target_net.load_state_dict(node.agent.policy_net.state_dict())

            # 清空累加器
            acc_success = 0
            acc_reward = 0
            acc_attempts = 0

    return history_util, history_reward, history_collision, env # 【修改返回值】


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"====== 开始消融实验 (Ablation Study) | 任务ID: {run_id} ======")

    # 1. 运行算法 A (基准线 Baseline：无目标链路质量感知)
    util_base, reward_base, env_base = run_experiment(use_link_quality=False, env_name="Baseline (A)")

    # 2. 运行算法 B (改进版 Proposed：加入 H_i 矩阵感知隐藏终端)
    util_prop, reward_prop, env_prop = run_experiment(use_link_quality=True, env_name="Proposed (B)")

    # 3. 估算环境最优理论吞吐量 (用任意一个 env 算即可，因为种子一样，拓扑一样)
    print("\n[分析] 正在估算当前拓扑的理论最大并发传输数...")
    opt_throughput, opt_links = estimate_optimal_throughput(env_base, iterations=5000)
    print(f">>> 理论最优吞吐量 (空间复用上限): {opt_throughput} 包/时隙")

    # 绘制带最优连线的拓扑图作为存档
    plot_topology(env_base, run_id, opt_links)

    # ================= 绘制消融实验对比图 =================
    plt.figure(figsize=(12, 14))
    window_size = max(10, len(util_base) // 50)

    # --- 图 1: 吞吐量对比 ---
    plt.subplot(2, 1, 1)

    # 绘制 Baseline (蓝色系)
    plt.plot(util_base, alpha=0.2, color='royalblue')
    if len(util_base) > window_size:
        smooth_base = np.convolve(util_base, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_base, color='blue', linewidth=2, label='Baseline (仅干扰历史)')

    # 绘制 Proposed (绿色系)
    plt.plot(util_prop, alpha=0.2, color='limegreen')
    if len(util_prop) > window_size:
        smooth_prop = np.convolve(util_prop, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_prop, color='green', linewidth=2.5, label='Proposed (干扰历史 + 链路质量 H_i)')

    # 绘制理论上限
    plt.axhline(y=opt_throughput, color='red', linestyle='--', linewidth=2,
                label=f'Theoretical Optimal ({opt_throughput})')

    plt.title("System Throughput Comparison (Ablation Study)", fontsize=14)
    plt.ylabel("Successful Packets / Slot")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.5)

    # --- 图 2: 奖励收敛对比 ---
    plt.subplot(2, 1, 2)

    # 绘制 Baseline (橙色)
    if len(reward_base) > window_size:
        smooth_rw_base = np.convolve(reward_base, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_rw_base, color='orange', linewidth=2, label='Baseline Reward')

    # 绘制 Proposed (红色)
    if len(reward_prop) > window_size:
        smooth_rw_prop = np.convolve(reward_prop, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_rw_prop, color='red', linewidth=2.5, label='Proposed Reward')

    plt.title("Convergence & Reward Comparison", fontsize=14)
    plt.ylabel("Average Reward")
    plt.xlabel(f"Training Epochs (x{STATS_INTERVAL} Slots)")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.5)

    # --- 图 3: 碰撞率对比 ---
    plt.subplot(3, 1, 3)
    if len(col_base) > window_size:
        smooth_col_base = np.convolve(col_base, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_col_base, color='blue', linewidth=2, label='Baseline Collision Rate')

    if len(col_prop) > window_size:
        smooth_col_prop = np.convolve(col_prop, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_col_prop, color='green', linewidth=2.5, label='Proposed Collision Rate')

    plt.title("Collision Rate Comparison", fontsize=14)
    plt.ylabel("Collision Rate")
    plt.xlabel(f"Training Epochs (x{STATS_INTERVAL} Slots)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    filename = f"Ablation_Result_{run_id}.png"
    plt.savefig(filename, dpi=200)
    print(f"\n====== 消融实验完成！对比结果已保存为: {filename} ======")
    plt.show()


if __name__ == "__main__":
    main()