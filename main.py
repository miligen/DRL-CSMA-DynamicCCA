# main.py
import matplotlib.pyplot as plt
import matplotlib
from environment import AdHocEnv
from config import *
import numpy as np

try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告：未找到中文字体，图表可能显示乱码。")


def main():
    print(f"仿真开始 | 总帧数: {NUM_FRAMES} | 节点数: {NUM_NODES}")
    env = AdHocEnv()

    # 打印拓扑信息，确认邻居关系
    print("\n=== 拓扑结构信息 (Topology Info) ===")
    for node in env.nodes:
        n_ids = [n for n in node.neighbors]
        print(f"Node {node.id} (位置 {node.pos}): 邻居列表 {n_ids}")
    print("==================================\n")

    # 定义两个列表，分别存利用率和奖励
    history_util = []
    history_reward = []
    history_real_util = []

    for f in range(NUM_FRAMES):
        env.reset()

        f_success = 0
        f_reward_sum = 0
        f_real_success = 0

        for _ in range(SLOTS_PER_FRAME):
            s, r, rs = env.run_slot(current_frame=f)
            f_success += s
            f_reward_sum += r
            f_real_success += rs

        # 计算本帧平均指标
        util = f_success / SLOTS_PER_FRAME
        avg_reward = f_reward_sum / SLOTS_PER_FRAME
        real_util = f_real_success / SLOTS_PER_FRAME

        # 存入历史列表
        history_util.append(util)
        history_reward.append(avg_reward)
        history_real_util.append(real_util)

        if f % 10 == 0:
            print(f"Frame {f} | 时隙利用率: {util:.2f} | 平均每时隙奖励: {avg_reward:.2f}")

        if f % TARGET_UPDATE == 0:
            # 更新所有节点的 Target Net
            for node in env.nodes:
                node.agent.target_net.load_state_dict(node.agent.policy_net.state_dict())

    # === 绘图部分 ===
    # 设置窗口大小 (宽10, 高8)，因为有两个图
    plt.figure(figsize=(10, 12))

    # 平滑窗口大小
    window_size = 10

    # --- 图 1: 时隙利用率 ---
    plt.subplot(3, 1, 1)
    plt.plot(history_util, alpha=0.3, color='blue', label='原始数据')

    # 计算时隙利用率平滑曲线
    if len(history_util) > window_size:
        smooth_util = np.convolve(history_util, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_util, color='blue', linewidth=2, label='平滑曲线')

    plt.title("每帧时隙利用率")
    plt.ylabel("利用率 (成功包数/时隙)")
    plt.legend(loc='upper left')
    plt.grid(True)

    # --- 图 2: 真实时隙利用率 ---
    plt.subplot(3, 1, 2)  # 改为 3行1列，第2张
    plt.plot(history_real_util, alpha=0.3, color='green', label='原始数据')
    if len(history_real_util) > window_size:
        smooth_real = np.convolve(history_real_util, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_real, color='green', linewidth=2, label='平滑曲线')
    plt.title("真实时隙利用率")
    plt.ylabel("真实利用率")
    plt.legend(loc='upper left')
    plt.grid(True)

    # --- 图 3: 平均奖励 ---
    plt.subplot(3, 1, 3)
    plt.plot(history_reward, alpha=0.3, color='orange', label='原始奖励')

    # 计算奖励平滑曲线
    if len(history_reward) > window_size:
        smooth_reward = np.convolve(history_reward, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smooth_reward, color='red', linewidth=2, label='平滑曲线')

    plt.title("训练收敛")
    plt.ylabel("平均奖励 (Reward)")
    plt.xlabel("训练帧数 (Frame)")
    plt.legend(loc='upper left')
    plt.grid(True)

    # 调整子图间距
    plt.tight_layout()

    filename = f"result_F{NUM_FRAMES}_S{SLOTS_PER_FRAME}_EPS_DECAY{EPSILON_DECAY}_EPS_END{EPSILON_END}.png"

    plt.savefig(filename)
    print(f"\n仿真结束，结果图已保存为: {filename}")
    plt.show()


if __name__ == "__main__":
    main()