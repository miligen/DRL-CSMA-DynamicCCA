import matplotlib.pyplot as plt
from environment import AdHocEnv
from config import *
import numpy as np


def main():
    print(f"仿真开始| Frames: {NUM_FRAMES} | Nodes: {NUM_NODES}")
    env = AdHocEnv()

    # 打印拓扑信息，确认邻居关系
    print("\n=== Topology Info ===")
    for node in env.nodes:
        n_ids = [n for n in node.neighbors]
        print(f"Node {node.id} @ {node.pos}: Neighbors {n_ids}")
    print("=====================\n")

    history_util = []

    for f in range(NUM_FRAMES):
        f_success = 0
        for _ in range(SLOTS_PER_FRAME):
            f_success += env.run_slot()

        util = f_success / SLOTS_PER_FRAME
        history_util.append(util)

        if f % 10 == 0:
            # 打印其中一个节点的 epsilon 状态作为参考
            eps_step = env.nodes[0].agent.steps_done
            print(f"Frame {f} | Avg Utility: {util:.2f} | Node 0 Steps: {eps_step}")

        if f % TARGET_UPDATE == 0:
            # 【修改】更新所有节点的 Target Net
            for node in env.nodes:
                node.agent.target_net.load_state_dict(node.agent.policy_net.state_dict())

    # 平滑曲线以便观察
    window_size = 10
    smoothed = np.convolve(history_util, np.ones(window_size) / window_size, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(history_util, alpha=0.3, color='blue', label='Raw')
    plt.plot(smoothed, color='red', linewidth=2, label='Smoothed')
    plt.title("Independent Learners Performance (Snapshot Fixed)")
    plt.ylabel("Packets / Slot")
    plt.xlabel("Frame")
    plt.legend()
    plt.grid(True)
    plt.savefig("result_independent_fixed.png")
    plt.show()


if __name__ == "__main__":
    main()