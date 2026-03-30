# 在 main.py 中找到 estimate_optimal_throughput 替换：

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

            # 【核心修改】基于布尔距离的冲突判断
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