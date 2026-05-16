"""公共工具函数与数据加载"""

import os
import csv
import pickle
import random
import numpy as np
import torch
import networkx as nx
from torch.utils.data import TensorDataset, DataLoader

SEEDS = [0, 1, 2, 3, 4]

# ========================= 网络拓扑缓存 =========================

_graph_stats_cache = {}


def _compute_graph_stats(file_name: str, model_type: str):
    """计算并缓存图的拓扑统计信息"""
    key = (file_name, model_type)
    if key in _graph_stats_cache:
        return _graph_stats_cache[key]

    edge_index = np.load(f"data/{model_type}/{file_name}/edge_index.npy")
    src = edge_index[0]
    dst = edge_index[1]

    all_nodes = np.unique(np.concatenate((src, dst)))
    node_mapping = {old: new for new, old in enumerate(sorted(all_nodes))}
    mapped_edges = [(node_mapping[s], node_mapping[d]) for s, d in zip(src, dst)]

    graph = nx.Graph()
    graph.add_nodes_from(range(len(all_nodes)))
    graph.add_edges_from(mapped_edges)

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_degree = 2.0 * num_edges / num_nodes if num_nodes > 0 else 0.0
    avg_clustering = nx.average_clustering(graph)
    density = nx.density(graph)
    diameter = nx.diameter(graph) if nx.is_connected(graph) else float("inf")

    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "avg_clustering": avg_clustering,
        "density": density,
        "diameter": diameter,
    }

    _graph_stats_cache[key] = stats
    return stats


def get_graph_stats(file_name: str, model_type: str):
    """获取图的统计信息"""
    return _compute_graph_stats(file_name, model_type)


# ========================= 定制传播参数表 =========================
# 适配设置：
#   - 源节点比例：10%
#   - 传播目标比例：30%
#
# 因为源节点已经占 10%，所以传播参数不宜过强。
# 目标是让传播额外扩散约 20% 节点，避免传播过弱或过饱和。

_PROPAGATION_PARAMS_CUSTOM = {
    # karate: 小图，传播参数可稍高
    ("karate", "SIR"): {"beta": (0.10, 0.25), "gamma": (0.05, 0.15)},
    ("karate", "SI"): {"beta": (0.10, 0.25)},
    ("karate", "LT"): {"threshold": (0.20, 0.35)},
    ("karate", "IC"): {"prob": (0.10, 0.25)},
    # jazz: 稠密图，beta / prob 不能太大
    ("jazz", "SIR"): {"beta": (0.005, 0.015), "gamma": (0.05, 0.12)},
    ("jazz", "SI"): {"beta": (0.008, 0.020)},
    ("jazz", "LT"): {"threshold": (0.15, 0.30)},
    ("jazz", "IC"): {"prob": (0.015, 0.040)},
    # net_science: 稀疏且可能不连通，参数中等
    ("net_science", "SIR"): {"beta": (0.08, 0.16), "gamma": (0.05, 0.15)},
    ("net_science", "SI"): {"beta": (0.08, 0.18)},
    ("net_science", "LT"): {"threshold": (0.18, 0.32)},
    ("net_science", "IC"): {"prob": (0.08, 0.20)},
    # cora_ml: 中大图，参数不要太弱
    ("cora_ml", "SIR"): {"beta": (0.05, 0.12), "gamma": (0.04, 0.12)},
    ("cora_ml", "SI"): {"beta": (0.06, 0.15)},
    ("cora_ml", "LT"): {"threshold": (0.15, 0.30)},
    ("cora_ml", "IC"): {"prob": (0.06, 0.18)},
    # power_grid: 很稀疏，传播参数需要偏高
    ("power_grid", "SIR"): {"beta": (0.20, 0.40), "gamma": (0.05, 0.15)},
    ("power_grid", "SI"): {"beta": (0.25, 0.50)},
    ("power_grid", "LT"): {"threshold": (0.25, 0.45)},
    ("power_grid", "IC"): {"prob": (0.25, 0.50)},
    # lastFM: 大图，参数中等偏强
    ("lastFM", "SIR"): {"beta": (0.05, 0.12), "gamma": (0.04, 0.12)},
    ("lastFM", "SI"): {"beta": (0.08, 0.20)},
    ("lastFM", "LT"): {"threshold": (0.12, 0.28)},
    ("lastFM", "IC"): {"prob": (0.06, 0.18)},
}


# 静态 fallback 参数
_PROPAGATION_PARAMS_FALLBACK = {
    "SIR": {"beta": (0.1, 0.3), "gamma": (0.05, 0.15)},
    "SI": {"beta": (0.1, 0.3)},
    "LT": {"threshold": (0.4, 0.6)},
    "IC": {"prob": (0.05, 0.15)},
}


def get_propagation_params(file_name: str, model_type: str):
    """返回指定数据集和传播模型的参数范围"""
    key = (file_name, model_type)

    if key in _PROPAGATION_PARAMS_CUSTOM:
        return _PROPAGATION_PARAMS_CUSTOM[key]

    # 若未命中定制表，则回退到动态计算
    try:
        stats = _compute_graph_stats(file_name, model_type)
        params_dict = _compute_propagation_params_from_stats(stats)
        return params_dict.get(
            model_type,
            _PROPAGATION_PARAMS_FALLBACK.get(model_type, {}),
        )
    except Exception:
        return _PROPAGATION_PARAMS_FALLBACK.get(model_type, {})


def _compute_propagation_params_from_stats(stats: dict):
    """基于图统计信息动态计算传播参数，作为 fallback 保留"""
    avg_deg = stats["avg_degree"]
    num_nodes = stats["num_nodes"]

    if avg_deg <= 0:
        return _PROPAGATION_PARAMS_FALLBACK

    # beta
    if avg_deg <= 5:
        C = 0.7
        beta_mid = C / avg_deg
    elif avg_deg <= 15:
        C = 0.9
        beta_mid = C / avg_deg
    else:
        C = 1.5
        beta_mid = C / (avg_deg**0.65)

    beta_mid = min(0.28, beta_mid)
    beta_low = max(0.05, round(beta_mid * 0.7, 4))
    beta_high = min(0.45, round(beta_mid * 1.4, 4))

    # gamma
    if num_nodes < 500:
        gamma_mid = beta_mid * 0.35
    elif num_nodes < 5000:
        gamma_mid = beta_mid * 0.45
    else:
        gamma_mid = beta_mid * 0.55

    gamma_low = max(0.02, round(gamma_mid * 0.5, 4))
    gamma_high = min(0.25, round(gamma_mid * 1.5, 4))

    # LT threshold
    if avg_deg < 3:
        thresh_mid = 0.22
    elif avg_deg < 6:
        thresh_mid = min(0.75, round(0.25 + avg_deg * 0.025, 4))
    else:
        thresh_mid = min(0.75, round(0.28 + avg_deg * 0.020, 4))

    thresh_low = max(0.12, round(thresh_mid - 0.12, 4))
    thresh_high = min(0.85, round(thresh_mid + 0.12, 4))

    # IC prob
    if avg_deg <= 5:
        C_ic = 0.7
        prob_mid = C_ic / avg_deg
    elif avg_deg <= 15:
        C_ic = 1.0
        prob_mid = C_ic / avg_deg
    else:
        C_ic = 1.8
        prob_mid = C_ic / (avg_deg**0.60)

    prob_mid = min(0.28, prob_mid)
    prob_low = max(0.02, round(prob_mid * 0.6, 4))
    prob_high = min(0.40, round(prob_mid * 1.4, 4))

    return {
        "SIR": {"beta": (beta_low, beta_high), "gamma": (gamma_low, gamma_high)},
        "SI": {"beta": (beta_low, beta_high)},
        "LT": {"threshold": (thresh_low, thresh_high)},
        "IC": {"prob": (prob_low, prob_high)},
    }


# ========================= 图加载 =========================


def get_graph(file_name: str, model_type: str):
    """获取图结构数据并确保节点编号连续"""
    edge_index = np.load(f"data/{model_type}/{file_name}/edge_index.npy")
    src = edge_index[0]
    dst = edge_index[1]

    all_nodes = np.unique(np.concatenate((src, dst)))
    node_mapping = {old: new for new, old in enumerate(sorted(all_nodes))}
    mapped_edges = [(node_mapping[s], node_mapping[d]) for s, d in zip(src, dst)]

    graph = nx.Graph()
    graph.add_nodes_from(range(len(all_nodes)))
    graph.add_edges_from(mapped_edges)

    return graph


def get_radio(graph: nx.Graph, status: dict, model_type: str = "SI"):
    """
    获取传播范围。

    对于 SIR：
        统计感染过的节点比例，即 I + R。
        status:
            0 = susceptible
            1 = infected
            2 = recovered

    对于 SI / LT / IC：
        status == 1 表示已感染或已激活。
    """
    if graph.number_of_nodes() == 0:
        return 0.0

    if model_type == "SIR":
        infected_or_recovered = sum(1 for state in status.values() if state in [1, 2])
        return infected_or_recovered / graph.number_of_nodes()

    infected = sum(1 for state in status.values() if state == 1)
    return infected / graph.number_of_nodes()


# ========================= 传播模型 =========================


def sir_propagation(
    graph: nx.Graph,
    seed_index: list,
    radio: float = 0.3,
    beta_range=(0.1, 0.3),
    gamma_range=(0.05, 0.15),
):
    """
    SIR 模型传播。

    传播停止条件：
        I + R 比例达到 radio。
    """
    status = {node: 0 for node in graph.nodes()}

    for node in seed_index:
        if node in status:
            status[node] = 1

    while get_radio(graph, status, model_type="SIR") < radio:
        new_status = status.copy()

        for node in graph.nodes():
            beta = random.uniform(*beta_range)
            gamma = random.uniform(*gamma_range)

            if status[node] == 0:
                infected_neighbors = sum(
                    1 for neighbor in graph.neighbors(node) if status[neighbor] == 1
                )

                if infected_neighbors > 0:
                    infect_prob = 1 - (1 - beta) ** infected_neighbors
                    if random.random() < infect_prob:
                        new_status[node] = 1

            elif status[node] == 1:
                if random.random() < gamma:
                    new_status[node] = 2

        if new_status == status:
            break

        status = new_status

    final_state = np.zeros((graph.number_of_nodes(), 3), dtype=np.float32)

    for node in graph.nodes():
        if status[node] == 0:
            final_state[node, 0] = 1.0
        elif status[node] == 1:
            final_state[node, 1] = 1.0
        elif status[node] == 2:
            final_state[node, 2] = 1.0

    return final_state


def si_propagation(
    graph: nx.Graph,
    seed_index: list,
    radio: float = 0.3,
    beta_range=(0.1, 0.3),
):
    """
    SI 模型传播。

    传播停止条件：
        感染节点比例达到 radio。
    """
    status = {node: 0 for node in graph.nodes()}

    for node in seed_index:
        if node in status:
            status[node] = 1

    while get_radio(graph, status, model_type="SI") < radio:
        new_status = status.copy()

        for node in graph.nodes():
            beta = random.uniform(*beta_range)

            if status[node] == 0:
                infected_neighbors = sum(
                    1 for neighbor in graph.neighbors(node) if status[neighbor] == 1
                )

                if infected_neighbors > 0:
                    infect_prob = 1 - (1 - beta) ** infected_neighbors
                    if random.random() < infect_prob:
                        new_status[node] = 1

        if new_status == status:
            break

        status = new_status

    final_state = np.zeros((graph.number_of_nodes(), 2), dtype=np.float32)

    for node in graph.nodes():
        final_state[node, 0] = 1.0 if status[node] == 0 else 0.0
        final_state[node, 1] = 1.0 if status[node] == 1 else 0.0

    return final_state


def lt_propagation(
    graph: nx.Graph,
    seed_index: list,
    radio: float = 0.3,
    threshold_range=(0.4, 0.6),
):
    """
    LT 线性阈值模型传播。

    传播停止条件：
        激活节点比例达到 radio。
    """
    status = {node: 0 for node in graph.nodes()}

    for node in seed_index:
        if node in status:
            status[node] = 1

    edge_weights = {}

    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        deg = len(neighbors)

        if deg == 0:
            continue

        for neighbor in neighbors:
            edge_weights[(neighbor, node)] = 1.0 / deg

    while get_radio(graph, status, model_type="LT") < radio:
        new_status = status.copy()
        to_activate = []

        for node in graph.nodes():
            if status[node] == 0:
                total_weight = sum(
                    edge_weights.get((neighbor, node), 0.0)
                    for neighbor in graph.neighbors(node)
                    if status[neighbor] == 1
                )

                threshold = random.uniform(*threshold_range)

                if total_weight >= threshold:
                    to_activate.append(node)

        for node in to_activate:
            new_status[node] = 1

        if new_status == status:
            break

        status = new_status

    final_state = np.zeros((graph.number_of_nodes(), 2), dtype=np.float32)

    for node in graph.nodes():
        final_state[node, 0] = 1.0 if status[node] == 0 else 0.0
        final_state[node, 1] = 1.0 if status[node] == 1 else 0.0

    return final_state


def ic_propagation(
    graph: nx.Graph,
    seed_index: list,
    radio: float = 0.3,
    prob_range=(0.05, 0.15),
):
    """
    IC 独立级联模型传播。

    传播停止条件：
        激活节点比例达到 radio。
    """
    status = {node: 0 for node in graph.nodes()}
    triggered_edges = set()

    for node in seed_index:
        if node in status:
            status[node] = 1

    active_nodes = set(seed_index)
    current_ratio = get_radio(graph, status, model_type="IC")

    while current_ratio < radio:
        new_active = set()
        current_active = list(active_nodes)
        active_nodes.clear()

        for u in current_active:
            if u not in graph:
                continue

            for v in graph.neighbors(u):
                if status[v] == 0 and (u, v) not in triggered_edges:
                    triggered_edges.add((u, v))

                    prob = random.uniform(*prob_range)
                    if random.random() < prob:
                        new_active.add(v)

        for v in new_active:
            status[v] = 1

        active_nodes.update(new_active)
        current_ratio = get_radio(graph, status, model_type="IC")

        if not new_active:
            break

    final_state = np.zeros((graph.number_of_nodes(), 2), dtype=np.float32)

    for node in graph.nodes():
        final_state[node, 0] = 1.0 if status[node] == 0 else 0.0
        final_state[node, 1] = 1.0 if status[node] == 1 else 0.0

    return final_state


def run_propagation(
    graph: nx.Graph,
    seed_nodes: list,
    model_type: str = "SIR",
    target_ratio: float = 0.3,
    file_name: str | None = None,
):
    """传播模型调度器"""
    params = get_propagation_params(file_name, model_type) if file_name else {}

    if model_type == "SIR":
        return sir_propagation(
            graph,
            seed_nodes,
            radio=target_ratio,
            beta_range=params.get("beta", (0.1, 0.3)),
            gamma_range=params.get("gamma", (0.05, 0.15)),
        )

    elif model_type == "SI":
        return si_propagation(
            graph,
            seed_nodes,
            radio=target_ratio,
            beta_range=params.get("beta", (0.1, 0.3)),
        )

    elif model_type == "LT":
        return lt_propagation(
            graph,
            seed_nodes,
            radio=target_ratio,
            threshold_range=params.get("threshold", (0.4, 0.6)),
        )

    elif model_type == "IC":
        return ic_propagation(
            graph,
            seed_nodes,
            radio=target_ratio,
            prob_range=params.get("prob", (0.05, 0.15)),
        )

    else:
        raise ValueError(f"Invalid propagation model: {model_type}")


# ========================= 数据生成与保存 =========================


def save_static_data(
    file_name: str,
    model_type: str = "SIR",
    seed_num: int = 0,
    batch_size: int = 1000,
    target_ratio: float = 0.3,
):
    """
    保存静态数据。

    当前标准设置：
        1. 源节点数量 = 全部节点的 10%
        2. 源节点位置 = 从全部节点中随机选择
        3. 传播目标 = 30% 节点被感染或激活
        4. SIR 中传播比例按 I + R 统计

    参数：
        file_name:
            数据集名称，例如 karate, jazz, net_science, cora_ml, power_grid, lastFM。

        model_type:
            传播模型类型，SIR / SI / LT / IC。

        seed_num:
            若 seed_num <= 0，则自动设置为 int(num_nodes * 0.1)。

        batch_size:
            生成样本数量。

        target_ratio:
            传播目标比例，默认 0.3。
    """
    graph = get_graph(file_name, model_type)
    num_nodes = graph.number_of_nodes()

    if num_nodes <= 0:
        raise ValueError(f"Graph {file_name}-{model_type} has no nodes.")

    # 选择全部节点的 10% 作为源节点
    if seed_num <= 0:
        seed_num = max(1, int(num_nodes * 0.1))

    # 避免极端情况下 seed_num 超过节点数
    seed_num = min(seed_num, num_nodes)

    num_states = 3 if model_type == "SIR" else 2

    state_data = np.zeros((batch_size, num_nodes, num_states), dtype=np.float32)
    seed_data = np.zeros((batch_size, num_nodes), dtype=np.float32)

    all_nodes = list(range(num_nodes))

    for i in range(batch_size):
        # 从全部节点中随机选择 10% 作为源节点
        seed_nodes = random.sample(all_nodes, seed_num)
        seed_data[i, seed_nodes] = 1.0

        state_data[i] = run_propagation(
            graph=graph,
            seed_nodes=seed_nodes,
            model_type=model_type,
            target_ratio=target_ratio,
            file_name=file_name,
        )

        if (i + 1) % 100 == 0:
            print(f"[{file_name}-{model_type}] generated {i + 1}/{batch_size} samples")

    os.makedirs(f"data/{model_type}/{file_name}/", exist_ok=True)

    np.save(f"data/{model_type}/{file_name}/state.npy", state_data)
    np.save(f"data/{model_type}/{file_name}/seed.npy", seed_data)

    print(
        f"Saved {model_type} data for {file_name}. "
        f"num_nodes={num_nodes}, "
        f"seed_num={seed_num}, "
        f"seed_ratio={seed_num / num_nodes:.4f}, "
        f"target_ratio={target_ratio}, "
        f"batch_size={batch_size}"
    )


def generate_datasets(
    batch_size: int = 1000,
    target_ratio: float = 0.3,
):
    """
    生成所有静态传播数据集。

    默认：
        源节点比例 10%
        感染目标比例 30%
    """
    datasets = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    models = ["SIR", "SI", "LT", "IC"]

    for model in models:
        for dataset in datasets:
            save_static_data(
                file_name=dataset,
                model_type=model,
                seed_num=0,
                batch_size=batch_size,
                target_ratio=target_ratio,
            )


# ========================= 结果统计与保存 =========================


def compute_stats(values):
    """计算均值和样本标准差"""
    values = np.asarray(values)

    if len(values) <= 1:
        return float(np.mean(values)), 0.0

    return float(np.mean(values)), float(np.std(values, ddof=1))


def save_csv(results, filepath):
    """保存结果为 CSV 文件，包含 AUC、Precision、Recall、F1 的均值与标准差"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "Dataset",
                "Type",
                "AUC_mean",
                "AUC_std",
                "Precision_mean",
                "Precision_std",
                "Recall_mean",
                "Recall_std",
                "F1_mean",
                "F1_std",
            ]
        )

        for (name, type_), m in results.items():
            writer.writerow(
                [
                    name,
                    type_,
                    f"{m['auc_mean']:.4f}",
                    f"{m['auc_std']:.4f}",
                    f"{m['pre_mean']:.4f}",
                    f"{m['pre_std']:.4f}",
                    f"{m['rec_mean']:.4f}",
                    f"{m['rec_std']:.4f}",
                    f"{m['f1_mean']:.4f}",
                    f"{m['f1_std']:.4f}",
                ]
            )


# ========================= 数据加载器（支持按 seed 随机划分） =========================


def load_data(file_name: str, model_type: str, seed: int = 0):
    """
    加载静态传播数据，并创建 DataLoader。

    划分比例：
        train : valid : test = 8 : 1 : 1

    支持按 seed 随机划分。
    """
    state = np.load(f"./data/{model_type}/{file_name}/state.npy")
    seed_labels = np.load(f"./data/{model_type}/{file_name}/seed.npy")
    edge_index = np.load(f"./data/{model_type}/{file_name}/edge_index.npy")

    state_tensor = torch.tensor(state, dtype=torch.float32)
    seed_tensor = torch.tensor(seed_labels, dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

    num_nodes = state.shape[1]
    num_states = state.shape[2]
    num_edges = edge_index.shape[1]

    total_size = len(state)
    train_size = int(0.8 * total_size)
    valid_size = int(0.1 * total_size)

    rng = np.random.RandomState(seed)
    indices = np.arange(total_size)
    rng.shuffle(indices)

    train_idx = indices[:train_size]
    valid_idx = indices[train_size : train_size + valid_size]
    test_idx = indices[train_size + valid_size :]

    train_set = TensorDataset(state_tensor[train_idx], seed_tensor[train_idx])
    valid_set = TensorDataset(state_tensor[valid_idx], seed_tensor[valid_idx])
    test_set = TensorDataset(state_tensor[test_idx], seed_tensor[test_idx])

    train_loader = DataLoader(train_set, batch_size=12, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

    return (
        train_loader,
        valid_loader,
        test_loader,
        edge_index_tensor,
        num_nodes,
        num_edges,
        num_states,
    )


# ========================= 真实级联数据处理 =========================


def resize_index(name: str):
    """
    对真实级联数据重新编号，使节点编号从 0 开始连续。

    输入文件：
        ./data/{name}/edges.txt
        ./data/{name}/cascade.txt
        ./data/{name}/cascadetest.txt
        ./data/{name}/cascadevalid.txt

    输出文件：
        ./data/{name}/edge_index.npy
        ./data/{name}/index_dict.pkl
    """
    input_file = f"./data/{name}/edges.txt"
    cascade = f"./data/{name}/cascade.txt"
    cascadetest = f"./data/{name}/cascadetest.txt"
    cascadevalid = f"./data/{name}/cascadevalid.txt"

    edges = []
    nodes = set()

    with open(input_file, "r") as f:
        for line in f:
            u, v = map(int, line.strip().split(","))
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)

    with open(cascade, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            for node in line:
                if node:
                    nodes.add(int(node.split(",")[0]))

    with open(cascadetest, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            for node in line:
                if node:
                    nodes.add(int(node.split(",")[0]))

    with open(cascadevalid, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            for node in line:
                if node:
                    nodes.add(int(node.split(",")[0]))

    sorted_nodes = sorted(nodes)
    index_dict = {node: idx for idx, node in enumerate(sorted_nodes)}

    new_edges = []

    for u, v in edges:
        new_u = index_dict[u]
        new_v = index_dict[v]
        new_edges.append((new_u, new_v))

    edge_index = np.array(new_edges).T

    np.save(f"./data/{name}/edge_index.npy", edge_index)

    with open(f"./data/{name}/index_dict.pkl", "wb") as f:
        pickle.dump(index_dict, f)

    print("处理完成：")
    print(f"原始节点数量：{len(nodes)}")
    print(f"边数量：{len(edges)}")


def load_cascade_data(path: str, num_nodes: int, stype: str):
    """
    加载真实级联数据并保存为 state/seed numpy 文件。

    path 示例：
        ./data/digg/cascade.txt
        ./data/digg/cascadetest.txt
        ./data/digg/cascadevalid.txt

    stype 示例：
        train / test / valid
    """
    name = path.split("/")[1]

    with open(f"./data/{name}/index_dict.pkl", "rb") as f:
        index_dict: dict[int, int] = pickle.load(f)

    with open(path, "r") as f:
        information = f.readlines()
        length = len(information)

        state = np.zeros((length, num_nodes, 2), dtype=np.float32)
        state[:, :, 0] = 1.0

        seed = np.zeros((length, num_nodes), dtype=np.float32)

        for i, info in enumerate(information):
            info = info.strip().split(" ")

            # 默认将前 10% 传播节点作为源节点
            sources = int(len(info) * 0.1) + 1

            nodes = [index_dict[int(node_.split(",")[0])] for node_ in info if node_]

            state[i, nodes, 0] = 0.0
            state[i, nodes, 1] = 1.0

            seed[i, nodes[:sources]] = 1.0

    np.save(f"./data/{name}/{stype}_state.npy", state)
    np.save(f"./data/{name}/{stype}_seed.npy", seed)


def get_true_cascade_dataset(name: str):
    """
    加载真实级联数据集。

    该函数使用预划分数据：
        train_state.npy
        train_seed.npy
        valid_state.npy
        valid_seed.npy
        test_state.npy
        test_seed.npy
        edge_index.npy
    """
    file_path = f"./data/{name}/"

    train_state = np.load(file_path + "train_state.npy")
    train_seed = np.load(file_path + "train_seed.npy")

    valid_state = np.load(file_path + "valid_state.npy")
    valid_seed = np.load(file_path + "valid_seed.npy")

    test_state = np.load(file_path + "test_state.npy")
    test_seed = np.load(file_path + "test_seed.npy")

    edge_index = np.load(file_path + "edge_index.npy")

    num_nodes = train_state.shape[1]
    num_edges = edge_index.shape[1]
    num_states = train_state.shape[2]

    train_state = torch.tensor(train_state, dtype=torch.float32)
    train_seed = torch.tensor(train_seed, dtype=torch.float32)

    valid_state = torch.tensor(valid_state, dtype=torch.float32)
    valid_seed = torch.tensor(valid_seed, dtype=torch.float32)

    test_state = torch.tensor(test_state, dtype=torch.float32)
    test_seed = torch.tensor(test_seed, dtype=torch.float32)

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

    train_set = TensorDataset(train_state, train_seed)
    valid_set = TensorDataset(valid_state, valid_seed)
    test_set = TensorDataset(test_state, test_seed)

    train_loader = DataLoader(train_set, batch_size=12, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

    return (
        train_loader,
        valid_loader,
        test_loader,
        edge_index_tensor,
        num_nodes,
        num_edges,
        num_states,
    )


if __name__ == "__main__":
    generate_datasets()
