"""数据预处理"""

import pickle
import random

import networkx as nx
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_graph(file_name: str, model_type: str):
    """获取图结构数据并确保节点连续性"""
    edge_index = np.load(f"data/{model_type}/{file_name}/edge_index.npy")
    src = edge_index[0]
    dst = edge_index[1]
    # 获取所有唯一节点并创建映射
    all_nodes = np.unique(np.concatenate((src, dst)))
    node_mapping = {old: new for new, old in enumerate(sorted(all_nodes))}
    # 创建映射后的边列表
    mapped_edges = [(node_mapping[s], node_mapping[d]) for s, d in zip(src, dst)]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(all_nodes)))
    graph.add_edges_from(mapped_edges)
    return graph


def get_graph_stats(file_name: str, model_type: str):
    """获取图的统计信息并处理不连通图的情况"""
    graph = get_graph(file_name, model_type)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    avg_degree = sum(dict(nx.degree(graph)).values()) / num_nodes
    average_clustering = nx.average_clustering(graph)
    density = nx.density(graph)
    diameter = nx.diameter(graph) if nx.is_connected(graph) else float("inf")
    return num_nodes, num_edges, avg_degree, average_clustering, density, diameter


def get_radio(graph: nx.Graph, status: dict):
    """获取传播范围"""
    return (
        sum(state for state in status.values() if state == 1) / graph.number_of_nodes()
    )


def sir_propagation(graph: nx.Graph, seed_index: list, radio: float = 0.3):
    """SIR模型传播"""
    status = {node: 0 for node in graph.nodes()}
    for node in seed_index:
        status[node] = 1

    while get_radio(graph, status) < radio:
        new_status = status.copy()
        for node in graph.nodes():
            beta = random.uniform(0.1, 0.3)  # 动态调整感染概率
            gamma = random.uniform(0.05, 0.15)  # 动态调整恢复概率
            if status[node] == 0:
                # 计算感染概率
                infected_neighbors = sum(
                    1 for neighbor in graph.neighbors(node) if status[neighbor] == 1
                )
                if random.random() < 1 - (1 - beta) ** infected_neighbors:
                    new_status[node] = 1
            elif status[node] == 1:
                # 感染者恢复
                if random.random() < gamma:
                    new_status[node] = 2
        if new_status == status:  # 无新感染，提前终止
            break
        status = new_status

    final_state = np.zeros((graph.number_of_nodes(), 3))
    # 记录每个节点的最终状态
    for node in graph.nodes():
        if status[node] == 0:
            final_state[node, 0] = 1
        elif status[node] == 1:
            final_state[node, 1] = 1
        elif status[node] == 2:
            final_state[node, 2] = 1

    return final_state


def si_propagation(graph: nx.Graph, seed_index: list, radio: float = 0.3):
    """SI模型传播"""
    status = {node: 0 for node in graph.nodes()}
    for node in seed_index:
        status[node] = 1  # 初始感染节点

    while get_radio(graph, status) < radio:
        new_status = status.copy()
        for node in graph.nodes():
            beta = random.uniform(0.1, 0.3)  # 动态调整感染概率
            if status[node] == 0:
                # 计算感染概率
                infected_neighbors = sum(
                    1 for neighbor in graph.neighbors(node) if status[neighbor] == 1
                )
                if random.random() < 1 - (1 - beta) ** infected_neighbors:
                    new_status[node] = 1
        if new_status == status:  # 无新感染，提前终止
            break
        status = new_status

    final_state = np.zeros((graph.number_of_nodes(), 2))
    for node in graph.nodes():
        final_state[node, 0] = 1 if status[node] == 0 else 0
        final_state[node, 1] = 1 if status[node] == 1 else 0
    return final_state


def lt_propagation(graph: nx.Graph, seed_index: list, radio: float = 0.3):
    """线性阈值模型传播"""
    status = {node: 0 for node in graph.nodes()}
    for node in seed_index:
        status[node] = 1

    # 预处理边权重（归一化）
    edge_weights = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        deg = len(neighbors)
        for neighbor in neighbors:
            edge_weights[(neighbor, node)] = 1.0 / deg  # 入边权重归一化

    while get_radio(graph, status) < radio:
        new_status = status.copy()
        to_activate = []
        for node in graph.nodes():
            if status[node] == 0:
                # 计算激活邻居的总权重
                total_weight = sum(
                    edge_weights.get((neighbor, node), 0)
                    for neighbor in graph.neighbors(node)
                    if status[neighbor] == 1
                )
                if total_weight >= random.uniform(0.4, 0.6):
                    to_activate.append(node)
        # 同步更新状态
        for node in to_activate:
            new_status[node] = 1
        if new_status == status:
            break
        status = new_status

    final_state = np.zeros((graph.number_of_nodes(), 2))
    for node in graph.nodes():
        final_state[node, 0] = 1 if status[node] == 0 else 0
        final_state[node, 1] = 1 if status[node] == 1 else 0
    return final_state


def ic_propagation(graph: nx.Graph, seed_index: list, radio: float = 0.3):
    """独立级联模型传播"""
    status = {node: 0 for node in graph.nodes()}
    triggered_edges = set()  # 记录已触发的边
    for node in seed_index:
        status[node] = 1

    active_nodes = set(seed_index)
    current_ratio = len(active_nodes) / graph.number_of_nodes()

    while current_ratio < radio:
        new_active = set()
        current_active = list(active_nodes)  # 当前激活节点传播
        active_nodes.clear()

        for u in current_active:
            for v in graph.neighbors(u):
                if status[v] == 0 and (u, v) not in triggered_edges:
                    triggered_edges.add((u, v))  # 标记边已触发
                    if random.random() < random.uniform(0.05, 0.15):
                        new_active.add(v)

        # 更新状态
        for v in new_active:
            status[v] = 1
        active_nodes.update(new_active)
        current_ratio = sum(status.values()) / graph.number_of_nodes()

        if not new_active:  # 无新激活则终止
            break

    final_state = np.zeros((graph.number_of_nodes(), 2))
    for node in graph.nodes():
        final_state[node, 0] = 1 if status[node] == 0 else 0
        final_state[node, 1] = 1 if status[node] == 1 else 0
    return final_state


def run_propagation(
    graph: nx.Graph,
    seed_nodes: list,
    model_type: str = "SIR",
    target_ratio: float = 0.3,
):
    """传播模型调度器"""
    model_mapping = {
        "SIR": sir_propagation,
        "SI": si_propagation,
        "LT": lt_propagation,
        "IC": ic_propagation,
    }

    if model_type not in model_mapping:
        raise ValueError(f"Invalid propagation model: {model_type}")

    return model_mapping[model_type](graph, seed_nodes, target_ratio)


def save_static_data(
    file_name: str,
    model_type: str = "SIR",
    seed_num: int = 1,
    batch_size: int = 1000,
    target_ratio: float = 0.3,
):
    """保存静态数据，优化种子选择逻辑"""
    graph = get_graph(file_name, model_type)
    num_nodes = graph.number_of_nodes()
    # 动态设置种子数量
    if seed_num <= 0:
        seed_num = max(1, int(num_nodes * 0.01))  # 至少1个种子
    # 只选择30%的节点作为可能的种子
    candidate_nodes = random.sample(range(num_nodes), int(num_nodes * 0.3))
    # 确定状态维度
    num_states = 3 if model_type == "SIR" else 2
    state_data = np.zeros((batch_size, num_nodes, num_states))
    seed_data = np.zeros((batch_size, num_nodes))

    for i in range(batch_size):
        seed_nodes = random.sample(candidate_nodes, seed_num)
        seed_data[i, seed_nodes] = 1
        state_data[i] = run_propagation(graph, seed_nodes, model_type, target_ratio)
    # 保存数据
    np.save(f"data/{model_type}/{file_name}/state.npy", state_data)
    np.save(f"data/{model_type}/{file_name}/seed.npy", seed_data)
    print(f"Saved {model_type} data for {file_name}.")


def generate_datasets():
    """生成所有数据集"""
    datasets = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    models = ["SIR", "SI", "LT", "IC"]

    for model in models:
        for dataset in datasets:
            save_static_data(dataset, model_type=model)


def load_data(file_name: str, model_type: str):
    """加载数据并创建数据加载器"""
    state = np.load(f"./data/{model_type}/{file_name}/state.npy")
    state_tensor = torch.tensor(state, dtype=torch.float32)

    seed = np.load(f"./data/{model_type}/{file_name}/seed.npy")
    seed_tensor = torch.tensor(seed, dtype=torch.float32)

    edge_index = np.load(f"./data/{model_type}/{file_name}/edge_index.npy")
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

    num_nodes = state.shape[1]
    num_states = state.shape[2]
    num_edges = edge_index.shape[1]

    # 数据集划分
    total_size = len(state)
    train_size = int(0.8 * total_size)
    valid_size = int(0.1 * total_size)

    # 创建数据集
    train_set = TensorDataset(state_tensor[:train_size], seed_tensor[:train_size])
    valid_set = TensorDataset(
        state_tensor[train_size : train_size + valid_size],
        seed_tensor[train_size : train_size + valid_size],
    )
    test_set = TensorDataset(
        state_tensor[train_size + valid_size :], seed_tensor[train_size + valid_size :]
    )

    # 创建数据加载器
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


def calculate_stats(values: list[float]) -> tuple[float, float]:
    """计算平均值和标准差"""
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = variance**0.5
    return mean, std_dev


def resize_index(name: str):
    input_file = f"./data/{name}/edges.txt"
    cascade = f"./data/{name}/cascade.txt"
    cascadetest = f"./data/{name}/cascadetest.txt"
    cascadevalid = f"./data/{name}/cascadevalid.txt"
    # 存储所有边和节点
    edges = []
    nodes = set()
    # 读取文件并提取节点和边
    with open(input_file, "r") as f:
        for line in f:
            # 去除空白字符并分割
            u, v = map(int, line.strip().split(","))
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)
    with open(cascade, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            for node in line:
                nodes.add(int(node.split(",")[0]))
    with open(cascadetest, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            for node in line:
                nodes.add(int(node.split(",")[0]))
    with open(cascadevalid, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            for node in line:
                nodes.add(int(node.split(",")[0]))
    # 对节点进行排序并创建映射字典
    sorted_nodes = sorted(nodes)
    index_dict = {node: idx for idx, node in enumerate(sorted_nodes)}
    # 将原始边数据转换为新的节点编号
    new_edges = []
    for u, v in edges:
        new_u = index_dict[u]
        new_v = index_dict[v]
        new_edges.append((new_u, new_v))
    # 转换为(2, num_edges)形状的数组
    edge_index = np.array(new_edges).T
    # 保存结果
    np.save(f"./data/{name}/edge_index.npy", edge_index)
    with open(f"./data/{name}/index_dict.pkl", "wb") as f:
        pickle.dump(index_dict, f)
    print(f"处理完成：")
    print(f"原始节点数量：{len(nodes)}")
    print(f"边数量：{len(edges)}")


def load_cascade_data(path: str, num_nodes: int, stype: str):
    name = path.split("/")[0]
    with open(f"./data/{name}/index_dict.pkl", "rb") as f:
        index_dict: dict[int, int] = pickle.load(f)
    with open(f"./data/{path}", "r") as f:
        information = f.readlines()
        length = len(information)
        state = np.zeros((length, num_nodes, 2))
        state[:, :, 0] = 1
        seed = np.zeros((length, num_nodes))
        for i, info in enumerate(information):
            info = info.strip().split(" ")
            sources = int(len(info) * 0.05)
            sources = sources if sources >= 1 else 1
            nodes = [index_dict[int(node_.split(",")[0])] for node_ in info]
            state[i, nodes, 0] = 0
            state[i, nodes, 1] = 1
            seed[i, nodes[:sources]] = 1
    np.save(f"./data/{name}/{stype}_state.npy", state)
    np.save(f"./data/{name}/{stype}_seed.npy", seed)


def get_true_cascade_dataset(name: str):
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

    train_state = torch.tensor(train_state, dtype=torch.float32)
    train_seed = torch.tensor(train_seed, dtype=torch.float32)
    valid_state = torch.tensor(valid_state, dtype=torch.float32)
    valid_seed = torch.tensor(valid_seed, dtype=torch.float32)
    test_state = torch.tensor(test_state, dtype=torch.float32)
    test_seed = torch.tensor(test_seed, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    train_dataset = TensorDataset(train_state, train_seed)
    valid_dataset = TensorDataset(valid_state, valid_seed)
    test_dataset = TensorDataset(test_state, test_seed)

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        num_nodes,
        num_edges,
        2,  # num_features
    )


# if __name__ == "__main__":
#     my_list = [("cascade", "train"), ("cascadetest", "test"), ("cascadevalid", "valid")]
#     # generate_datasets()
#     # print("All datasets generated successfully.")
#     # resize_index("twitter")
#     for i, j in my_list:
#         load_cascade_data(f"twitter/{i}.txt", 12627, j)

# douban
# 原始节点数量：25348
# 边数量：758310
# twitter
# 原始节点数量：12627
# 边数量：619262
