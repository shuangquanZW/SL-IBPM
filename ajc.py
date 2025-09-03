import os

# 设置环境变量避免内存泄漏
os.environ["OMP_NUM_THREADS"] = "1"

import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def _build_adj(edge_index: np.ndarray, num_nodes: int) -> csr_matrix:
    """向量化构建对称邻接矩阵"""
    row, col = edge_index
    data = np.ones(len(row), dtype=float)
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj.maximum(adj.T)  # 确保对称性


def _reconstruct_path(predecessors: np.ndarray, start: int, end: int) -> np.ndarray:
    """从前驱节点矩阵重建最短路径"""
    path = []
    current = end
    while current != start:
        path.append(current)
        current = predecessors[current]
        if current == -9999:  # 无路径时退出
            return np.array([])
    path.append(start)
    return np.array(path[::-1])  # 反转得到从起点到终点的路径


def _ensure_connectivity(
    adj: csr_matrix, cand: np.ndarray, obs_infected: np.ndarray
) -> np.ndarray:
    """确保候选子图连通（论文Algorithm 1）"""
    # 构建候选节点诱导子图
    sub_adj = adj[np.ix_(cand, cand)]

    # 检查连通性
    n_components, labels = connected_components(sub_adj, directed=False)
    if n_components == 1:
        return cand

    # 处理不连通情况：添加代表节点间的最短路径
    obs_idx = np.where(obs_infected[cand])[0]
    if len(obs_idx) == 0:
        return cand  # 无观测节点无法处理

    # 选择每个分量的一个代表节点
    comp_reps = []
    for i in range(n_components):
        comp_nodes = np.where(labels == i)[0]
        comp_obs = np.intersect1d(comp_nodes, obs_idx)
        # 优先选择观测节点作为代表
        comp_reps.append(comp_obs[0] if len(comp_obs) > 0 else comp_nodes[0])

    # 添加代表节点间的最短路径（映射到原始节点ID）
    added_nodes = set()
    base_node = cand[comp_reps[0]]  # 基准节点（原始ID）
    # 计算基准节点到所有其他节点的最短路径
    _, predecessors = shortest_path(adj, indices=base_node, return_predecessors=True)

    for i in range(1, len(comp_reps)):
        target_node = cand[comp_reps[i]]  # 目标节点（原始ID）
        path = _reconstruct_path(predecessors, base_node, target_node)  # type: ignore
        if len(path) > 0:
            added_nodes.update(path)

    # 合并新增节点并去重
    all_nodes = np.concatenate([cand, np.array(list(added_nodes))])
    return np.unique(all_nodes)


def _candidate_selection(
    adj: csr_matrix, obs_infected: np.ndarray, Y: int
) -> np.ndarray:
    """向量化候选节点选择（论文步骤1）"""
    infected_mask = obs_infected.astype(bool)
    # 计算每个节点的感染邻居数
    neighbor_infected = adj.dot(infected_mask.astype(float)).astype(int)
    # 条件1: 感染节点自动加入候选集
    cond1 = infected_mask
    # 条件2: 非感染节点但有≥Y个感染邻居
    cond2 = (~infected_mask) & (neighbor_infected >= Y)  # type: ignore
    return np.where(cond1 | cond2)[0]


def _compute_distances(
    adj: csr_matrix, cand: np.ndarray, obs_infected: np.ndarray
) -> np.ndarray:
    """高效计算候选节点到观测感染节点的距离（论文步骤2前置）"""
    # 获取候选子图中的观测节点索引（相对候选集的索引）
    obs_in_subgraph = np.where(obs_infected[cand])[0]
    if len(obs_in_subgraph) == 0:
        return np.zeros((len(cand), 0))  # 空距离矩阵

    # 计算子图中观测节点到所有候选节点的最短路径
    sub_adj = adj[np.ix_(cand, cand)]
    dist_matrix = shortest_path(
        sub_adj, directed=False, indices=obs_in_subgraph, unweighted=True  # 跳数距离
    )
    return dist_matrix.T  # 转置为 [候选节点 × 观测节点]


def ajc(
    edge_index: np.ndarray, state: np.ndarray, seed: np.ndarray, Y: int = 2
) -> np.ndarray:
    """
    AJC算法实现（论文3.2节）

    参数:
        edge_index: 边索引矩阵，形状为[2, E]
        state: 节点状态矩阵，形状为[B, N, 1]，其中B为批次大小，N为节点数
        seed: 真实源节点矩阵，用于确定源数量，形状为[B, N]
        Y: 候选节点选择阈值（论文Algorithm 1中的Y）

    返回:
        预测的源节点矩阵，形状与seed一致
    """
    # 预构建全局邻接矩阵
    num_nodes = state.shape[1]
    A = _build_adj(edge_index, num_nodes)

    batch_size = state.shape[0]
    sources = np.zeros_like(seed)
    # 提取观测到的感染节点（I/R状态）
    obs_mask = (state[..., 0] < 0.5).astype(int)  # 假设<0.5代表感染/恢复状态

    for b in range(batch_size):
        obs_infected = obs_mask[b]
        # 步骤1: 候选节点选择
        cand = _candidate_selection(A, obs_infected, Y=Y)

        # 确保候选子图连通（论文Algorithm 1）
        cand = _ensure_connectivity(A, cand, obs_infected)
        # 确定当前批次的源数量
        m = int(np.count_nonzero(seed[b]))

        # 处理候选节点不足的情况
        if len(cand) < m:
            # 补充高感染度节点
            obs_degrees = A.dot(obs_infected)
            top_obs = np.argsort(-obs_degrees)[: max(0, m - len(cand))]
            cand = np.unique(np.concatenate([cand, top_obs]))

        # 步骤2: 计算距离矩阵
        D = _compute_distances(A, cand, obs_infected)
        if D.shape[1] == 0:  # 无观测节点时无法推断
            continue

        # 步骤3: K-Means聚类（近似Jordan覆盖）
        kmeans = KMeans(
            n_clusters=min(m, len(cand)),  # 聚类数=源数量
            n_init=10,
            max_iter=50,
            random_state=42,
        ).fit(D)

        # 选择每个聚类的中心节点（最小感染离心率）
        source_idx = []
        for i in range(kmeans.n_clusters):  # type: ignore
            cluster_mask = kmeans.labels_ == i
            if not np.any(cluster_mask):
                continue
            # 计算聚类内每个节点到观测节点的最大距离（离心率）
            cluster_dist = D[cluster_mask]
            eccentricity = np.max(cluster_dist, axis=1)
            # 选择离心率最小的节点作为中心
            min_idx = np.argmin(eccentricity)
            source_idx.append(np.where(cluster_mask)[0][min_idx])

        # 映射回原始节点ID并标记
        selected_nodes = cand[source_idx]
        sources[b, selected_nodes] = 1

    return sources


def evaluate(pred: np.ndarray, true: np.ndarray):
    # 展平为二维数组，方便 sklearn 计算
    y_true = true.reshape(-1)
    y_score = pred.reshape(-1)

    roc = roc_auc_score(y_true, y_score)
    # 计算二值预测
    y_pred = np.where(y_score > 0.5, 1, 0)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return roc, precision, recall, f1


def load_data(file_name: str, type: str):
    state = np.load(f"./data/{type}/{file_name}/state.npy")
    seed = np.load(f"./data/{type}/{file_name}/seed.npy")
    edge_index = np.load(f"./data/{type}/{file_name}/edge_index.npy")

    train_size = int(0.8 * len(state))
    valid_size = int(0.1 * len(state))
    y_test = seed[train_size + valid_size :]
    x_test = state[train_size + valid_size :]

    return x_test, y_test, edge_index


def predict_ajc(file_name: str, type_: str, Y: int):
    x_test, y_test, edge_index = load_data(file_name, type_)
    # 调用 AJC 算法
    sources = ajc(edge_index, x_test, y_test, Y)
    # 评估
    roc, precision, recall, f1 = evaluate(sources, y_test)
    return roc, precision, recall, f1


if __name__ == "__main__":
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]
    result = {}
    Y = 2

    # 创建结果文件
    with open("result/AJC.txt", "w") as f:
        f.write("AJC Algorithm Results\n")
        f.write("-" * 70 + "\n")

        for type_ in type_list:
            for name in file_list:
                start = time.time()
                roc, precision, recall, f1 = predict_ajc(
                    name, type_, Y
                )  # 修改为返回4个指标
                end = time.time()
                f.write(f"{name} ({type_}): ")
                f.write(
                    f"{name} ({type_}): "
                    f"Time: {end - start:.4f}s, "
                    f"AUROC: {roc:.4f}, "
                    f"Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, "
                    f"F1: {f1:.4f}\n"
                )
                f.write("-" * 50 + "\n")
