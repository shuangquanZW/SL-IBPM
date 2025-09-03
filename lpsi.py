import time
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import inv


def lpsi(
    edge_index: np.ndarray, state: np.ndarray, seed: np.ndarray, alpha: float
) -> np.ndarray:
    """
    收敛版 LPSI：edge_index 输入

    参数
    ----
    edge_index : np.ndarray, shape=(2, num_edges)
        COO 格式的无向边列表
    state : np.ndarray, shape=(batch_size, num_nodes, num_states)
        最终快照
    seed : np.ndarray, shape=(batch_size, num_nodes)
        初始源指示矩阵（占位，实际输出会覆盖）
    alpha : float, default=0.5
        传播参数

    返回
    ----
    sources : np.ndarray, shape=(batch_size, num_nodes)
        检测到的源节点指示矩阵
    """
    batch_size, num_nodes, _ = state.shape
    Y = np.where(state[..., 0] > 0.5, -1.0, 1.0)  # 感染状态

    # 1. 构造稀疏邻接矩阵 A
    row, col = edge_index
    data = np.ones_like(row, dtype=float)
    A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)  # 确保无向图

    # 2. 对称归一化传播矩阵 S
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = diags(
        np.divide(
            1.0,
            np.sqrt(np.maximum(deg, 1.0)),
            out=np.zeros_like(deg, dtype=float),
            where=deg > 0,
        )
    )
    S = deg_inv_sqrt @ A @ deg_inv_sqrt

    # 3. 收敛传播：直接求逆
    I = diags(np.ones(num_nodes))
    M = I - alpha * S
    Minv = inv(M.tocsc())
    G = (1 - alpha) * (Minv @ Y.T).T  # shape=(batch_size, num_nodes)

    # 4. 源节点检测
    sources = np.zeros_like(seed)
    A_no_self = A.copy()
    A_no_self.setdiag(0)
    A_no_self.eliminate_zeros()

    for b in range(batch_size):
        infected_mask = Y[b] == 1
        g_values = G[b]

        max_neighbor = np.full(num_nodes, -np.inf)
        for i in range(num_nodes):
            neighbors = A_no_self[i].indices
            if neighbors.size:
                max_neighbor[i] = g_values[neighbors].max()

        sources[b] = (infected_mask & (g_values > max_neighbor)).astype(int)

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


def predict_lpsi(file_name: str, type_: str, alpha: float):
    x_test, y_test, edge_index = load_data(file_name, type_)
    # 调用 LPSI 算法
    sources = lpsi(edge_index, x_test, y_test, alpha)
    # 评估
    roc, precision, recall, f1 = evaluate(sources, y_test)
    return roc, precision, recall, f1


if __name__ == "__main__":
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]
    result = {}
    alpha = 0.5

    # 创建结果文件
    with open("result/LPSI.txt", "w") as f:
        f.write("LPSI Algorithm Results\n")
        f.write("-" * 70 + "\n")

        for type_ in type_list:
            for name in file_list:
                start = time.time()
                roc, precision, recall, f1 = predict_lpsi(
                    name, type_, alpha
                )  # 修改为返回4个指标
                end = time.time()
                result[(name, type_)] = {
                    "time": end - start,
                    "roc": roc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
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
