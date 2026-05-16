import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import inv
from utils import SEEDS, compute_stats, save_csv


def lpsi(edge_index, state, seed, alpha):
    batch_size, num_nodes, _ = state.shape
    Y = np.where(state[..., 0] > 0.5, -1.0, 1.0)
    row, col = edge_index
    data = np.ones_like(row, dtype=float)
    A = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = diags(np.divide(1.0, np.sqrt(np.maximum(deg, 1.0)), where=deg > 0))
    S = deg_inv_sqrt @ A @ deg_inv_sqrt
    I = diags(np.ones(num_nodes))
    M = I - alpha * S
    Minv = inv(M.tocsc())
    G = (1 - alpha) * (Minv @ Y.T).T
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


def evaluate(pred, true):
    y_true = true.reshape(-1)
    y_score = pred.reshape(-1)
    roc = roc_auc_score(y_true, y_score)
    y_pred = np.where(y_score > 0.5, 1, 0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return roc, precision, recall, f1


def load_lpsi_data(file_name: str, type_: str, seed_: int = 0):
    state = np.load(f"./data/{type_}/{file_name}/state.npy")
    seed = np.load(f"./data/{type_}/{file_name}/seed.npy")
    edge_index = np.load(f"./data/{type_}/{file_name}/edge_index.npy")
    rng = np.random.RandomState(seed_)
    indices = np.arange(len(state))
    rng.shuffle(indices)
    train_size = int(0.8 * len(state))
    valid_size = int(0.1 * len(state))
    test_idx = indices[train_size + valid_size :]
    y_test = seed[test_idx]
    x_test = state[test_idx]
    return x_test, y_test, edge_index


def predict_lpsi(file_name: str, type_: str, alpha: float, seed_: int = 0):
    x_test, y_test, edge_index = load_lpsi_data(file_name, type_, seed_)
    sources = lpsi(edge_index, x_test, y_test, alpha)
    roc, precision, recall, f1 = evaluate(sources, y_test)
    return roc, precision, recall, f1


def main():
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]
    alpha = 0.5
    results = {}
    for type_ in type_list:
        for name in file_list:
            auc_list, pre_list, rec_list, f1_list = [], [], [], []
            for seed in SEEDS:
                np.random.seed(seed)
                roc, pre, rec, f1 = predict_lpsi(name, type_, alpha, seed)
                auc_list.append(roc)
                pre_list.append(pre)
                rec_list.append(rec)
                f1_list.append(f1)
            results[(name, type_)] = {
                "auc_mean": compute_stats(auc_list)[0],
                "auc_std": compute_stats(auc_list)[1],
                "pre_mean": compute_stats(pre_list)[0],
                "pre_std": compute_stats(pre_list)[1],
                "rec_mean": compute_stats(rec_list)[0],
                "rec_std": compute_stats(rec_list)[1],
                "f1_mean": compute_stats(f1_list)[0],
                "f1_std": compute_stats(f1_list)[1],
            }
    save_csv(results, "result/LPSI.csv")


if __name__ == "__main__":
    main()
