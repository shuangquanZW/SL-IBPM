import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from utils import SEEDS, compute_stats, save_csv

os.environ["OMP_NUM_THREADS"] = "1"


def _build_adj(edge_index, num_nodes):
    row, col = edge_index
    data = np.ones(len(row), dtype=float)
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return adj.maximum(adj.T)


def _reconstruct_path(predecessors, start, end):
    path, current = [], end
    while current != start:
        path.append(current)
        current = predecessors[current]
        if current == -9999:
            return np.array([])
    path.append(start)
    return np.array(path[::-1])


def _ensure_connectivity(adj, cand, obs_infected):
    sub_adj = adj[np.ix_(cand, cand)]
    n_components, labels = connected_components(sub_adj, directed=False)
    if n_components == 1:
        return cand
    obs_idx = np.where(obs_infected[cand])[0]
    if len(obs_idx) == 0:
        return cand
    comp_reps = []
    for i in range(n_components):
        comp_nodes = np.where(labels == i)[0]
        comp_obs = np.intersect1d(comp_nodes, obs_idx)
        comp_reps.append(comp_obs[0] if len(comp_obs) > 0 else comp_nodes[0])
    added_nodes = set()
    base_node = cand[comp_reps[0]]
    _, predecessors = shortest_path(adj, indices=base_node, return_predecessors=True)
    for i in range(1, len(comp_reps)):
        target_node = cand[comp_reps[i]]
        path = _reconstruct_path(predecessors, base_node, target_node)
        if len(path) > 0:
            added_nodes.update(path)
    all_nodes = np.concatenate([cand, np.array(list(added_nodes))])
    return np.unique(all_nodes)


def _candidate_selection(adj, obs_infected, Y):
    infected_mask = obs_infected.astype(bool)
    neighbor_infected = adj.dot(infected_mask.astype(float)).astype(int)
    cond1 = infected_mask
    cond2 = (~infected_mask) & (neighbor_infected >= Y)
    return np.where(cond1 | cond2)[0]


def _compute_distances(adj, cand, obs_infected):
    obs_in_subgraph = np.where(obs_infected[cand])[0]
    if len(obs_in_subgraph) == 0:
        return np.zeros((len(cand), 0))
    sub_adj = adj[np.ix_(cand, cand)]
    dist_matrix = shortest_path(
        sub_adj, directed=False, indices=obs_in_subgraph, unweighted=True
    )
    return dist_matrix.T


def ajc(edge_index, state, seed, Y=2, seed_val=42):
    num_nodes = state.shape[1]
    A = _build_adj(edge_index, num_nodes)
    batch_size = state.shape[0]
    sources = np.zeros_like(seed)
    obs_mask = (state[..., 0] < 0.5).astype(int)

    for b in range(batch_size):
        obs_infected = obs_mask[b]
        cand = _candidate_selection(A, obs_infected, Y=Y)
        cand = _ensure_connectivity(A, cand, obs_infected)
        m = int(np.count_nonzero(seed[b]))
        if len(cand) < m:
            obs_degrees = A.dot(obs_infected)
            top_obs = np.argsort(-obs_degrees)[: max(0, m - len(cand))]
            cand = np.unique(np.concatenate([cand, top_obs]))
        D = _compute_distances(A, cand, obs_infected)
        if D.shape[1] == 0:
            continue
        kmeans = KMeans(
            n_clusters=min(m, len(cand)),
            n_init=10,
            max_iter=50,
            random_state=seed_val,
        ).fit(D)
        source_idx = []
        for i in range(kmeans.n_clusters):  # type: ignore
            cluster_mask = kmeans.labels_ == i
            if not np.any(cluster_mask):
                continue
            cluster_dist = D[cluster_mask]
            eccentricity = np.max(cluster_dist, axis=1)
            min_idx = np.argmin(eccentricity)
            source_idx.append(np.where(cluster_mask)[0][min_idx])
        selected_nodes = cand[source_idx]
        sources[b, selected_nodes] = 1
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


def load_ajc_data(file_name: str, type_: str, seed: int = 0):
    """Load AJC data with seed-based random splits."""
    state = np.load(f"./data/{type_}/{file_name}/state.npy")
    seed_labels = np.load(f"./data/{type_}/{file_name}/seed.npy")
    edge_index = np.load(f"./data/{type_}/{file_name}/edge_index.npy")

    total_size = len(state)
    train_size = int(0.8 * total_size)
    valid_size = int(0.1 * total_size)

    rng = np.random.RandomState(seed)
    indices = np.arange(total_size)
    rng.shuffle(indices)

    test_idx = indices[train_size + valid_size :]
    y_test = seed_labels[test_idx]
    x_test = state[test_idx]
    return x_test, y_test, edge_index


def predict_ajc(file_name: str, type_: str, Y: int, seed_val: int, split_seed: int = 0):
    x_test, y_test, edge_index = load_ajc_data(file_name, type_, seed=split_seed)
    sources = ajc(edge_index, x_test, y_test, Y, seed_val)
    roc, precision, recall, f1 = evaluate(sources, y_test)
    return roc, precision, recall, f1


def main():
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]
    Y = 2
    results = {}
    for type_ in type_list:
        for name in file_list:
            auc_list, pre_list, rec_list, f1_list = [], [], [], []
            for seed in SEEDS:
                np.random.seed(seed)
                roc, pre, rec, f1 = predict_ajc(
                    name, type_, Y, seed_val=seed, split_seed=seed
                )
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
    save_csv(results, "result/AJC.csv")


if __name__ == "__main__":
    main()
