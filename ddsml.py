import os
import math
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ----------------- 工具函数 -----------------
def set_seed(seed):
    """设置全局随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_stats(vals):
    return float(np.mean(vals)), float(np.std(vals))


def save_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("dataset,model,metric,mean,std\n")
        for (name, typ), met in results.items():
            for k in ["auc", "pre", "rec", "f1"]:
                f.write(f"{name},{typ},{k},{met[k+'_mean']:.4f},{met[k+'_std']:.4f}\n")


# ----------------- 时间步嵌入 -----------------
class TimeEmbedding(nn.Module):
    """正弦位置编码，用于将离散时间步 t 映射为连续向量"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


# ----------------- 蒙特卡洛仿真 -----------------
def simulate_SIR(g, sources, beta, gamma, T):
    N = g.number_of_nodes()
    state = np.zeros(N, dtype=np.int32)
    state[list(sources)] = 1
    traj = [state.copy()]
    for _ in range(1, T + 1):
        nxt = state.copy()
        for i in range(N):
            if state[i] == 0:
                if any(state[j] == 1 for j in g.neighbors(i)):
                    if np.random.random() < beta:
                        nxt[i] = 1
            elif state[i] == 1:
                if np.random.random() < gamma:
                    nxt[i] = 2
        state = nxt
        traj.append(state.copy())
    return np.array(traj)


def simulate_SI(g, sources, beta, T):
    N = g.number_of_nodes()
    state = np.zeros(N, dtype=np.int32)
    state[list(sources)] = 1
    traj = [state.copy()]
    for _ in range(1, T + 1):
        nxt = state.copy()
        for i in range(N):
            if state[i] == 0:
                if any(state[j] == 1 for j in g.neighbors(i)):
                    if np.random.random() < beta:
                        nxt[i] = 1
        state = nxt
        traj.append(state.copy())
    return np.array(traj)


def simulate_IC(g, sources, p, T):
    N = g.number_of_nodes()
    active = np.zeros(N, dtype=bool)
    active[list(sources)] = True
    traj = [active.astype(int).copy()]
    newly = set(sources)
    for _ in range(1, T + 1):
        next_newly = set()
        for u in newly:
            for v in g.neighbors(u):
                if not active[v] and np.random.random() < p:
                    active[v] = True
                    next_newly.add(v)
        newly = next_newly
        traj.append(active.astype(int).copy())
    return np.array(traj)


def simulate_LT(g, sources, T):
    N = g.number_of_nodes()
    thresh = np.random.random(N)
    active = np.zeros(N, dtype=bool)
    active[list(sources)] = True
    traj = [active.astype(int).copy()]
    for _ in range(1, T + 1):
        changed = False
        for i in range(N):
            if not active[i]:
                neigh = list(g.neighbors(i))
                if len(neigh) > 0:
                    ratio = np.mean([active[j] for j in neigh])
                    if ratio >= thresh[i]:
                        active[i] = True
                        changed = True
        traj.append(active.astype(int).copy())
        if not changed:
            break
    while len(traj) <= T:
        traj.append(traj[-1].copy())
    return np.array(traj)


# ----------------- 概率估计与 Qt 生成 -----------------
def estimate_Pt(model, g, sources, T, n_sim=500):
    N = g.number_of_nodes()
    if model == "SIR":
        beta, gamma = 0.03, 0.015
    elif model == "SI":
        beta, gamma = 0.03, 0.0
    elif model == "IC":
        p = 0.1
    elif model == "LT":
        pass
    else:
        raise ValueError(f"Unknown model: {model}")

    M = 3 if model == "SIR" else 2
    counts = [np.zeros((N, M)) for _ in range(T + 1)]
    for _ in range(n_sim):
        if model == "SIR":
            traj = simulate_SIR(g, sources, beta, gamma, T)
        elif model == "SI":
            traj = simulate_SI(g, sources, beta, T)
        elif model == "IC":
            traj = simulate_IC(g, sources, p, T)
        elif model == "LT":
            traj = simulate_LT(g, sources, T)
        else:
            raise ValueError(f"Unknown model: {model}")
        for t in range(T + 1):
            states = traj[t]
            for n in range(N):
                counts[t][n, states[n]] += 1
    return [c / n_sim for c in counts]


def Pt_to_Qt_SIR(prob):
    T = len(prob) - 1
    N = prob[0].shape[0]
    Qt = np.zeros((T, N, 3, 3))
    for t in range(T):
        for i in range(N):
            ps0, pi0, pr0 = prob[t][i]
            ps1, pi1, pr1 = prob[t + 1][i]
            beta_i = np.clip(1.0 - ps1 / (ps0 + 1e-12), 0, 1)
            gamma_i = np.clip((pr1 - pr0) / (pi0 + 1e-12), 0, 1)
            Qt[t, i] = np.array(
                [[1 - beta_i, beta_i, 0], [0, 1 - gamma_i, gamma_i], [0, 0, 1]]
            )
    return Qt


def Pt_to_Qt_binary(prob):
    T = len(prob) - 1
    N = prob[0].shape[0]
    Qt = np.zeros((T, N, 2, 2))
    for t in range(T):
        for i in range(N):
            ps0 = prob[t][i, 0]
            ps1 = prob[t + 1][i, 0]
            beta_i = np.clip(1.0 - ps1 / (ps0 + 1e-12), 0, 1)
            Qt[t, i] = np.array([[1 - beta_i, beta_i], [0, 1]])
    return Qt


def generate_Qt_batch(g, seed_list, model, T, n_sim=500):
    Qt_batch = []
    for seeds in seed_list:
        prob = estimate_Pt(model, g, seeds, T, n_sim)
        if model == "SIR":
            Qt = Pt_to_Qt_SIR(prob)
        else:
            Qt = Pt_to_Qt_binary(prob)
        Qt_batch.append(Qt)
    return np.array(Qt_batch)


def load_or_generate_Qt(g, seeds, model_type, T, data_dir, dataset_name, n_sim=500):
    cache_file = os.path.join(data_dir, f"{dataset_name}_{model_type}_Qt.npy")
    if os.path.exists(cache_file):
        print(f"Loading cached Qt from {cache_file}")
        Qt = np.load(cache_file)
    else:
        print(f"Generating Qt for {dataset_name}/{model_type} (no cache found)...")
        seed_lists = [np.where(seeds[i] == 1)[0].tolist() for i in range(len(seeds))]
        Qt = generate_Qt_batch(g, seed_lists, model_type, T, n_sim=n_sim)
        np.save(cache_file, Qt)
        print(f"Qt saved to {cache_file}")
    return Qt


# ----------------- 前向扩散采样 -----------------
def q_sample(x0, Q_bar_t):
    probs = torch.matmul(x0.unsqueeze(-2), Q_bar_t).squeeze(-2)
    xt = torch.distributions.Categorical(probs).sample()
    xt_onehot = F.one_hot(xt, num_classes=x0.shape[-1]).float()
    return xt_onehot


# ----------------- 约束损失 -----------------
def constraint_loss1(x0_pred, xt):
    return F.mse_loss(x0_pred.mean(dim=1), xt.mean(dim=1))


def constraint_loss2(x0_pred):
    source_prob = x0_pred[..., 1]
    return source_prob.mean()


# ----------------- 图卷积与去噪网络 -----------------
class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim))
        self.b = nn.Parameter(torch.randn(out_dim))

    def forward(self, adj, X):
        deg = torch.sum(adj, dim=1, keepdim=True).clamp(min=1.0)
        norm_adj = adj / torch.sqrt(deg) / torch.sqrt(deg.T)
        X = X.matmul(self.W)
        out = torch.stack([norm_adj @ x for x in X], dim=0) + X + self.b
        return F.leaky_relu(out, negative_slope=0.01)


class GCNB(nn.Module):
    def __init__(self, num_states):
        super().__init__()
        self.linears = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(num_states, 128)),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
        )
        self.conv1 = nn.utils.parametrizations.spectral_norm(
            GraphConvolution(128, 128), name="W"
        )
        self.conv2 = nn.utils.parametrizations.spectral_norm(
            GraphConvolution(128, 128), name="W"
        )
        self.conv3 = nn.utils.parametrizations.spectral_norm(
            GraphConvolution(128, 128), name="W"
        )
        self.bn = nn.BatchNorm1d(128, track_running_stats=False)
        self.time_mlp = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128)
        )
        self.fc = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(128, 32)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.utils.spectral_norm(nn.Linear(32, num_states)),
        )

    def forward(self, adj, x, t_emb):
        _, N, _ = x.shape
        x = self.linears(x)
        te = self.time_mlp(t_emb)
        x = x + te.unsqueeze(1)
        ident = x
        for _ in range(4):
            out = self.conv1(adj, x)
            out = self.bn(out.transpose(1, 2)).transpose(1, 2)
            x = F.mish(out) + ident
            x = F.dropout(x, 0.5, self.training)
            ident = x
        out = self.conv3(adj, x)
        out = self.bn(out.transpose(1, 2)).transpose(1, 2)
        x = F.mish(out) + ident
        x = F.dropout(x, 0.5, self.training)
        logits = self.fc(x)
        return logits


# ----------------- 扩散运算函数 -----------------
def xt_X_Qtrans(x, Q):
    B, N, M = x.shape
    x_ = x.unsqueeze(2)
    Q_T = Q.transpose(2, 3)
    res = torch.bmm(x_.reshape(B * N, 1, M), Q_T.reshape(B * N, M, M))
    return res.reshape(B, N, M)


def xt_X_Q(x, Q):
    B, N, M = x.shape
    x_ = x.unsqueeze(2)
    res = torch.bmm(x_.reshape(B * N, 1, M), Q.reshape(B * N, M, M))
    return res.reshape(B, N, M)


def Q_mult(Q1, Q2):
    B, N, M, _ = Q1.shape
    return torch.bmm(Q1.reshape(B * N, M, M), Q2.reshape(B * N, M, M)).reshape(
        B, N, M, M
    )


def Q_bar(Q, t):
    B, T_full, N, M, _ = Q.shape
    if t == 0:
        return Q[:, 0]
    res = Q[:, 0]
    for i in range(1, t + 1):
        res = Q_mult(res, Q[:, i])
    return res


def posterior_q(xt, x0, Q, t):
    Q_t = Q[:, t - 1]
    Q_bar_t1 = (
        Q_bar(Q, t - 2)
        if t >= 2
        else torch.eye(Q.shape[-1]).to(xt.device).expand(Q.shape[0], Q.shape[2], -1, -1)
    )
    term1 = xt_X_Qtrans(xt, Q_t)
    term2 = xt_X_Q(x0, Q_bar_t1)
    unnorm = term1 * term2
    return unnorm / unnorm.sum(dim=2, keepdim=True).clamp(min=1e-12)


def p_theta(xt, x0_pred, Q, t):
    B, N, M = xt.shape
    Q_t = Q[:, t - 1]
    Q_bar_t1 = (
        Q_bar(Q, t - 2) if t >= 2 else torch.eye(M).to(xt.device).expand(B, N, -1, -1)
    )
    q_xt_xt1 = xt_X_Qtrans(xt, Q_t)
    x0_S = torch.zeros(B, N, M, device=xt.device)
    x0_S[:, :, 0] = 1.0
    x0_I = torch.zeros(B, N, M, device=xt.device)
    x0_I[:, :, 1] = 1.0
    prob_I = x0_pred.unsqueeze(2)
    prob_S = 1.0 - prob_I
    term_S = xt_X_Q(x0_S, Q_bar_t1) * prob_S
    term_I = xt_X_Q(x0_I, Q_bar_t1) * prob_I
    unnorm = (term_S + term_I) * q_xt_xt1
    return unnorm / unnorm.sum(dim=2, keepdim=True).clamp(min=1e-12)


def sample_posterior(xt, x0_pred, Q, t, hard=False):
    probs = p_theta(xt, x0_pred, Q, t)
    return F.gumbel_softmax(torch.log(probs + 1e-12), tau=0.001, hard=hard)


# ----------------- 损失函数 -----------------
def Lvb_loss(x0_true, x0_pred_prob, xt, Q, t):
    M = Q.shape[-1]
    x0_onehot = F.one_hot(x0_true.long(), num_classes=M).float()
    q_posterior = posterior_q(xt, x0_onehot, Q, t)
    p_posterior = p_theta(xt, x0_pred_prob, Q, t)
    p_posterior_safe = p_posterior.clamp(min=1e-12)
    return F.kl_div(p_posterior_safe.log(), q_posterior, reduction="batchmean")


# ----------------- 训练器 -----------------
class DDMSLTrainer:
    def __init__(self, model, adj, T, lr, device, num_states, pos_weight=9.0):
        self.model = model.to(device)
        self.adj = adj.to(device)
        self.T = T
        self.num_states = num_states
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.time_embed = TimeEmbedding(dim=128).to(device)
        self.pos_weight = torch.tensor([pos_weight], device=device)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def train_step(self, x0, xt_raw, Qt, t):
        self.model.train()
        B, N = x0.shape
        M = self.num_states

        x0_onehot = F.one_hot(x0.long(), num_classes=M).float().to(self.device)
        t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
        t_emb = self.time_embed(t_tensor)

        Q_bar_t = Q_bar(Qt, t - 1)
        xt = q_sample(x0_onehot, Q_bar_t)

        logits = self.model(self.adj, xt, t_emb)
        x0_pred_prob = F.softmax(logits, dim=-1)
        source_prob = x0_pred_prob[..., 1]

        lvb = Lvb_loss(x0, source_prob, xt, Qt, t)
        lc1 = constraint_loss1(x0_pred_prob, xt)
        lc2 = constraint_loss2(x0_pred_prob)

        source_logits = logits[..., 1]
        source_target = x0.float()
        cls_loss = self.bce_loss(source_logits, source_target)

        loss = lvb + lc1 + lc2 + cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "lvb": lvb.item(),
            "lc1": lc1.item(),
            "lc2": lc2.item(),
            "cls": cls_loss.item(),
        }

    def infer_sources(self, xt_obs, Qt):
        self.model.eval()
        M = self.num_states

        if xt_obs.dim() == 3:
            B, N, _ = xt_obs.shape
            current = xt_obs.float().to(self.device)
        else:
            B, N = xt_obs.shape
            current = F.one_hot(xt_obs.long(), num_classes=M).float().to(self.device)

        with torch.no_grad():
            for t in range(self.T, 0, -1):
                t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)
                t_emb = self.time_embed(t_tensor)

                logits = self.model(self.adj, current, t_emb)
                x0_pred_prob = F.softmax(logits, dim=-1)
                source_prob = x0_pred_prob[..., 1]

                if t > 1:
                    current = sample_posterior(current, source_prob, Qt, t, hard=True)
                else:
                    return source_prob
        return source_prob

    def test_epoch(self, test_loader, num_states, threshold=0.5):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_x0, batch_xt, batch_Qt in test_loader:
                batch_x0 = batch_x0.to(self.device)
                batch_xt = batch_xt.to(self.device)
                batch_Qt = batch_Qt.to(self.device)
                x0_pred = self.infer_sources(batch_xt, batch_Qt)
                y_true.append(batch_x0.cpu().numpy().ravel())
                y_pred.append(x0_pred.cpu().numpy().ravel())
        yt = np.concatenate(y_true)
        yp = np.concatenate(y_pred)
        auc = roc_auc_score(yt, yp)
        y_bin = (yp > threshold).astype(int)
        pre = precision_score(yt, y_bin, zero_division=0)
        rec = recall_score(yt, y_bin, zero_division=0)
        f1 = f1_score(yt, y_bin, zero_division=0)
        return auc, pre, rec, f1


# ----------------- 单次实验函数 -----------------
def run_single_experiment(
    seed, g, adj, seeds, X_T, Qt, num_states, T, lr, epochs, device
):
    """以指定随机种子运行单次实验"""
    set_seed(seed)

    N_total = len(seeds)
    indices = np.random.permutation(N_total)
    train_split = int(0.7 * N_total)
    val_split = int(0.8 * N_total)
    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]
    test_idx = indices[val_split:]

    train_seeds = torch.tensor(seeds[train_idx])
    train_XT = torch.tensor(X_T[train_idx], dtype=torch.long)
    train_Qt = Qt[train_idx]

    val_seeds = torch.tensor(seeds[val_idx])
    val_XT = torch.tensor(X_T[val_idx], dtype=torch.long)
    val_Qt = Qt[val_idx]

    test_seeds = torch.tensor(seeds[test_idx])
    test_XT = torch.tensor(X_T[test_idx], dtype=torch.long)
    test_Qt = Qt[test_idx]

    train_ds = TensorDataset(train_seeds, train_XT, train_Qt)
    val_ds = TensorDataset(val_seeds, val_XT, val_Qt)
    test_ds = TensorDataset(test_seeds, test_XT, test_Qt)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = GCNB(num_states)
    trainer = DDMSLTrainer(model, adj, T, lr, device, num_states)

    best_val_f1 = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x0, batch_xt, batch_Qt in train_loader:
            batch_x0 = batch_x0.to(device)
            batch_Qt = batch_Qt.to(device)
            t = np.random.randint(1, T + 1)
            loss_dict = trainer.train_step(batch_x0, batch_xt, batch_Qt, t)
            total_loss += loss_dict["loss"]

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"  [Seed {seed}] Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")
            val_auc, val_pre, val_rec, val_f1 = trainer.test_epoch(
                val_loader, num_states, threshold=0.5
            )
            print(
                f"  [Seed {seed}] Val: AUC={val_auc:.4f}, Prec={val_pre:.4f}, "
                f"Rec={val_rec:.4f}, F1={val_f1:.4f}"
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

    auc, pre, rec, f1 = trainer.test_epoch(test_loader, num_states, threshold=0.5)
    print(
        f"  [Seed {seed}] Final Test: AUC={auc:.4f}, Prec={pre:.4f}, "
        f"Rec={rec:.4f}, F1={f1:.4f}"
    )
    return auc, pre, rec, f1


# ----------------- 主函数 -----------------
def main(
    data_root,
    dataset_list,
    model_types,
    T=20,
    n_sim=500,
    lr=0.001,
    epochs=50,
    seeds_list=(0, 1, 2, 3, 4),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = {}

    for name in dataset_list:
        for typ in model_types:
            print(f"\n===== Processing {typ}/{name} =====")
            seeds_path = os.path.join(data_root, typ, name, "seed.npy")
            snap_path = os.path.join(data_root, typ, name, "state.npy")
            edge_path = os.path.join(data_root, typ, name, "edge_index.npy")
            if not all(os.path.exists(p) for p in [seeds_path, snap_path, edge_path]):
                print(f"Missing files for {typ}/{name}, skipping")
                continue

            seeds = np.load(seeds_path)
            X_T = np.load(snap_path)
            edges = np.load(edge_path)

            if X_T.ndim == 2:
                num_states = int(X_T.max()) + 1
            else:
                num_states = X_T.shape[-1]

            N_nodes = X_T.shape[1]
            g = nx.Graph()
            g.add_nodes_from(range(N_nodes))
            g.add_edges_from(edges.T.tolist())
            adj = torch.tensor(nx.adjacency_matrix(g).toarray(), dtype=torch.float)

            Qt = load_or_generate_Qt(g, seeds, typ, T, data_root, name, n_sim=n_sim)
            Qt = torch.tensor(Qt, dtype=torch.float)

            # 多次实验
            auc_list, pre_list, rec_list, f1_list = [], [], [], []
            for seed in seeds_list:
                print(f"\n--- Running with seed={seed} ---")
                auc, pre, rec, f1 = run_single_experiment(
                    seed, g, adj, seeds, X_T, Qt, num_states, T, lr, epochs, device
                )
                auc_list.append(auc)
                pre_list.append(pre)
                rec_list.append(rec)
                f1_list.append(f1)

            auc_mean, auc_std = compute_stats(auc_list)
            pre_mean, pre_std = compute_stats(pre_list)
            rec_mean, rec_std = compute_stats(rec_list)
            f1_mean, f1_std = compute_stats(f1_list)

            print(
                f"\n===== Summary for {typ}/{name} (over {len(seeds_list)} seeds) ====="
            )
            print(f"  AUC : {auc_mean:.4f} ± {auc_std:.4f}")
            print(f"  Prec: {pre_mean:.4f} ± {pre_std:.4f}")
            print(f"  Rec : {rec_mean:.4f} ± {rec_std:.4f}")
            print(f"  F1  : {f1_mean:.4f} ± {f1_std:.4f}")

            all_results[(name, typ)] = {
                "auc_mean": auc_mean,
                "auc_std": auc_std,
                "pre_mean": pre_mean,
                "pre_std": pre_std,
                "rec_mean": rec_mean,
                "rec_std": rec_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
            }

            # 每完成一个数据集即保存一次（防止中断丢失结果）
            save_csv(all_results, "result/DDMSL.csv")

    save_csv(all_results, "result/DDMSL.csv")
    print("\nAll experiments completed. Results saved to result/DDMSL.csv")


if __name__ == "__main__":
    main(
        data_root="./data",
        dataset_list=[
            "karate",
            "jazz",
            "cora_ml",
            "net_science",
            "power_grid",
            # "lastFM",
        ],
        model_types=["SIR", "SI", "IC", "LT"],
        T=20,
        n_sim=500,
        epochs=100,
        seeds_list=(0, 1, 2, 3, 4),
    )
