import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import inv
from utils import load_data, SEEDS, compute_stats, save_csv


class InputGenerator:
    def __init__(self, edge_index, num_nodes, alpha=0.5, device="cuda:0"):
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.device = device
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).astype(float)
        adj = adj.maximum(adj.T)
        deg = np.array(adj.sum(1)).flatten()
        deg_inv_sqrt = diags(np.power(np.maximum(deg, 1.0), -0.5))
        self.S = deg_inv_sqrt @ adj @ deg_inv_sqrt
        self.Minv = None

    def forward(self, state):
        Y = 1 - state[..., 0].cpu().numpy()
        if self.Minv is None:
            N = Y.shape[1]
            I = diags(np.ones(N))
            M = I - self.alpha * self.S
            self.Minv = inv(M.tocsc())
        d1 = Y
        v3 = np.where(Y == 1, 1, 0)
        v4 = np.where(Y == -1, 1, 0)
        d2 = (1 - self.alpha) * (self.Minv @ Y.T).T
        d3 = (1 - self.alpha) * (self.Minv @ v3.T).T
        d4 = (1 - self.alpha) * (self.Minv @ v4.T).T
        feat = np.stack([d1, d2, d3, d4], axis=2)
        return torch.tensor(feat, dtype=torch.float32).to(self.device)


class GCNSI(nn.Module):
    def __init__(self, in_dim=4, hidden=512, layers=5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.dense = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        B, N, _ = x.shape
        x = x.view(B * N, 4)
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        x = self.dense(x)
        return torch.sigmoid(x.view(B, N))


class GCNSITrainer:
    def __init__(self, model: GCNSI, lr: float, reduction: str, device: str):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.estimator = nn.BCELoss(reduction=reduction)

    def train_step(self, train_loader, edge_index, mask=None):
        edge_index = edge_index.to(self.device)
        total_loss = 0.0
        for x, true in train_loader:
            x, true = x.to(self.device), true.to(self.device)
            if mask:
                x[:, mask, :] = 0
            self.optimizer.zero_grad()
            pred = self.model(x, edge_index)
            loss = self.estimator(pred, true)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def valid_step(self, valid_loader, edge_index, mask=None):
        edge_index = edge_index.to(self.device)
        total_loss = 0.0
        with torch.no_grad():
            for x, true in valid_loader:
                x, true = x.to(self.device), true.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                pred = self.model(x, edge_index)
                loss = self.estimator(pred, true)
                total_loss += loss.item()
        return total_loss / len(valid_loader)

    def test_step(self, test_loader, edge_index, mask=None):
        edge_index = edge_index.to(self.device)
        auroc_list, precision_list, recall_list, f1_list = [], [], [], []
        with torch.no_grad():
            for x, true in test_loader:
                x, true = x.to(self.device), true.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                pred = self.model(x, edge_index).reshape(-1).cpu().numpy()
                true_np = true.reshape(-1).cpu().numpy()
                auroc_list.append(roc_auc_score(true_np, pred))
                pred_binary = (pred > 0.5).astype(int)
                precision_list.append(
                    precision_score(true_np, pred_binary, zero_division=0)
                )
                recall_list.append(recall_score(true_np, pred_binary, zero_division=0))
                f1_list.append(f1_score(true_np, pred_binary, zero_division=0))
        return (
            float(np.mean(auroc_list)),
            float(np.mean(precision_list)),
            float(np.mean(recall_list)),
            float(np.mean(f1_list)),
        )

    def fit(self, train_loader, valid_loader, edge_index, epochs, mask=None):
        for epoch in range(epochs):
            self.model.train()
            train_loss = self.train_step(train_loader, edge_index, mask)
            self.model.eval()
            valid_loss = self.valid_step(valid_loader, edge_index, mask)
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
                )

    def evaluate(self, test_loader, edge_index, mask=None):
        self.model.eval()
        return self.test_step(test_loader, edge_index, mask)


def preprocess_dataloader(dataloader, generator, device):
    all_states, all_labels = [], []
    for state, label in dataloader:
        all_states.append(state)
        all_labels.append(label)
    state = torch.cat(all_states, dim=0).to(device)
    label = torch.cat(all_labels, dim=0).to(device)
    with torch.no_grad():
        new_x = generator.forward(state)
    dataset = TensorDataset(new_x, label)
    return DataLoader(dataset, batch_size=dataloader.batch_size)


def train_gcnsi(
    file_name: str,
    type_: str,
    seed: int = 0,
    epochs: int = 100,
    lr: float = 0.001,
    reduction: str = "mean",
    device: str = "cuda:0",
):
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        num_nodes,
        _,
        _,
    ) = load_data(file_name, type_)
    generator = InputGenerator(edge_index=edge_index.to(device), num_nodes=num_nodes)
    train_loader = preprocess_dataloader(train_loader, generator, device)
    valid_loader = preprocess_dataloader(valid_loader, generator, device)
    test_loader = preprocess_dataloader(test_loader, generator, device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GCNSI()
    trainer = GCNSITrainer(model=model, lr=lr, reduction=reduction, device=device)
    trainer.fit(train_loader, valid_loader, edge_index, epochs)
    auc, pre, rec, f1 = trainer.evaluate(test_loader, edge_index)
    return auc, pre, rec, f1


def main():
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]
    results = {}
    for name in file_list:
        for type_ in type_list:
            auc_list, pre_list, rec_list, f1_list = [], [], [], []
            for seed in SEEDS:
                auc, pre, rec, f1 = train_gcnsi(file_name=name, type_=type_, seed=seed)
                auc_list.append(auc)
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
    save_csv(results, "result/GCNSI.csv")


def cascade_study(device="cuda:0", epochs=100):
    from utils import get_true_cascade_dataset

    file_list = ["android", "christianity", "douban", "twitter"]
    results = {}
    for name in file_list:
        train_ld, valid_ld, test_ld, ei, N, _, _ = get_true_cascade_dataset(name)
        gen = InputGenerator(ei.to(device), N)
        train_ld = preprocess_dataloader(train_ld, gen, device)
        valid_ld = preprocess_dataloader(valid_ld, gen, device)
        test_ld = preprocess_dataloader(test_ld, gen, device)
        auc_list, pre_list, rec_list, f1_list = [], [], [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = GCNSI()
            trainer = GCNSITrainer(model, lr=0.001, reduction="mean", device=device)
            trainer.fit(train_ld, valid_ld, ei, epochs)
            auc, pre, rec, f1 = trainer.evaluate(test_ld, ei)
            auc_list.append(auc)
            pre_list.append(pre)
            rec_list.append(rec)
            f1_list.append(f1)
        results[(name, "cascade")] = {
            "auc_mean": compute_stats(auc_list)[0],
            "auc_std": compute_stats(auc_list)[1],
            "pre_mean": compute_stats(pre_list)[0],
            "pre_std": compute_stats(pre_list)[1],
            "rec_mean": compute_stats(rec_list)[0],
            "rec_std": compute_stats(rec_list)[1],
            "f1_mean": compute_stats(f1_list)[0],
            "f1_std": compute_stats(f1_list)[1],
        }
    save_csv(results, "result/GCNSI_cascade.csv")


if __name__ == "__main__":
    main()
    cascade_study(device="cuda:1")
