import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GATConv
from torch_geometric.utils import degree
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import networkx as nx
from utils import load_data, SEEDS, compute_stats, save_csv


class HFSDFeatureGenerator:
    def __init__(self, edge_index, num_nodes):
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.G = nx.Graph()
        self.G.add_edges_from(edge_index.T.tolist())
        self.closeness = nx.closeness_centrality(self.G)
        self.closeness = torch.tensor(
            [self.closeness.get(i, 0) for i in range(num_nodes)], dtype=torch.float32
        ).to(edge_index.device)

    def _neighbor_ratio(self, y_single):
        row, col = self.edge_index
        adj_t = torch.sparse_coo_tensor(
            indices=torch.stack([col, row]),
            values=torch.ones_like(row, dtype=torch.float32),
            size=(self.num_nodes, self.num_nodes),
        )
        neighbor_sum = torch.sparse.mm(adj_t, y_single.unsqueeze(1)).squeeze(1)
        neighbor_count = degree(row, num_nodes=self.num_nodes)
        return neighbor_sum / (neighbor_count + 1e-6)

    def _degree_norm(self):
        deg = degree(self.edge_index[0], num_nodes=self.num_nodes)
        return deg / deg.max()

    def forward(self, state):
        B = state.shape[0]
        state = state[..., 0]
        features = []
        for b in range(B):
            y = state[b]
            x1 = y.float()
            x2 = self._neighbor_ratio(y)
            x3 = self._degree_norm()
            x4 = 1 - x1
            x5 = 1 - x2
            x6 = self.closeness
            feat = torch.stack([x1, x2, x3, x4, x5, x6], dim=1)  # type: ignore
            features.append(feat)
        return torch.stack(features, dim=0)


class HFSDModel(nn.Module):
    def __init__(self, in_dim=6, hidden=128, layers=3, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden, heads=heads))
        for _ in range(layers - 2):
            self.convs.append(GATConv(hidden * heads, hidden, heads=heads))
        self.convs.append(GATConv(hidden * heads, hidden))
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        B, N, _ = x.shape
        x = x.view(B * N, -1)
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        x = self.classifier(x)
        return torch.sigmoid(x.view(B, N))


class BiasedEstimator(nn.Module):
    def __init__(self, alpha: float, reduction: str, device: str):
        super().__init__()
        self.pos_weight = torch.tensor([1 / alpha]).to(device)
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight, reduction=reduction
        )

    def forward(self, pred, true):
        loss = self.criterion(pred, true.float())
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum() if self.reduction == "sum" else loss


class HFSDTrainer:
    def __init__(self, model, num_nodes, lr=1e-3, device="cuda", lambda_=0.05):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = BiasedEstimator(alpha=lambda_, reduction="mean", device=device)
        self.num_nodes = num_nodes

    def train_step(self, loader, edge_index, mask=None):
        edge_index = edge_index.to(self.device)
        total_loss = 0.0
        for state, label in loader:
            state, label = state.to(self.device), label.to(self.device)
            if mask:
                state[:, mask, :] = 0
            self.optimizer.zero_grad()
            pred = self.model(state, edge_index)
            loss = self.criterion(pred, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def valid_step(self, loader, edge_index, mask=None):
        edge_index = edge_index.to(self.device)
        total_loss = 0.0
        with torch.no_grad():
            for state, label in loader:
                state, label = state.to(self.device), label.to(self.device)
                if mask:
                    state[:, mask, :] = 0
                pred = self.model(state, edge_index)
                loss = self.criterion(pred, label)
                total_loss += loss.item()
        return total_loss / len(loader)

    def test_step(self, loader, edge_index, mask=None):
        edge_index = edge_index.to(self.device)
        auroc_list, precision_list, recall_list, f1_list = [], [], [], []
        with torch.no_grad():
            for state, label in loader:
                state, label = state.to(self.device), label.to(self.device)
                if mask:
                    state[:, mask, :] = 0
                y_hat = self.model(state, edge_index).reshape(-1).cpu().numpy()
                y = label.reshape(-1).cpu().numpy()
                auroc_list.append(roc_auc_score(y, y_hat))
                y_hat = (y_hat > 0.5).astype(int)
                precision_list.append(precision_score(y, y_hat, zero_division=0))
                recall_list.append(recall_score(y, y_hat, zero_division=0))
                f1_list.append(f1_score(y, y_hat, zero_division=0))
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
                    f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Valid: {valid_loss:.4f}"
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


def train_hfsd(
    file_name: str, type_: str, epochs=100, lr=1e-3, lambda_=0.05, device="cuda"
):
    train_loader, valid_loader, test_loader, edge_index, num_nodes, _, _ = load_data(
        file_name, type_
    )
    generator = HFSDFeatureGenerator(
        edge_index=edge_index.to(device), num_nodes=num_nodes
    )
    train_loader = preprocess_dataloader(train_loader, generator, device)
    valid_loader = preprocess_dataloader(valid_loader, generator, device)
    test_loader = preprocess_dataloader(test_loader, generator, device)

    auc_list, pre_list, rec_list, f1_list = [], [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = HFSDModel()
        trainer = HFSDTrainer(model, num_nodes, lr=lr, lambda_=lambda_, device=device)
        trainer.fit(train_loader, valid_loader, edge_index, epochs=epochs)
        auc, pre, rec, f1 = trainer.evaluate(test_loader, edge_index)
        auc_list.append(auc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)
    return {
        "auc_mean": compute_stats(auc_list)[0],
        "auc_std": compute_stats(auc_list)[1],
        "pre_mean": compute_stats(pre_list)[0],
        "pre_std": compute_stats(pre_list)[1],
        "rec_mean": compute_stats(rec_list)[0],
        "rec_std": compute_stats(rec_list)[1],
        "f1_mean": compute_stats(f1_list)[0],
        "f1_std": compute_stats(f1_list)[1],
    }


def main():
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]
    results = {}
    for name in file_list:
        for type_ in type_list:
            metrics = train_hfsd(file_name=name, type_=type_)
            results[(name, type_)] = metrics
    save_csv(results, "result/HFSD.csv")


def cascade_study(device="cuda", epochs=100):
    from utils import get_true_cascade_dataset

    file_list = ["android", "christianity", "douban", "twitter"]
    results = {}
    for name in file_list:
        train_ld, valid_ld, test_ld, ei, N, _, _ = get_true_cascade_dataset(name)
        gen = HFSDFeatureGenerator(ei.to(device), N)
        train_ld = preprocess_dataloader(train_ld, gen, device)
        valid_ld = preprocess_dataloader(valid_ld, gen, device)
        test_ld = preprocess_dataloader(test_ld, gen, device)
        auc_list, pre_list, rec_list, f1_list = [], [], [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = HFSDModel()
            trainer = HFSDTrainer(model, N, lr=1e-3, device=device, lambda_=0.05)
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
    save_csv(results, "result/HFSD_cascade.csv")


if __name__ == "__main__":
    main()
    # cascade_study(device="cuda:1")
