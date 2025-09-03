import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GATConv
from torch_geometric.utils import degree
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import networkx as nx
from util import load_data, get_true_cascade_dataset


class HFSDFeatureGenerator:
    def __init__(self, edge_index, num_nodes):
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.G = self._build_graph()
        self.closeness = nx.closeness_centrality(self.G)
        self.closeness = torch.tensor(
            [self.closeness.get(i, 0) for i in range(num_nodes)], dtype=torch.float32
        ).to(edge_index.device)

    def _build_graph(self):
        G = nx.Graph()
        G.add_edges_from(self.edge_index.T.tolist())
        return G

    def forward(self, state):
        """
        state: (B, N, 1)
        return: (B, N, 6)
        """
        B = state.shape[0]
        state = state[..., 0]  # (B, N)

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
        return torch.stack(features, dim=0)  # (B, N, 6)

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
    """有偏估计器"""

    def __init__(self, alpha: float, reduction: str, device: str) -> None:
        super().__init__()
        self.pos_weight = torch.tensor([1 / alpha]).to(device)
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight, reduction=reduction
        )

    def forward(self, pred, true):
        """计算有偏估计二元交叉熵损失"""
        loss = self.criterion(pred, true.float())
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class HFSDTrainer:
    def __init__(self, model, num_nodes, lr=1e-3, device="cuda", lambda_=0.05):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = BiasedEstimator(alpha=lambda_, reduction="mean", device=device)
        self.lambda_ = lambda_
        self.num_nodes = num_nodes

    def train_step(
        self,
        loader,
        edge_index,
        mask: list | None = None,
    ):
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

    def valid_step(
        self,
        loader,
        edge_index,
        mask: list | None = None,
    ):
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

    def test_step(
        self,
        loader,
        edge_index,
        mask: list | None = None,
    ):
        edge_index = edge_index.to(self.device)
        auroc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        with torch.no_grad():
            for state, label in loader:
                state, label = state.to(self.device), label.to(self.device)
                if mask:
                    state[:, mask, :] = 0
                y_hat = self.model(state, edge_index).reshape(-1).cpu().numpy()
                y = label.reshape(-1).cpu().numpy()
                auroc = roc_auc_score(y, y_hat)
                y_hat = (y_hat > 0.5).float()
                precision = precision_score(y, y_hat)
                recall = recall_score(y, y_hat)
                f1 = f1_score(y, y_hat)
                auroc_list.append(auroc)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
        return (
            np.mean(auroc_list),
            np.mean(precision_list),
            np.mean(recall_list),
            np.mean(f1_list),
        )

    def fit(
        self,
        train_loader,
        valid_loader,
        edge_index,
        epochs=100,
        mask: list | None = None,
    ):
        for epoch in range(epochs):
            self.model.train()
            train_loss = self.train_step(train_loader, edge_index, mask)
            self.model.eval()
            valid_loss = self.valid_step(valid_loader, edge_index, mask)
            print(
                f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Valid: {valid_loss:.4f}"
            )

    def evaluate(
        self,
        test_loader,
        edge_index,
        mask: list | None = None,
    ):
        self.model.eval()
        return self.test_step(test_loader, edge_index, mask)


def preprocess_dataloader(dataloader, generator, device):
    all_states = []
    all_labels = []
    for state, label in dataloader:
        all_states.append(state)
        all_labels.append(label)

    state = torch.cat(all_states, dim=0).to(device)
    label = torch.cat(all_labels, dim=0).to(device)

    # 一次性生成特征
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

    auroc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for _ in range(10):
        model = HFSDModel(in_dim=6, hidden=128, layers=3, heads=4)
        trainer = HFSDTrainer(model, num_nodes, lr=lr, lambda_=lambda_, device=device)
        trainer.fit(train_loader, valid_loader, edge_index, epochs=epochs)
        auroc, precision, recall, f1 = trainer.evaluate(test_loader, edge_index)
        auroc_list.append(auroc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return (
        np.mean(auroc_list),
        np.mean(precision_list),
        np.mean(recall_list),
        np.mean(f1_list),
    )


def train_cascade(file_name: str, epochs=100, lr=1e-3, lambda_=0.05, device="cuda"):
    train_loader, valid_loader, test_loader, edge_index, num_nodes, _, _ = (
        get_true_cascade_dataset(file_name)
    )
    generator = HFSDFeatureGenerator(
        edge_index=edge_index.to(device), num_nodes=num_nodes
    )
    train_loader = preprocess_dataloader(train_loader, generator, device)
    valid_loader = preprocess_dataloader(valid_loader, generator, device)
    test_loader = preprocess_dataloader(test_loader, generator, device)

    auroc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for _ in range(2):
        model = HFSDModel(in_dim=6, hidden=128, layers=3, heads=4)
        trainer = HFSDTrainer(model, num_nodes, lr=lr, lambda_=lambda_, device=device)
        trainer.fit(train_loader, valid_loader, edge_index, epochs=epochs)
        auroc, precision, recall, f1 = trainer.evaluate(test_loader, edge_index)
        auroc_list.append(auroc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return (
        np.mean(auroc_list),
        np.mean(precision_list),
        np.mean(recall_list),
        np.mean(f1_list),
    )


def main():
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]
    result = {}

    for name in file_list:
        for type_ in type_list:
            auc, precision, recall, f1 = train_hfsd(
                file_name=name,
                type_=type_,
            )
            result[(name, type_)] = {
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    with open("result/HFSD.txt", "w") as f:
        f.write("Summary of all experiments using HFSD:\n")
        for (name, type_), metrics in result.items():
            f.write(f"{name} ({type_}): ")
            f.write(
                f"{name} ({type_}): "
                f"AUROC: {metrics["auc"]:.4f}, "
                f"Precision: {metrics["precision"]:.4f}, "
                f"Recall: {metrics["recall"]:.4f}, "
                f"F1: {metrics["f1"]:.4f}\n"
            )


def cascade_study():
    file_list = ["douban", "twitter"]
    result = {}

    for name in file_list:
        auc, precision, recall, f1 = train_cascade(
            file_name=name,
        )
        result[name] = {
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    with open("result/HFSD_cascade.txt", "w") as f:
        f.write("True cascade experiments using HFSD:\n")
        for name, metrics in result.items():
            f.write(
                f"{name}, "
                f"AUROC: {metrics["auc"]:.4f}, "
                f"Precision: {metrics["precision"]:.4f}, "
                f"Recall: {metrics["recall"]:.4f}, "
                f"F1: {metrics["f1"]:.4f}\n"
            )


if __name__ == "__main__":
    main()
