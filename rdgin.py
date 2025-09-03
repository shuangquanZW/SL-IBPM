import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch_geometric.nn import GINConv
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from util import load_data, get_true_cascade_dataset


class RSDGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.preprocess = nn.Linear(input_dim + 4, hidden_dim)

        self.gin_convs = nn.ModuleList(
            [
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
                for _ in range(2)
            ]
        )

        self.centrality_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        # 新增 dropout 与 BN，使训练更稳定
        self.source_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index):
        batch_size, num_nodes, _ = x.size()

        # 展平为 2D
        x_flat = x.view(-1, x.size(-1))  # (B*N, F)
        S = x_flat[:, 1]  # 状态特征
        T = self._state_encoding(S)  # (B*N, 4)
        x_flat = torch.cat([x_flat, T], dim=1)
        x_flat = F.relu(self.preprocess(x_flat))  # (B*N, H)

        # GIN
        for conv in self.gin_convs:
            x_flat = F.relu(conv(x_flat, edge_index))
        Em = x_flat

        # 谣言源概率：对所有节点直接输出
        probs = self.source_mlp(Em).squeeze()  # (B*N,)
        return probs.view(batch_size, num_nodes)

    def _state_encoding(self, S):
        """把 0/1 状态编码为 4 维"""
        T1 = torch.where(S == 1, 1.0, -1.0)
        T2 = torch.where(S == 1, 1.0, 0.0)
        T3 = torch.where(S == 0, 0.0, -1.0)
        return torch.stack([S, T1, T2, T3], dim=1)


class RSDGINTrainer:
    def __init__(self, model: RSDGIN, lr: float, reduction: str, device: str):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.estimator = nn.BCELoss(reduction=reduction)

    def train_step(
        self,
        train_loader: DataLoader,
        edge_index: Tensor,
        mask: list | None = None,
    ) -> float:
        edge_index = edge_index.to(self.device)
        total_loss = 0.0
        self.model.train()
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

    def valid_step(
        self,
        valid_loader: DataLoader,
        edge_index: Tensor,
        mask: list | None = None,
    ) -> float:
        edge_index = edge_index.to(self.device)
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for x, true in valid_loader:
                x, true = x.to(self.device), true.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                pred = self.model(x, edge_index)
                loss = self.estimator(pred, true)
                total_loss += loss.item()
        return total_loss / len(valid_loader)

    def test_step(
        self,
        test_loader: DataLoader,
        edge_index: Tensor,
        mask: list | None = None,
    ):
        edge_index = edge_index.to(self.device)
        auroc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        self.model.eval()
        with torch.no_grad():
            for x, true in test_loader:
                x, true = x.to(self.device), true.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                pred = self.model(x, edge_index)
                y_hat = pred.reshape(-1).cpu().numpy()
                y = true.reshape(-1).cpu().numpy()
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
        train_loader: DataLoader,
        valid_loader: DataLoader,
        edge_index: Tensor,
        epochs: int,
        mask: list | None = None,
    ) -> None:
        for epoch in range(epochs):
            train_loss = self.train_step(train_loader, edge_index, mask)
            valid_loss = self.valid_step(valid_loader, edge_index, mask)
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}"
            )

    def evaluate(
        self,
        test_loader: DataLoader,
        edge_index,
        mask: list | None = None,
    ):
        return self.test_step(test_loader, edge_index, mask)


def train_rdgin(
    file_name: str,
    type_: str,
    epochs: int = 100,
    lr: float = 0.001,
    reduction: str = "mean",
    device: str = "cuda:0",
):
    train_loader, valid_loader, test_loader, edge_index, _, _, num_states = load_data(
        file_name, type_
    )

    auroc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for run in range(10):
        torch.manual_seed(run)
        model = RSDGIN(num_states, 256)
        trainer = RSDGINTrainer(model, lr, reduction, device)
        trainer.fit(train_loader, valid_loader, edge_index, epochs)
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


def train_cascade(
    file_name: str,
    epochs: int = 100,
    lr: float = 0.001,
    reduction: str = "mean",
    device: str = "cuda:0",
):
    train_loader, valid_loader, test_loader, edge_index, _, _, num_states = (
        get_true_cascade_dataset(file_name)
    )

    auroc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for run in range(2):
        torch.manual_seed(run)
        model = RSDGIN(num_states, 256)
        trainer = RSDGINTrainer(model, lr, reduction, device)
        trainer.fit(train_loader, valid_loader, edge_index, epochs)
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

    with open("./RDGIN_cascade.txt", "w") as f:
        f.write("True cascade experiments using RDGIN:\n")
        for name, metrics in result.items():
            f.write(
                f"{name}: "
                f"AUROC: {metrics["auc"]:.4f}, "
                f"Precision: {metrics["precision"]:.4f}, "
                f"Recall: {metrics["recall"]:.4f}, "
                f"F1: {metrics["f1"]:.4f}\n"
            )


def main():
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]
    result = {}

    for name in file_list:
        for type_ in type_list:
            auc, precision, recall, f1 = train_rdgin(
                file_name=name,
                type_=type_,
            )
            result[(name, type_)] = {
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    with open("result/RSDGIN.txt", "w") as f:
        f.write("Summary of all experiments using RSDGIN:\n")
        for (name, type_), metrics in result.items():
            f.write(f"{name} ({type_}): ")
            f.write(
                f"{name} ({type_}): "
                f"AUROC: {metrics["auc"]:.4f}, "
                f"Precision: {metrics["precision"]:.4f}, "
                f"Recall: {metrics["recall"]:.4f}, "
                f"F1: {metrics["f1"]:.4f}\n"
            )


if __name__ == "__main__":
    main()
