import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from util import load_data, get_true_cascade_dataset


class InvertibleGraphResidualNet(nn.Module):
    """
    可逆图残差网络，确保 Lipschitz 常数 < 1
    """

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),  # 保证 Lipschitz < 1
            nn.Linear(hidden, in_dim),
        )
        self.g1 = GCNConv(in_dim, hidden)
        self.g2 = GCNConv(hidden, in_dim)
        self._lipschitz_clip()

    def _lipschitz_clip(self):
        """简单裁剪权重，近似 Lipschitz 约束"""
        for m in self.f:
            if isinstance(m, nn.Linear):
                nn.utils.spectral_norm(m)
        for m in [self.g1, self.g2]:
            if isinstance(m, GCNConv):
                nn.utils.spectral_norm(m.lin)

    def forward(self, y, edge_index, iters=3):
        """
        固定点迭代求逆
        y = (g(z) + z) / 2  =>  z = 2y - g(z)
        x = 2z - f(x)
        """
        z = y
        for _ in range(iters):
            z = 2 * y - torch.tanh(self.g2(self.g1(z, edge_index), edge_index))
        x = z
        for _ in range(iters):
            x = 2 * z - self.f(x)
        return x


class ErrorCompensator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return z + self.net(z) - 0.5


class IVGD(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.invert_nets = nn.ModuleList(
            [InvertibleGraphResidualNet(in_dim) for _ in range(2)]
        )
        self.compensators = nn.ModuleList([ErrorCompensator(in_dim) for _ in range(2)])
        self.predict = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, y_T, edge_index):
        for l in self.invert_nets:
            y_T = l(y_T, edge_index)
        z = y_T
        for l in self.compensators:
            z = l(z)
        x_hat = z
        return self.predict(x_hat).squeeze(-1)


class IVGDTrainer:

    def __init__(self, model: IVGD, lr: float, reduction: str, device: str):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.estimator = nn.BCELoss(reduction=reduction)

    def train_step(
        self,
        train_loader: DataLoader,
        edge_index,
        mask: list | None = None,
    ) -> float:
        """训练步"""
        edge_index = edge_index.to(self.device)
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            if mask:
                x[:, mask, :] = 0
            self.optimizer.zero_grad()
            y_hat = self.model(x, edge_index)
            loss = self.estimator(y_hat, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def valid_step(
        self,
        valid_loader: DataLoader,
        edge_index,
        mask: list | None = None,
    ) -> float:
        """验证步"""
        edge_index = edge_index.to(self.device)
        total_loss = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                y_hat = self.model(x, edge_index)
                loss = self.estimator(y_hat, y)
                total_loss += loss.item()
        return total_loss / len(valid_loader)

    def test_step(
        self,
        test_loader: DataLoader,
        edge_index,
        mask: list | None = None,
    ):
        """测试步"""
        edge_index = edge_index.to(self.device)
        auroc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                y_hat = self.model(x, edge_index)
                y_hat = y_hat.reshape(-1).cpu().numpy()
                y = y.reshape(-1).cpu().numpy()
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
        edge_index,
        epochs: int,
        mask: list | None = None,
    ) -> None:
        for epoch in range(epochs):
            self.model.train()
            train_loss = self.train_step(train_loader, edge_index, mask)
            self.model.eval()
            valid_loss = self.valid_step(valid_loader, edge_index, mask)
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
            )

    def evaluate(
        self,
        test_loader: DataLoader,
        edge_index,
        mask: list | None = None,
    ):
        """评估模型"""
        self.model.eval()
        return self.test_step(test_loader, edge_index, mask)


def train_ivgd(
    file_name: str,
    type_: str,
    epochs: int = 100,
    lr: float = 0.001,
    reduction: str = "mean",
    device: str = "cuda:0",
):
    """训练IVGD模型"""
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        _,
        _,
        states,
    ) = load_data(file_name, type_)

    auroc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for _ in range(10):
        model = IVGD(states)
        trainer = IVGDTrainer(
            model=model,
            lr=lr,
            reduction=reduction,
            device=device,
        )
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
    """训练IVGD模型"""
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        _,
        _,
        states,
    ) = get_true_cascade_dataset(file_name)

    auroc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for _ in range(2):
        model = IVGD(states)
        trainer = IVGDTrainer(
            model=model,
            lr=lr,
            reduction=reduction,
            device=device,
        )
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


def main():
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]
    result = {}

    for name in file_list:
        for type_ in type_list:
            auc, precision, recall, f1 = train_ivgd(
                file_name=name,
                type_=type_,
            )
            result[(name, type_)] = {
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    with open("result/IVGD.txt", "w") as f:
        f.write("Summary of all experiments using IVGD:\n")
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

    with open("result/IVGD_cascade.txt", "w") as f:
        f.write("True cascade experiments using IVGD:\n")
        for name, metrics in result.items():
            f.write(f"{name}: ")
            f.write(
                f"{name}: "
                f"AUROC: {metrics["auc"]:.4f}, "
                f"Precision: {metrics["precision"]:.4f}, "
                f"Recall: {metrics["recall"]:.4f}, "
                f"F1: {metrics["f1"]:.4f}\n"
            )


if __name__ == "__main__":
    main()
