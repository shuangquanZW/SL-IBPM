import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import init
from torch.utils.data import DataLoader
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from util import load_data


class BackPropagation(nn.Module):
    """反向传播模块（无信息衰减）"""

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        num_states: int = 3,
        required_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_states = num_states
        self.required_self_loops = required_self_loops

        feature_size = num_edges + num_nodes if required_self_loops else num_edges
        self.linear_weight = nn.Parameter(torch.empty((2, feature_size)))
        self.r_damp = nn.Parameter(torch.empty(num_nodes))
        self.i_damp = nn.Parameter(torch.empty(num_nodes))
        self.epsilon = nn.Parameter(torch.empty(2, num_nodes))
        self.init_parameters()

    def init_parameters(self) -> None:
        init.xavier_normal_(self.linear_weight)
        init.normal_(self.r_damp)
        init.normal_(self.i_damp)
        init.xavier_normal_(self.epsilon)

    def r2i(
        self, recovery: Tensor, infect: Tensor, src: Tensor, dst: Tensor
    ) -> tuple[Tensor, Tensor]:
        r_i_delta = recovery[:, src] * self.linear_weight[0]
        r_i_delta = scatter(r_i_delta, dst, reduce="sum") + self.epsilon[0]
        r_i_delta = torch.relu(infect * r_i_delta)
        return recovery, r_i_delta + infect

    def i2s(
        self, infect: Tensor, susceptible: Tensor, src: Tensor, dst: Tensor
    ) -> tuple[Tensor, Tensor]:
        i_s_delta = infect[:, src] * self.linear_weight[1]
        i_s_delta = scatter(i_s_delta, dst, reduce="sum") + self.epsilon[1]
        i_s_delta = torch.relu(susceptible * i_s_delta)
        return infect, i_s_delta + susceptible

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.required_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        src, dst = edge_index

        if self.num_states == 3:
            s, i, r = x[:, :, 0], x[:, :, 1], x[:, :, 2]
            r, i = self.r2i(r, i, src, dst)
            i, s = self.i2s(i, s, src, dst)
            out = torch.stack((s, i, r), dim=2)
        else:
            s, i = x[:, :, 0], x[:, :, 1]
            i, s = self.i2s(i, s, src, dst)
            out = torch.stack((s, i), dim=2)
        return out


class SL_IBPM(nn.Module):
    """源定位模型"""

    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        num_states: int = 3,
        required_self_loops: bool = True,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.bp = nn.ModuleList(
            [
                BackPropagation(num_nodes, num_edges, num_states, required_self_loops)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_nodes),
            nn.Linear(num_states, 32),
            nn.Linear(32, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for layer in self.bp:
            x = layer(x, edge_index)
        return self.fc(x).squeeze(-1)


class BiasedEstimator(nn.Module):
    """有偏估计器"""

    def __init__(self, alpha: float, reduction: str, device: str) -> None:
        super().__init__()
        self.pos_weight = torch.tensor([1 / alpha]).to(device)
        self.reduction = reduction
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight, reduction=reduction
        )

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        loss = self.criterion(pred, true.float())
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class IBPMTrainer:
    """信息反向传播机理训练器"""

    def __init__(
        self,
        model: SL_IBPM,
        lr: float,
        weight_decay: float,
        alpha: float,
        reduction: str,
        device: str,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.estimator = BiasedEstimator(
            alpha=alpha, reduction=reduction, device=device
        )

    def train_step(
        self,
        train_loader: DataLoader,
        edge_index: Tensor,
        mask: list | None = None,
    ) -> float:
        """训练步"""
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

    def valid_step(
        self,
        valid_loader: DataLoader,
        edge_index: Tensor,
        mask: list | None = None,
    ) -> float:
        """验证步"""
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

    def test_step(
        self, test_loader: DataLoader, edge_index: Tensor, mask: list | None = None
    ) -> tuple[float, float, float, float]:
        self.model.eval()
        y_true_all, y_pred_all = [], []
        edge_index = edge_index.to(self.device)
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                pred = torch.sigmoid(self.model(x, edge_index))
                y_true_all.append(y.cpu().numpy())
                y_pred_all.append(pred.cpu().numpy())
        y_true = np.concatenate(y_true_all).ravel()
        y_pred = np.concatenate(y_pred_all).ravel()
        roc = roc_auc_score(y_true, y_pred)
        y_pred_bin = (y_pred > 0.5).astype(int)
        precision = precision_score(y_true, y_pred_bin)
        recall = recall_score(y_true, y_pred_bin)
        f1 = f1_score(y_true, y_pred_bin)
        return float(roc), float(precision), float(recall), float(f1)

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        edge_index: Tensor,
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

    def get_loss(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        edge_index: Tensor,
        epochs: int,
        mask: list | None = None,
    ) -> tuple[list[float], list[float]]:
        train_loss_list = []
        valid_loss_list = []
        for epoch in range(epochs):
            self.model.train()
            train_loss = self.train_step(train_loader, edge_index, mask)
            self.model.eval()
            valid_loss = self.valid_step(valid_loader, edge_index, mask)
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
            )
            if epoch % 10 == 0:
                train_loss_list.append(train_loss)
                valid_loss_list.append(valid_loss)
        return train_loss_list, valid_loss_list

    def evaluate(
        self,
        test_loader: DataLoader,
        edge_index: Tensor,
        mask: list | None = None,
    ) -> tuple[float, float, float, float]:
        """评估模型"""
        self.model.eval()
        roc, precision, recall, f1 = self.test_step(test_loader, edge_index, mask)
        return roc, precision, recall, f1


def main(
    lr: float = 0.001,
    reduction: str = "mean",
    alpha: float = 1.0,
    weight_decay: float = 0.0,
    device: str = "cuda:0",
):
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]

    with open("result/wo_lambda.txt", "w") as f:
        f.write("wo_lambda Results (AUROC, Precision, Recall, F1)\n")
        f.write("-" * 70 + "\n")
        for name in file_list:
            for type_ in type_list:
                (
                    train_loader,
                    valid_loader,
                    test_loader,
                    edge_index,
                    num_nodes,
                    num_edges,
                    num_states,
                ) = load_data(name, type_)
                auc_list = []
                precision_list = []
                recall_list = []
                f1_list = []
                for _ in range(5):
                    model = SL_IBPM(num_nodes, num_edges, num_states)
                    trainer = IBPMTrainer(
                        model,
                        lr=lr,
                        reduction=reduction,
                        alpha=alpha,
                        weight_decay=weight_decay,
                        device=device,
                    )
                    trainer.fit(train_loader, valid_loader, edge_index, 20)
                    roc, precision, recall, f1 = trainer.test_step(
                        test_loader, edge_index
                    )
                    auc_list.append(roc)
                    precision_list.append(precision)
                    recall_list.append(recall)
                    f1_list.append(f1)
                f.write(
                    f"{name} ({type_}) | "
                    f"AUROC: {np.mean(auc_list):.4f}, Precision: {np.mean(precision_list):.4f}, "
                    f"Recall: {np.mean(recall_list):.4f}, F1: {np.mean(f1_list):.4f}\n"
                )
                f.write("-" * 70 + "\n")


if __name__ == "__main__":
    main()
