import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import init
from torch_geometric.utils import add_self_loops
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from utils import SEEDS, compute_stats, save_csv, load_data


class BackPropagation(nn.Module):
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

    def r2i(self, recovery: Tensor, infect: Tensor) -> tuple[Tensor, Tensor]:
        recovery = recovery - recovery * self.r_damp
        return recovery, infect

    def i2s(self, infect: Tensor, susceptible: Tensor) -> tuple[Tensor, Tensor]:
        infect = infect - infect * self.i_damp
        return infect, susceptible

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.required_self_loops is False:
            edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)

        if self.num_states == 3:
            s, i, r = x[:, :, 0], x[:, :, 1], x[:, :, 2]
            r, i = self.r2i(r, i)
            i, s = self.i2s(i, s)
            out = torch.stack((s, i, r), dim=2)
        else:
            s, i = x[:, :, 0], x[:, :, 1]
            i, s = self.i2s(i, s)
            out = torch.stack((s, i), dim=2)
        return out


class SL_IBPM(nn.Module):
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
            nn.Linear(num_states, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for layer in self.bp:
            x = layer(x, edge_index)
        return self.fc(x).squeeze(-1)


class BiasedEstimator(nn.Module):
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
        return loss


class IBPMTrainer:
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
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.estimator = BiasedEstimator(
            alpha=alpha, reduction=reduction, device=device
        )

    def train_step(self, train_loader, edge_index, mask=None) -> float:
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

    def valid_step(self, valid_loader, edge_index, mask=None) -> float:
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
        precision = precision_score(y_true, y_pred_bin, zero_division=0)
        recall = recall_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        return float(roc), float(precision), float(recall), float(f1)

    def fit(
        self, train_loader, valid_loader, test_loader, edge_index, epochs, mask=None
    ) -> dict:
        history = {
            "epoch": [],
            "train_loss": [],
            "valid_loss": [],
            "test_auc": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1": [],
        }
        for epoch in range(epochs):
            self.model.train()
            train_loss = self.train_step(train_loader, edge_index, mask)
            self.model.eval()
            valid_loss = self.valid_step(valid_loader, edge_index, mask)
            roc, precision, recall, f1 = self.test_step(test_loader, edge_index, mask)

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(train_loss)
            history["valid_loss"].append(valid_loss)
            history["test_auc"].append(roc)
            history["test_precision"].append(precision)
            history["test_recall"].append(recall)
            history["test_f1"].append(f1)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                    f"Valid Loss: {valid_loss:.4f}, Test F1: {f1:.4f}"
                )
        return history

    def evaluate(self, test_loader, edge_index, mask=None):
        self.model.eval()
        return self.test_step(test_loader, edge_index, mask)


# ===================== 跨seed聚合工具 =====================
def aggregate_histories(histories):
    if not histories:
        return {}
    epochs = len(histories[0]["epoch"])
    metrics = [
        "train_loss",
        "valid_loss",
        "test_auc",
        "test_precision",
        "test_recall",
        "test_f1",
    ]

    result = {"epoch": list(range(1, epochs + 1))}
    for metric in metrics:
        result[f"{metric}_mean"] = []
        result[f"{metric}_std"] = []
        for e in range(epochs):
            values = [h[metric][e] for h in histories]
            mean, std = compute_stats(values)
            result[f"{metric}_mean"].append(mean)  # type: ignore
            result[f"{metric}_std"].append(std)  # type: ignore
    return result


def save_history_csv(agg, save_path):
    import csv, os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss_mean",
        "train_loss_std",
        "valid_loss_mean",
        "valid_loss_std",
        "test_auc_mean",
        "test_auc_std",
        "test_precision_mean",
        "test_precision_std",
        "test_recall_mean",
        "test_recall_std",
        "test_f1_mean",
        "test_f1_std",
    ]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(agg["epoch"])):
            writer.writerow({k: agg[k][i] for k in fieldnames})


def main(
    lr: float = 0.001,
    reduction: str = "mean",
    alpha: float = 0.1,
    weight_decay: float = 0.0,
    device: str = "cuda:0",
    epochs: int = 100,
    history_dir: str = "result/ablation_history",
):
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]
    results = {}

    for name in file_list:
        for type_ in type_list:
            all_histories = []
            auc_list, precision_list, recall_list, f1_list = [], [], [], []
            for seed in SEEDS:
                (
                    train_loader,
                    valid_loader,
                    test_loader,
                    edge_index,
                    num_nodes,
                    num_edges,
                    num_states,
                ) = load_data(name, type_, seed=seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = SL_IBPM(num_nodes, num_edges, num_states)
                trainer = IBPMTrainer(
                    model,
                    lr=lr,
                    reduction=reduction,
                    alpha=alpha,
                    weight_decay=weight_decay,
                    device=device,
                )
                history = trainer.fit(
                    train_loader, valid_loader, test_loader, edge_index, epochs=epochs
                )
                all_histories.append(history)
                roc, precision, recall, f1 = trainer.evaluate(test_loader, edge_index)
                auc_list.append(roc)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

            agg = aggregate_histories(all_histories)
            save_history_csv(agg, f"{history_dir}/wo_aggr_{name}_{type_}.csv")

            results[(name, type_)] = {
                "auc_mean": compute_stats(auc_list)[0],
                "auc_std": compute_stats(auc_list)[1],
                "pre_mean": compute_stats(precision_list)[0],
                "pre_std": compute_stats(precision_list)[1],
                "rec_mean": compute_stats(recall_list)[0],
                "rec_std": compute_stats(recall_list)[1],
                "f1_mean": compute_stats(f1_list)[0],
                "f1_std": compute_stats(f1_list)[1],
            }
    save_csv(results, "result/wo_aggr.csv")


if __name__ == "__main__":
    main(epochs=100)
