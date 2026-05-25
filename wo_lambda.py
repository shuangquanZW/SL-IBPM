import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import init
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from cascade_experiment import run_cascade_experiment, run_epoch_sweep
from utils import compute_stats


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


class IBPMTrainer:
    def __init__(
        self,
        model: SL_IBPM,
        lr: float,
        weight_decay: float,
        device: str,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.estimator = nn.BCEWithLogitsLoss()

    def train_step(self, train_loader, edge_index, mask=None) -> float:
        edge_index = edge_index.to(self.device)
        total_loss = 0.0
        for x, true in train_loader:
            x, true = x.to(self.device), true.to(self.device)
            if mask:
                x[:, mask, :] = 0
            self.optimizer.zero_grad()
            pred = self.model(x, edge_index)
            loss = self.estimator(pred, true.float())
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
                loss = self.estimator(pred, true.float())
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
    ):
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
    weight_decay: float = 0.0,
    device: str = "cuda:0",
    epochs: int = 100,
    output_csv: str | None = None,
    history_dir: str | None = None,
):
    result_key = "wo_lambda"
    output_csv = output_csv or f"result/{result_key}_cascade_e{epochs}.csv"
    history_dir = history_dir or f"result/ablation_history_cascade_e{epochs}"
    return run_cascade_experiment(
        model_factory=lambda num_nodes, num_edges, num_states: SL_IBPM(
            num_nodes, num_edges, num_states
        ),
        trainer_factory=lambda model: IBPMTrainer(
            model,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        ),
        aggregate_histories=aggregate_histories,
        save_history_csv=save_history_csv,
        result_key=result_key,
        output_csv=output_csv,
        epochs=epochs,
        history_dir=history_dir,
    )


if __name__ == "__main__":
    run_epoch_sweep(main)
