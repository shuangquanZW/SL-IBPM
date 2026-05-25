import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)
from cascade_experiment import run_cascade_experiment, run_epoch_sweep
from utils import compute_stats


class BiasedEstimator(nn.Module):
    def __init__(self, alpha: float, reduction: str, device: str) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        num_neg = (true == 0).sum().float()
        num_pos = (true == 1).sum().float() + 1e-8
        ratio = num_neg / num_pos
        weight = torch.where(
            true == 1,
            ratio.expand_as(pred),
            torch.ones_like(pred),
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred, true.float(), weight=weight, reduction="none"
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class Plain_GCN(nn.Module):
    def __init__(self, num_nodes, num_edges, num_states):
        super().__init__()
        self.conv1 = GCNConv(num_states, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index):
        B, N, F = x.shape
        x = x.view(B * N, F)
        edge_index_batch = edge_index.clone()
        for b in range(1, B):
            edge_index_batch = torch.cat([edge_index_batch, edge_index + b * N], dim=1)

        h = self.conv1(x, edge_index_batch)
        h = torch.nn.functional.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index_batch)
        h = torch.nn.functional.relu(h)
        h = self.dropout(h)

        h = h.view(B, N, -1)
        out = self.predictor(h)
        return out.squeeze(-1)


class GCNTrainer:
    def __init__(self, model, lr, weight_decay, alpha, reduction, device):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.estimator = BiasedEstimator(
            alpha=alpha, reduction=reduction, device=device
        )

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
    lr=0.001,
    reduction="mean",
    alpha=0.1,
    weight_decay=0.0,
    device="cuda:2",
    epochs=100,
    output_csv=None,
    history_dir=None,
):
    result_key = "plain_gnn"
    output_csv = output_csv or f"result/{result_key}_cascade_e{epochs}.csv"
    history_dir = history_dir or f"result/ablation_history_cascade_e{epochs}"
    return run_cascade_experiment(
        model_factory=lambda num_nodes, num_edges, num_states: Plain_GCN(
            num_nodes, num_edges, num_states
        ),
        trainer_factory=lambda model: GCNTrainer(
            model,
            lr=lr,
            weight_decay=weight_decay,
            alpha=alpha,
            reduction=reduction,
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
