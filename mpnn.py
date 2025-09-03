import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from util import load_data, get_true_cascade_dataset


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 论文中GCN层无自环
        self.gcn = GCNConv(in_dim, out_dim, add_self_loops=False)
        # 使用LeakyReLU激活函数
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, edge_index):
        # x: (batch, num_nodes, in_dim)
        origin = x
        batch_size, num_nodes, in_dim = x.shape

        # 处理batch维度
        x = x.view(-1, in_dim)  # (batch*num_nodes, in_dim)
        x = self.gcn(x, edge_index)  # (batch*num_nodes, out_dim)
        x = x.view(batch_size, num_nodes, -1)  # (batch, num_nodes, out_dim)
        x = self.activation(x)
        return x + origin  # 残差连接


class ResGCN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        out_dim=1,
        num_layers=6,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        # x: (batch, num_nodes, in_dim)
        x = self.input_proj(x)  # (batch, num_nodes, hidden_dim)
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.output_proj(x).squeeze(-1)  # (batch, num_nodes)


class ResGCNTrainer:
    def __init__(self, model: ResGCN, lr: float, device: str):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_step(
        self,
        train_loader: DataLoader,
        edge_index,
        mask: list | None = None,
    ):
        edge_index = edge_index.to(self.device)
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            if mask:
                x[:, mask, :] = 0
            self.optimizer.zero_grad()
            y_hat = self.model(x, edge_index)
            loss = self.criterion(y_hat, y.float())
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def valid_step(
        self,
        valid_loader: DataLoader,
        edge_index,
        mask: list | None = None,
    ):
        edge_index = edge_index.to(self.device)
        total_loss = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                y_hat = self.model(x, edge_index)
                loss = self.criterion(y_hat, y.float())
                total_loss += loss.item()
        return total_loss / len(valid_loader)

    def test_step(
        self,
        test_loader: DataLoader,
        edge_index,
        mask: list | None = None,
    ):
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
                y_hat = self.model(x, edge_index).cpu().numpy()
                y = y.cpu().numpy()

                # 对batch中的每个样本单独处理
                for i in range(y.shape[0]):
                    y_true = y[i].ravel()
                    y_pred = y_hat[i].ravel()

                    if np.unique(y_true).size < 2:
                        continue

                    try:
                        auroc = roc_auc_score(y_true, y_pred)
                        y_pred = np.where(y_pred > 0.5, 1, 0)
                        precision = precision_score(y_true, y_pred)
                        recall = recall_score(y_true, y_pred)
                        f1 = f1_score(y_true, y_pred)
                        auroc_list.append(auroc)
                        precision_list.append(precision)
                        recall_list.append(recall)
                        f1_list.append(f1)
                    except ValueError:
                        continue

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
        epochs,
        mask: list | None = None,
    ):
        edge_index = edge_index.to(self.device)
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
        test_loader,
        edge_index,
        mask: list | None = None,
    ):
        edge_index = edge_index.to(self.device)
        self.model.eval()
        auroc, precision, recall, f1 = self.test_step(test_loader, edge_index, mask)
        return auroc, precision, recall, f1


def train_resgcn(
    file_name: str,
    type_: str,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cuda:0",
):
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        _,
        _,
        num_states,
    ) = load_data(file_name, type_)

    auroc_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    # 运行10次取平均
    for _ in range(10):
        model = ResGCN(in_dim=num_states, hidden_dim=128, out_dim=1, num_layers=5)
        trainer = ResGCNTrainer(
            model=model,
            lr=lr,
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
    device: str = "cuda:0",
):
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        _,
        _,
        num_states,
    ) = get_true_cascade_dataset(file_name)

    auroc_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    # 运行10次取平均
    for _ in range(2):
        model = ResGCN(in_dim=num_states, hidden_dim=128, out_dim=1, num_layers=5)
        trainer = ResGCNTrainer(
            model=model,
            lr=lr,
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
            auc, precision, recall, f1 = train_resgcn(
                file_name=name,
                type_=type_,
            )
            result[(name, type_)] = {
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    with open("result/MPNN.txt", "w") as f:
        f.write("Summary of all experiments using MPNN:\n")
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

    with open("./MPNN_cascade.txt", "w") as f:
        f.write("True cascade experiments using MPNN:\n")
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
