import time
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import pickle as pkl
from scipy.sparse import diags
from scipy.sparse.linalg import inv

from util import load_data, calculate_stats


class InputGenerator:
    def __init__(self, edge_index, num_nodes, alpha=0.5, device="cuda:0"):
        """
        edge_index: (2, E) tensor -> 构建稀疏邻接矩阵
        """
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.device = device
        # 构造对称归一化邻接矩阵 S
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).astype(float)
        adj = adj.maximum(adj.T)  # 无向图
        deg = np.array(adj.sum(1)).flatten()
        deg_inv_sqrt = diags(np.power(np.maximum(deg, 1.0), -0.5))
        self.S = deg_inv_sqrt @ adj @ deg_inv_sqrt
        self.Minv = None

    def forward(self, state):
        """
        state: (B, N, S)  -> 取第一维作为 Y
        return: (B, N, 4) 四维特征
        """
        Y = 1 - state[..., 0].cpu().numpy()  # (B, N)
        if self.Minv is None:
            N = Y.shape[1]
            I = diags(np.ones(N))
            M = I - self.alpha * self.S
            Minv = inv(M.tocsc())
            self.Minv = Minv
        # 向量化处理整个批次
        d1 = Y
        v3 = np.where(Y == 1, 1, 0)
        v4 = np.where(Y == -1, 1, 0)

        d2 = (1 - self.alpha) * (self.Minv @ Y.T).T
        d3 = (1 - self.alpha) * (self.Minv @ v3.T).T
        d4 = (1 - self.alpha) * (self.Minv @ v4.T).T

        feat = np.stack([d1, d2, d3, d4], axis=2)  # (B, N, 4)
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
        # x: (B, N, 4) -> reshape to (B*N, 4)
        B, N, _ = x.shape
        x = x.view(B * N, 4)
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index))
        x = self.dense(x)  # (B*N, 1)
        return torch.sigmoid(x.view(B, N))  # (B, N)


class GCNSITrainer:

    def __init__(
        self, model: GCNSI, num_nodes: int, lr: float, reduction: str, device: str
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.estimator = nn.BCELoss(reduction=reduction)
        self.num_nodes = num_nodes

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

        with torch.no_grad():
            for x, true in test_loader:
                x, true = x.to(self.device), true.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                pred = self.model(x, edge_index)
                pred_np = pred.reshape(-1).cpu().numpy()
                true_np = true.reshape(-1).cpu().numpy()

                auroc_list.append(roc_auc_score(true_np, pred_np))
                # 计算二值预测
                pred_binary = (pred_np > 0.5).astype(int)

                precision_list.append(precision_score(true_np, pred_binary))
                recall_list.append(recall_score(true_np, pred_binary))
                f1_list.append(f1_score(true_np, pred_binary))

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
        edge_index: Tensor,
        mask: list | None = None,
    ):
        """评估模型"""
        self.model.eval()
        auc, pre, rec, f1 = self.test_step(test_loader, edge_index, mask)
        return auc, pre, rec, f1


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


def train_gcnsi(
    file_name: str,
    type_: str,
    epochs: int = 100,
    lr: float = 0.001,
    reduction: str = "mean",
    device: str = "cuda:0",
):
    """训练GCNSI模型"""
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

    time_list = []
    auc_list = []
    pre_list = []
    rec_list = []
    f1_list = []
    for _ in range(10):
        start_time = time.time()
        model = GCNSI()
        trainer = GCNSITrainer(
            model=model,
            num_nodes=num_nodes,
            lr=lr,
            reduction=reduction,
            device=device,
        )
        trainer.fit(train_loader, valid_loader, edge_index, epochs)
        auc, pre, rec, f1 = trainer.evaluate(test_loader, edge_index)
        end_time = time.time()
        time_list.append(end_time - start_time)
        auc_list.append(auc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)

    avg_time = sum(time_list[1:]) / (10 - 1)

    return (
        avg_time,
        np.mean(auc_list),
        np.mean(pre_list),
        np.mean(rec_list),
        np.mean(f1_list),
    )


def main():
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]

    with open("result/GCNSI.txt", "w") as f:
        f.write("GCNSI Algorithm Results\n")
        f.write("-" * 70 + "\n")

        for name in file_list:
            for type_ in type_list:
                avg_time, auc, pre, rec, f1 = train_gcnsi(
                    file_name=name,
                    type_=type_,
                )
                f.write(f"{name} ({type_}): ")
                f.write(
                    f"{name} ({type_}): "
                    f"Time: {avg_time:.4f}s, "
                    f"AUROC: {auc:.4f}, "
                    f"Precision: {pre:.4f}, "
                    f"Recall: {rec:.4f}, "
                    f"F1: {f1:.4f}\n"
                )
                f.write("-" * 50 + "\n")


def partial():
    file_list = ["karate", "jazz", "net_science", "cora_ml"]
    type_ = "SIR"
    result = {}
    device = "cuda:0"

    for name in file_list:
        (
            train_loader,
            valid_loader,
            test_loader,
            edge_index,
            num_nodes,
            _,
            _,
        ) = load_data(name, type_)
        generator = InputGenerator(
            edge_index=edge_index.to(device), num_nodes=num_nodes
        )
        train_loader = preprocess_dataloader(train_loader, generator, device)
        valid_loader = preprocess_dataloader(valid_loader, generator, device)
        test_loader = preprocess_dataloader(test_loader, generator, device)
        with open(f"data/{type_}/{name}/dict_observe.pkl", "rb") as f:
            dict_observe = pkl.load(f)
        for i in range(1, 10):
            observe = list(dict_observe[i])
            roc_list = []
            for _ in range(4):
                model = GCNSI()
                trainer = GCNSITrainer(
                    model=model,
                    num_nodes=num_nodes,
                    lr=0.001,
                    reduction="mean",
                    device="cuda:0",
                )
                trainer.fit(
                    train_loader,
                    valid_loader,
                    edge_index,
                    100,
                    mask=observe,
                )
                auroc = trainer.evaluate(
                    test_loader,
                    edge_index,
                    mask=observe,
                )[0]
                roc_list.append(auroc)
            roc_avg, _ = calculate_stats(roc_list)
            result[(name, i)] = roc_avg

    print("\nExperiments with partial observation:")
    for (name, i), roc in result.items():
        print(f"{name} (Mask {i}0%): AUROC: {roc:.4f}")


if __name__ == "__main__":
    main()
    # partial()
