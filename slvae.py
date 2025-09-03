import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torch_scatter import scatter

from util import load_data


# --------------------------------------------------
# 1. 网络结构
# --------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128, z_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hid, z_dim)
        self.logvar = nn.Linear(hid, z_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, n_nodes: int, z_dim: int = 32):
        super().__init__()
        self.enc = Encoder(n_nodes, z_dim=z_dim)
        self.dec = Decoder(z_dim, n_nodes)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


class DiffusionPropagate(nn.Module):
    def __init__(self, edge_index: torch.Tensor, n_iter: int = 2):
        super().__init__()
        self.n_iter = n_iter
        self.edge_index = edge_index.to("cuda")

    def forward(self, p0: torch.Tensor):
        src, dst = self.edge_index
        for _ in range(self.n_iter):
            p0 = scatter(p0[:, src], dst, dim=1, reduce="mean")
        return torch.sigmoid(p0)


class GNN(nn.Module):
    """
    把 VAE 重构出的 x_hat 进一步映射到节点级概率
    """

    def __init__(self, n_nodes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_nodes, 128), nn.ReLU(), nn.Linear(128, n_nodes), nn.Sigmoid()
        )

    def forward(self, x_hat):
        return self.mlp(x_hat)


class SLVAE(nn.Module):
    """
    完整模型：VAE + GNN + Diffusion
    """

    def __init__(self, n_nodes: int, edge_index: torch.Tensor):
        super().__init__()
        self.vae = VAE(n_nodes)
        self.gnn = GNN(n_nodes)
        self.diffusion = DiffusionPropagate(edge_index)

    def forward(self, x):
        # x : (batch, n_nodes, 1)
        x = 1 - x[..., 0]  # (batch, n_nodes)
        x_hat, mu, logvar = self.vae(x)
        p0 = self.gnn(x_hat)  # (batch, n_nodes)
        y_hat = self.diffusion(p0)  # (batch, n_nodes)
        return x_hat, mu, logvar, y_hat

    @staticmethod
    def loss(x, x_hat, mu, logvar, y_hat, y):
        x = 1 - x[..., 0]
        recon = F.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        fwd = F.mse_loss(y_hat, y, reduction="sum")
        return recon + kl + fwd


class SLVAETrainer:

    def __init__(self, model: SLVAE, lr: float, device: str):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.estimator = model.loss

    def train_step(
        self,
        train_loader: DataLoader,
        mask: list | None = None,
    ) -> float:
        """训练步"""
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            if mask:
                x[:, mask, :] = 0
            self.optimizer.zero_grad()
            x_hat, mu, logvar, y_hat = self.model(x)
            loss = self.estimator(x, x_hat, mu, logvar, y_hat, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def valid_step(
        self,
        valid_loader: DataLoader,
        mask: list | None = None,
    ) -> float:
        """验证步"""
        total_loss = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                x_hat, mu, logvar, y_hat = self.model(x)
                loss = self.estimator(x, x_hat, mu, logvar, y_hat, y)
                total_loss += loss.item()
        return total_loss / len(valid_loader)

    def test_step(
        self,
        test_loader: DataLoader,
        mask: list | None = None,
    ):
        """测试步"""
        auroc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                _, _, _, y_hat = self.model(x)
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
        epochs: int,
        mask: list | None = None,
    ) -> None:
        for epoch in range(epochs):
            self.model.train()
            train_loss = self.train_step(train_loader, mask)
            self.model.eval()
            valid_loss = self.valid_step(valid_loader, mask)
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
            )

    def evaluate(
        self,
        test_loader: DataLoader,
        mask: list | None = None,
    ):
        """评估模型"""
        self.model.eval()
        auroc, precision, recall, f1 = self.test_step(test_loader, mask)
        return auroc, precision, recall, f1


def train_slvae(
    file_name: str,
    type_: str,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = "cuda:0",
):
    """训练SLVAE模型"""
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        num_nodes,
        _,
        _,
    ) = load_data(file_name, type_)

    auroc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for _ in range(4):
        model = SLVAE(num_nodes, edge_index)
        trainer = SLVAETrainer(
            model=model,
            lr=lr,
            device=device,
        )
        trainer.fit(train_loader, valid_loader, epochs)
        auroc, precision, recall, f1 = trainer.evaluate(test_loader)
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


# def train_cascade(
#     file_name: str,
#     epochs: int = 100,
#     lr: float = 0.001,
#     device: str = "cuda:0",
# ):
#     """训练SLVAE模型"""
#     (
#         train_loader,
#         valid_loader,
#         test_loader,
#         edge_index,
#         num_nodes,
#         _,
#         _,
#     ) = get_true_cascade_dataset(file_name)

#     time_list = []
#     roc_list = []
#     prc_list = []
#     for _ in range(4):
#         start_time = time.time()
#         model = SLVAE(num_nodes, edge_index)
#         trainer = SLVAETrainer(
#             model=model,
#             lr=lr,
#             device=device,
#         )
#         trainer.fit(train_loader, valid_loader, epochs)
#         auroc, auprc = trainer.evaluate(test_loader)
#         end_time = time.time()
#         time_list.append(end_time - start_time)
#         roc_list.append(auroc)
#         prc_list.append(auprc)

#     avg_time = sum(time_list[1:]) / (4 - 1)
#     roc_avg, roc_std = calculate_stats(roc_list)
#     prc_avg, prc_std = calculate_stats(prc_list)

#     return avg_time, roc_avg, roc_std, prc_avg, prc_std


def main():
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]
    result = {}

    for name in file_list:
        for type_ in type_list:
            auc, precision, recall, f1 = train_slvae(
                file_name=name,
                type_=type_,
            )
            result[(name, type_)] = {
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    with open("result/SLVAE.txt", "w") as f:
        f.write("Summary of all experiments using SLVAE:\n")
        for (name, type_), metrics in result.items():
            f.write(f"{name} ({type_}): ")
            f.write(
                f"{name} ({type_}): "
                f"AUROC: {metrics["auc"]:.4f}, "
                f"Precision: {metrics["precision"]:.4f}, "
                f"Recall: {metrics["recall"]:.4f}, "
                f"F1: {metrics["f1"]:.4f}\n"
            )


# def partial():
#     file_list = ["karate", "jazz", "net_science", "cora_ml"]
#     type_ = "SIR"
#     result = {}

#     for name in file_list:
#         (
#             train_loader,
#             valid_loader,
#             test_loader,
#             edge_index,
#             num_nodes,
#             _,
#             _,
#         ) = load_data(name, type_)
#         with open(f"data/{type_}/{name}/dict_observe.pkl", "rb") as f:
#             dict_observe = pkl.load(f)
#         for i in range(1, 10):
#             observe = list(dict_observe[i])
#             roc_list = []
#             for _ in range(4):
#                 model = SLVAE(num_nodes, edge_index)
#                 trainer = SLVAETrainer(
#                     model=model,
#                     lr=0.001,
#                     device="cuda:0",
#                 )
#                 trainer.fit(
#                     train_loader,
#                     valid_loader,
#                     100,
#                     mask=observe,
#                 )
#                 auroc, _ = trainer.evaluate(
#                     test_loader,
#                     mask=observe,
#                 )
#                 roc_list.append(auroc)
#             roc_avg, _ = calculate_stats(roc_list)
#             result[(name, i)] = roc_avg

#     with open("./partial_slvae.txt", "w") as f:
#         for (name, i), metrics in result.items():
#             f.write(f"{name}-Mask{i}0%: " f"AUROC: {metrics:.4f}\n")


# def cascade_study():
#     file_list = ["douban", "twitter"]
#     result = {}

#     for name in file_list:
#         avg_time, roc_avg, roc_std, prc_avg, prc_std = train_cascade(
#             file_name=name,
#         )
#         result[name] = {
#             "time": avg_time,
#             "roc": (roc_avg, roc_std),
#             "prc": (prc_avg, prc_std),
#         }

#     with open("./cascade_slvae.txt", "w") as f:
#         f.write("Summary of all experiments using SLVAE:\n")
#         for name, metrics in result.items():
#             f.write(
#                 f"{name}: "
#                 f"Time: {metrics['time']:.4f}s, "
#                 f"AUROC: {metrics['roc'][0]:.4f} ± {metrics['roc'][1]:.4f}, "
#                 f"AUPRC: {metrics['prc'][0]:.4f} ± {metrics['prc'][1]:.4f}\n"
#             )


if __name__ == "__main__":
    main()
    # partial()
    # cascade_study()
