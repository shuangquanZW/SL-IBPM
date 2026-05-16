import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch_scatter import scatter
from utils import load_data, SEEDS, compute_stats, save_csv


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
    def __init__(self, n_nodes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_nodes, 128), nn.ReLU(), nn.Linear(128, n_nodes), nn.Sigmoid()
        )

    def forward(self, x_hat):
        return self.mlp(x_hat)


class SLVAE(nn.Module):
    def __init__(self, n_nodes: int, edge_index: torch.Tensor):
        super().__init__()
        self.vae = VAE(n_nodes)
        self.gnn = GNN(n_nodes)
        self.diffusion = DiffusionPropagate(edge_index)

    def forward(self, x):
        x = 1 - x[..., 0]
        x_hat, mu, logvar = self.vae(x)
        p0 = self.gnn(x_hat)
        y_hat = self.diffusion(p0)
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

    def train_step(self, train_loader, mask=None):
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

    def valid_step(self, valid_loader, mask=None):
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

    def test_step(self, test_loader, mask=None):
        auroc_list, precision_list, recall_list, f1_list = [], [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                _, _, _, y_hat = self.model(x)
                y_hat = y_hat.reshape(-1).cpu().numpy()
                y = y.reshape(-1).cpu().numpy()
                auroc_list.append(roc_auc_score(y, y_hat))
                y_hat = (y_hat > 0.5).astype(int)
                precision_list.append(precision_score(y, y_hat, zero_division=0))
                recall_list.append(recall_score(y, y_hat, zero_division=0))
                f1_list.append(f1_score(y, y_hat, zero_division=0))
        return (
            float(np.mean(auroc_list)),
            float(np.mean(precision_list)),
            float(np.mean(recall_list)),
            float(np.mean(f1_list)),
        )

    def fit(self, train_loader, valid_loader, epochs, mask=None):
        for epoch in range(epochs):
            self.model.train()
            train_loss = self.train_step(train_loader, mask)
            self.model.eval()
            valid_loss = self.valid_step(valid_loader, mask)
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
                )

    def evaluate(self, test_loader, mask=None):
        self.model.eval()
        return self.test_step(test_loader, mask)


def train_slvae(file_name: str, type_: str, epochs=100, lr=0.001, device="cuda:0"):
    train_loader, valid_loader, test_loader, edge_index, num_nodes, _, _ = load_data(
        file_name, type_
    )
    auc_list, pre_list, rec_list, f1_list = [], [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SLVAE(num_nodes, edge_index)
        trainer = SLVAETrainer(model=model, lr=lr, device=device)
        trainer.fit(train_loader, valid_loader, epochs)
        auc, pre, rec, f1 = trainer.evaluate(test_loader)
        auc_list.append(auc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)
    return {
        "auc_mean": compute_stats(auc_list)[0],
        "auc_std": compute_stats(auc_list)[1],
        "pre_mean": compute_stats(pre_list)[0],
        "pre_std": compute_stats(pre_list)[1],
        "rec_mean": compute_stats(rec_list)[0],
        "rec_std": compute_stats(rec_list)[1],
        "f1_mean": compute_stats(f1_list)[0],
        "f1_std": compute_stats(f1_list)[1],
    }


def main():
    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR", "SI", "LT", "IC"]
    results = {}
    for name in file_list:
        for type_ in type_list:
            metrics = train_slvae(file_name=name, type_=type_)
            results[(name, type_)] = metrics
    save_csv(results, "result/SLVAE.csv")


def cascade_study(device="cuda:0", epochs=100):
    from utils import get_true_cascade_dataset

    file_list = ["android", "christianity", "douban", "twitter"]
    results = {}
    for name in file_list:
        train_ld, valid_ld, test_ld, ei, N, _, _ = get_true_cascade_dataset(name)
        auc_list, pre_list, rec_list, f1_list = [], [], [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = SLVAE(N, ei)
            trainer = SLVAETrainer(model, lr=0.001, device=device)
            trainer.fit(train_ld, valid_ld, epochs)
            auc, pre, rec, f1 = trainer.evaluate(test_ld)
            auc_list.append(auc)
            pre_list.append(pre)
            rec_list.append(rec)
            f1_list.append(f1)
        results[(name, "cascade")] = {
            "auc_mean": compute_stats(auc_list)[0],
            "auc_std": compute_stats(auc_list)[1],
            "pre_mean": compute_stats(pre_list)[0],
            "pre_std": compute_stats(pre_list)[1],
            "rec_mean": compute_stats(rec_list)[0],
            "rec_std": compute_stats(rec_list)[1],
            "f1_mean": compute_stats(f1_list)[0],
            "f1_std": compute_stats(f1_list)[1],
        }
    save_csv(results, "result/SLVAE_cascade.csv")


if __name__ == "__main__":
    main()
    cascade_study()
