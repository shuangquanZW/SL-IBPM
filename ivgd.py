import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from utils import load_data, SEEDS, compute_stats, save_csv


class InvertibleGraphResidualNet(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, in_dim),
        )
        self.g1 = GCNConv(in_dim, hidden)
        self.g2 = GCNConv(hidden, in_dim)
        # apply spectral norm
        for m in self.f:
            if isinstance(m, nn.Linear):
                nn.utils.spectral_norm(m)
        nn.utils.spectral_norm(self.g1.lin)
        nn.utils.spectral_norm(self.g2.lin)

    def forward(self, y, edge_index, iters=3):
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
        return self.predict(z).squeeze(-1)


class IVGDTrainer:
    def __init__(self, model: IVGD, lr: float, reduction: str, device: str):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.estimator = nn.BCELoss(reduction=reduction)

    def train_step(self, train_loader, edge_index, mask=None):
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

    def valid_step(self, valid_loader, edge_index, mask=None):
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

    def test_step(self, test_loader, edge_index, mask=None):
        edge_index = edge_index.to(self.device)
        auroc_list, precision_list, recall_list, f1_list = [], [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                y_hat = self.model(x, edge_index).reshape(-1).cpu().numpy()
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

    def fit(self, train_loader, valid_loader, edge_index, epochs, mask=None):
        for epoch in range(epochs):
            self.model.train()
            train_loss = self.train_step(train_loader, edge_index, mask)
            self.model.eval()
            valid_loss = self.valid_step(valid_loader, edge_index, mask)
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
                )

    def evaluate(self, test_loader, edge_index, mask=None):
        self.model.eval()
        return self.test_step(test_loader, edge_index, mask)


def train_ivgd(
    file_name: str, type_: str, epochs=100, lr=0.001, reduction="mean", device="cuda:0"
):
    train_loader, valid_loader, test_loader, edge_index, _, _, states = load_data(
        file_name, type_
    )
    auc_list, pre_list, rec_list, f1_list = [], [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = IVGD(states)
        trainer = IVGDTrainer(model=model, lr=lr, reduction=reduction, device=device)
        trainer.fit(train_loader, valid_loader, edge_index, epochs)
        auc, pre, rec, f1 = trainer.evaluate(test_loader, edge_index)
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
            metrics = train_ivgd(file_name=name, type_=type_)
            results[(name, type_)] = metrics
    save_csv(results, "result/IVGD.csv")


def cascade_study(device="cuda:0", epochs=100):
    from utils import get_true_cascade_dataset

    file_list = ["android", "christianity", "douban", "twitter"]
    results = {}
    for name in file_list:
        train_ld, valid_ld, test_ld, ei, _, _, states = get_true_cascade_dataset(name)
        auc_list, pre_list, rec_list, f1_list = [], [], [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = IVGD(states)
            trainer = IVGDTrainer(model, lr=0.001, reduction="mean", device=device)
            trainer.fit(train_ld, valid_ld, ei, epochs)
            auc, pre, rec, f1 = trainer.evaluate(test_ld, ei)
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
    save_csv(results, "result/IVGD_cascade.csv")


if __name__ == "__main__":
    main()
    # cascade_study(device="cuda:1")
