import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from utils import load_data, SEEDS, compute_stats, save_csv


class RSDGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.preprocess = nn.Linear(input_dim + 4, hidden_dim)
        self.gin_convs = nn.ModuleList(
            [
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
                for _ in range(2)
            ]
        )
        self.source_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _state_encoding(self, S):
        T1 = torch.where(S == 1, 1.0, -1.0)
        T2 = torch.where(S == 1, 1.0, 0.0)
        T3 = torch.where(S == 0, 0.0, -1.0)
        return torch.stack([S, T1, T2, T3], dim=1)

    def forward(self, x, edge_index):
        batch_size, num_nodes, _ = x.size()
        x_flat = x.view(-1, x.size(-1))
        S = x_flat[:, 1]
        T = self._state_encoding(S)
        x_flat = torch.cat([x_flat, T], dim=1)
        x_flat = F.relu(self.preprocess(x_flat))
        for conv in self.gin_convs:
            x_flat = F.relu(conv(x_flat, edge_index))
        probs = self.source_mlp(x_flat).squeeze()
        return probs.view(batch_size, num_nodes)


class RSDGINTrainer:
    def __init__(self, model: RSDGIN, lr: float, reduction: str, device: str):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.estimator = nn.BCELoss(reduction=reduction)

    def train_step(self, train_loader, edge_index, mask=None):
        edge_index = edge_index.to(self.device)
        total_loss = 0.0
        self.model.train()
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
        self.model.eval()
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
        edge_index = edge_index.to(self.device)
        auroc_list, precision_list, recall_list, f1_list = [], [], [], []
        self.model.eval()
        with torch.no_grad():
            for x, true in test_loader:
                x, true = x.to(self.device), true.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                pred = self.model(x, edge_index).reshape(-1).cpu().numpy()
                y = true.reshape(-1).cpu().numpy()
                auroc_list.append(roc_auc_score(y, pred))
                pred = (pred > 0.5).astype(int)
                precision_list.append(precision_score(y, pred, zero_division=0))
                recall_list.append(recall_score(y, pred, zero_division=0))
                f1_list.append(f1_score(y, pred, zero_division=0))
        return (
            float(np.mean(auroc_list)),
            float(np.mean(precision_list)),
            float(np.mean(recall_list)),
            float(np.mean(f1_list)),
        )

    def fit(self, train_loader, valid_loader, edge_index, epochs, mask=None):
        for epoch in range(epochs):
            train_loss = self.train_step(train_loader, edge_index, mask)
            valid_loss = self.valid_step(valid_loader, edge_index, mask)
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
                )

    def evaluate(self, test_loader, edge_index, mask=None):
        return self.test_step(test_loader, edge_index, mask)


def train_rdgin(
    file_name: str, type_: str, epochs=100, lr=0.001, reduction="mean", device="cuda:0"
):
    train_loader, valid_loader, test_loader, edge_index, _, _, num_states = load_data(
        file_name, type_
    )
    auc_list, pre_list, rec_list, f1_list = [], [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = RSDGIN(num_states, 256)
        trainer = RSDGINTrainer(model, lr, reduction, device)
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
            metrics = train_rdgin(file_name=name, type_=type_)
            results[(name, type_)] = metrics
    save_csv(results, "result/RDGIN.csv")


def cascade_study(device="cuda:0", epochs=100):
    from utils import get_true_cascade_dataset

    file_list = ["android", "christianity", "douban", "twitter"]
    results = {}
    for name in file_list:
        train_ld, valid_ld, test_ld, ei, _, _, num_states = get_true_cascade_dataset(
            name
        )
        auc_list, pre_list, rec_list, f1_list = [], [], [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = RSDGIN(num_states, 256)
            trainer = RSDGINTrainer(model, lr=0.001, reduction="mean", device=device)
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
    save_csv(results, "result/RDGIN_cascade.csv")


if __name__ == "__main__":
    main()
    # cascade_study(device="cuda:1")
