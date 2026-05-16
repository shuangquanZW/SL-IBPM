import torch
from torch import nn
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from utils import load_data, SEEDS, compute_stats, save_csv


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gcn = GCNConv(in_dim, out_dim, add_self_loops=False)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, edge_index):
        origin = x
        batch_size, num_nodes, in_dim = x.shape
        x = x.view(-1, in_dim)
        x = self.gcn(x, edge_index)
        x = x.view(batch_size, num_nodes, -1)
        x = self.activation(x)
        return x + origin


class ResGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=1, num_layers=6):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.output_proj(x).squeeze(-1)


class ResGCNTrainer:
    def __init__(self, model: ResGCN, lr: float, device: str):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_step(self, train_loader, edge_index, mask=None):
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

    def valid_step(self, valid_loader, edge_index, mask=None):
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

    def test_step(self, test_loader, edge_index, mask=None):
        edge_index = edge_index.to(self.device)
        auroc_list, precision_list, recall_list, f1_list = [], [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                if mask:
                    x[:, mask, :] = 0
                y_hat = self.model(x, edge_index).cpu().numpy()
                y = y.cpu().numpy()
                for i in range(y.shape[0]):
                    y_true = y[i].ravel()
                    y_pred = y_hat[i].ravel()
                    if np.unique(y_true).size < 2:
                        continue
                    auroc_list.append(roc_auc_score(y_true, y_pred))
                    y_pred = (y_pred > 0.5).astype(int)
                    precision_list.append(
                        precision_score(y_true, y_pred, zero_division=0)
                    )
                    recall_list.append(recall_score(y_true, y_pred, zero_division=0))
                    f1_list.append(f1_score(y_true, y_pred, zero_division=0))
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


def train_resgcn(file_name: str, type_: str, epochs=100, lr=0.001, device="cuda:0"):
    train_loader, valid_loader, test_loader, edge_index, _, _, num_states = load_data(
        file_name, type_
    )
    auc_list, pre_list, rec_list, f1_list = [], [], [], []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ResGCN(in_dim=num_states)
        trainer = ResGCNTrainer(model=model, lr=lr, device=device)
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
            metrics = train_resgcn(file_name=name, type_=type_)
            results[(name, type_)] = metrics
    save_csv(results, "result/MPNN.csv")


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
            model = ResGCN(in_dim=num_states)
            trainer = ResGCNTrainer(model, lr=0.001, device=device)
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
    save_csv(results, "result/MPNN_cascade.csv")


if __name__ == "__main__":
    main()
    # cascade_study(device="cuda:1")
