import numpy as np
import torch

from utils import SEEDS, compute_stats, get_true_cascade_dataset, save_csv

CASCADE_DATASETS = ["android", "christianity", "douban", "twitter"]


def run_cascade_experiment(
    model_factory,
    trainer_factory,
    aggregate_histories,
    save_history_csv,
    result_key: str,
    output_csv: str,
    epochs: int,
    history_dir: str,
):
    results = {}
    for name in CASCADE_DATASETS:
        (
            train_loader,
            valid_loader,
            test_loader,
            edge_index,
            num_nodes,
            num_edges,
            num_states,
        ) = get_true_cascade_dataset(name)

        all_histories = []
        auc_list, precision_list, recall_list, f1_list = [], [], [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = model_factory(num_nodes, num_edges, num_states)
            trainer = trainer_factory(model)
            history = trainer.fit(
                train_loader,
                valid_loader,
                test_loader,
                edge_index,
                epochs=epochs,
            )
            all_histories.append(history)

            roc, precision, recall, f1 = trainer.evaluate(test_loader, edge_index)
            auc_list.append(roc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        agg = aggregate_histories(all_histories)
        save_history_csv(agg, f"{history_dir}/{result_key}_{name}_cascade.csv")

        results[(name, "cascade")] = {
            "auc_mean": compute_stats(auc_list)[0],
            "auc_std": compute_stats(auc_list)[1],
            "pre_mean": compute_stats(precision_list)[0],
            "pre_std": compute_stats(precision_list)[1],
            "rec_mean": compute_stats(recall_list)[0],
            "rec_std": compute_stats(recall_list)[1],
            "f1_mean": compute_stats(f1_list)[0],
            "f1_std": compute_stats(f1_list)[1],
        }
        print(f"Saved aggregated cascade history for {result_key}-{name}")

    save_csv(results, output_csv)
    return results


def run_epoch_sweep(run_one, epochs_list=(100,)):
    for epochs in epochs_list:
        run_one(epochs=epochs)
