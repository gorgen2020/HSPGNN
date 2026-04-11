"""Training entrypoint for HSPGCN / HSPGCN_L.

This script keeps the original training logic but improves:
- argument documentation
- path management
- device handling
- reproducibility helpers
- experiment directory layout
- overall readability
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from lib.data_preparation import read_and_generate_dataset
from lib.utils import compute_val_loss, evaluate, predict
from model import HSPGCN, HSPGCN_L


PROJECT_ROOT = Path(__file__).resolve().parent


DATASET_CONFIG = {
    "pems_bay": {
        "num_features": 1,
        "batch_size": 16,
        "merge": False,
        "adj_path": PROJECT_ROOT / "data" / "pems_bay" / "pems_bay_adj.npz",
    },
    "Electricity": {
        "num_features": 1,
        "batch_size": 16,
        "merge": False,
        "adj_path": PROJECT_ROOT / "data",
    },
    "AQI": {
        "num_features": 1,
        "batch_size": 12,
        "merge": False,
        "adj_path": PROJECT_ROOT / "data" / "AQI" / "AQI_adj.npz",
    },
    "AQI36": {
        "num_features": 1,
        "batch_size": 8,
        "merge": False,
        "adj_path": PROJECT_ROOT / "data" / "AQI" / "AQI_small_adj.npz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HSPGCN for time-series imputation.")
    parser.add_argument("--device", type=str, default="cuda:0", help="PyTorch device, e.g. cuda:0 or cpu")
    parser.add_argument("--max_epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam"], help="Optimizer type")
    parser.add_argument("--length", type=int, default=30, help="Reserved legacy argument; kept for compatibility")
    parser.add_argument("--force", action="store_true", default=True, help="Overwrite existing experiment directory")

    parser.add_argument("--data_name", type=str, default='AQI36',
                        help="AQI,AQI36,pems_bay,Electricity", required=False)
    parser.add_argument('--num_point', type=int, default=36,
                        help='road Point Number [437/36/325/370] ', required=False)
    parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')
    parser.add_argument('--model', type=str, default='HSPGCN', help='HSPGCN or HSPGCN_L ')

    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'test'],
                        help='train: train model; test: load checkpoint and reproduce results')

    parser.add_argument('--checkpoint', type=str, default='./save_params/HSPGCN_AQI36_epoch_38_1.18.params',
                        help='Path to a saved .params file')

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument(
        "--experiment_root",
        type=str,
        default=None,
        help="Optional custom directory for checkpoints and metrics",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def build_support_matrix(dataset_name: str, device: torch.device) -> tuple[np.ndarray, torch.Tensor, int, int, bool]:
    cfg = DATASET_CONFIG[dataset_name]
    batch_size = cfg["batch_size"]
    num_features = cfg["num_features"]
    merge = cfg["merge"]

    if dataset_name == "pems_bay":
        adj = np.load(cfg["adj_path"])["adj"]
        adj = np.array(adj > 0.0, dtype=float)
        adj = np.matmul(adj, adj)
        adj = np.array(adj > 0.0, dtype=float)
    elif dataset_name == "Electricity":
        adj = np.ones((370, 370), dtype=float)
    elif dataset_name == "AQI":
        adj1 = np.load(cfg["adj_path"])["adj"]
        adj = np.matmul(adj1, adj1)
        adj = np.where(adj > 0, 1.0, 0.0)
    elif dataset_name == "AQI36":
        adj1 = np.load(cfg["adj_path"])["adj"]
        adj = np.matmul(adj1, adj1)
        adj2 = np.where(adj1 > 0, adj1, adj)
        adj = np.matmul(adj2, adj1)
        adj = np.where(adj2 > 0, adj2, adj)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    supports = torch.tensor(adj, dtype=torch.float32, device=device)
    return adj, supports, num_features, batch_size, merge


def make_dataloader(split: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(split["week"], dtype=torch.float32),
            torch.tensor(split["week_mask"], dtype=torch.float32),
            torch.tensor(split["day"], dtype=torch.float32),
            torch.tensor(split["day_mask"], dtype=torch.float32),
            torch.tensor(split["recent"], dtype=torch.float32),
            torch.tensor(split["recent_mask"], dtype=torch.float32),
            torch.tensor(split["target"], dtype=torch.float32),
            torch.tensor(split["target_mask"], dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def save_stats(all_data: dict, output_dir: Path) -> None:
    stats_data = {}
    for feature_type in ["week", "day", "recent"]:
        stats = all_data["stats"][feature_type]
        stats_data[f"{feature_type}_mean"] = stats["mean"]
        stats_data[f"{feature_type}_std"] = stats["std"]
    np.savez_compressed(output_dir / "stats_data.npz", **stats_data)


def build_model(model_name: str, num_features: int, num_nodes: int) -> torch.nn.Module:
    model_cls = HSPGCN if model_name == "HSPGCN" else HSPGCN_L
    return model_cls(
        c_in=num_features,
        c_out=64,
        num_nodes=num_nodes,
        week=12,
        day=12,
        recent=36,
        K=3,
        Kt=3,
    )


def prepare_experiment_dir(args: argparse.Namespace) -> Path:
    model_tag = f"{args.model}_{args.data_name}"
    base_dir = Path(args.experiment_root) if args.experiment_root else PROJECT_ROOT / f"experiment_{args.model}"
    params_path = base_dir / model_tag

    if params_path.exists() and not args.force:
        raise SystemExit(f"Experiment directory already exists: {params_path}. Use --force to overwrite.")

    if params_path.exists():
        shutil.rmtree(params_path)
    params_path.mkdir(parents=True, exist_ok=True)
    return params_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    adj, supports, num_features, batch_size, merge = build_support_matrix(args.data_name, device)
    params_path = prepare_experiment_dir(args)
    model_name = f"{args.model}_{args.data_name}"
    prediction_path = PROJECT_ROOT / f"{args.model}_imputation_{args.data_name}.npz"

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Experiment directory: {params_path}")
    print("Reading data...")


    all_data = read_and_generate_dataset(
        args.data_name,
        num_of_weeks=1,
        num_of_days=1,
        num_of_hours=3,
        num_for_predict=12,
        points_per_hour=12,
        merge=merge,
    )

    true_value = all_data["test"]["target"]
    true_value_mask = all_data["test"]["target_mask"]

    train_loader = make_dataloader(all_data["train"], batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(all_data["val"], batch_size=batch_size, shuffle=False)
    test_loader = make_dataloader(all_data["test"], batch_size=batch_size, shuffle=False)
    save_stats(all_data, params_path)

    loss_function = torch.nn.SmoothL1Loss(reduction="mean", beta=0.5)

    net = build_model(args.model, num_features=num_features, num_nodes=args.num_point).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.decay)

    compute_val_loss(net, val_loader, loss_function, supports, device, epoch=0)
    evaluate(net, test_loader, true_value, true_value_mask, supports, device, epoch=0)



    history = []
    train_times = []

    if args.mode == "test":
        if args.checkpoint is None:
            raise ValueError("In test mode, --checkpoint must be provided.")
        best_path = args.checkpoint
        print('Restore the saved best model.')

    else:
        for epoch in range(1, args.max_epoch + 1):
            epoch_losses = []
            start_time = time()
            net.train()

            for train_w, train_w_mask, train_d, train_d_mask, train_r, train_r_mask, train_t, train_t_mask in train_loader:
                train_w = train_w.to(device)
                train_w_mask = train_w_mask.to(device)
                train_d = train_d.to(device)
                train_d_mask = train_d_mask.to(device)
                train_r = train_r.to(device)
                train_r_mask = train_r_mask.to(device)
                train_t = train_t.to(device)
                train_t_mask = 1 - train_t_mask.to(device)

                optimizer.zero_grad()
                output, _, _, _ = net(
                    train_w,
                    train_w_mask,
                    train_d,
                    train_d_mask,
                    train_r,
                    train_r_mask,
                    train_t_mask,
                    supports,
                )
                loss = loss_function(output * train_t_mask, train_t * train_t_mask)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            elapsed = time() - start_time
            train_times.append(elapsed)
            train_loss = float(np.mean(epoch_losses))
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | time={elapsed:.2f}s")

            valid_loss = compute_val_loss(net, val_loader, loss_function, supports, device, epoch)
            history.append(valid_loss)
            evaluate(net, test_loader, true_value, true_value_mask, supports, device, epoch)

            ckpt_name = f"{model_name}_epoch_{epoch}_{round(valid_loss, 2)}.params"
            torch.save(net.state_dict(), params_path / ckpt_name)
            print(f"Saved checkpoint: {params_path / ckpt_name}")

        print("Training finished")
        print(f"Average training time per epoch: {np.mean(train_times):.2f}s")

        best_idx = int(np.argmin(history))
        best_epoch = best_idx + 1
        best_loss = history[best_idx]
        best_path = params_path / f"{model_name}_epoch_{best_epoch}_{round(best_loss, 2)}.params"
        print(f"Best validation checkpoint: epoch {best_epoch}, val_loss={best_loss:.4f}")

    net.load_state_dict(torch.load(best_path, map_location=device))
    start_test = time()
    prediction, spatial_at, parameter_adj = predict(net, test_loader, supports, device)
    test_time = time() - start_test
    evaluate(net, test_loader, true_value, true_value_mask, supports, device, epoch=args.max_epoch)
    print(f"Test time: {test_time:.2f}s")

    np.savez_compressed(
        prediction_path,
        adj=adj,
        prediction=prediction,
        spatial_at=spatial_at,
        parameter_adj=parameter_adj,
        ground_truth=all_data["test"]["target"],
        ground_truth_mask=all_data["test"]["target_mask"],
        created_at=datetime.now().isoformat(timespec="seconds"),
        model=model_name,
    )
    print(f"Saved predictions to: {prediction_path}")


if __name__ == "__main__":
    main()
