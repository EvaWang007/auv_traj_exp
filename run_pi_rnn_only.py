#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "auv_trajectory_smoke_experiment.py"


def load_src():
    spec = importlib.util.spec_from_file_location("auv_smoke", SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def predict_model(module, model, loader, y_scaler, device: str):
    model.eval()
    preds, trues, raws = [], [], []
    for xb, yb, xraw, yraw in loader:
        xb = xb.to(device)
        pred_norm = model(xb).cpu().numpy()
        pred = y_scaler.inverse_transform(pred_norm)
        preds.append(pred)
        trues.append(yraw.numpy())
        raws.append(xraw.numpy())
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(raws)


def run_pi_rnn(module, npz_path: Path, out_dir: Path, label: str, args):
    print(f"\n=== Running PI-RNN only dataset: {label} ===")
    loaded = np.load(npz_path)
    data = loaded["data"]

    train_traj, val_traj, test_traj = module.split_by_trajectory(data, 0.7, 0.1)
    X_train, Y_train = module.make_windows(train_traj, args.input_len, args.pred_len)
    X_val, Y_val = module.make_windows(val_traj, args.input_len, args.pred_len)
    X_test, Y_test = module.make_windows(test_traj, args.input_len, args.pred_len)

    x_scaler = module.Standardizer().fit(X_train)
    y_scaler = module.Standardizer().fit(Y_train)

    train_ds = module.AUVDataset(X_train, Y_train, x_scaler, y_scaler)
    val_ds = module.AUVDataset(X_val, Y_val, x_scaler, y_scaler)
    test_ds = module.AUVDataset(X_test, Y_test, x_scaler, y_scaler)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"device={device}, train_windows={len(train_ds)}, val_windows={len(val_ds)}, test_windows={len(test_ds)}")

    pi_rnn = module.RNNPredictor(
        input_dim=5,
        hidden_dim=args.hidden_dim,
        pred_len=args.pred_len,
        cell="rnn",
    )
    pi_rnn = module.train_model(
        "PI_RNN",
        pi_rnn,
        train_loader,
        val_loader,
        y_scaler,
        args.dt,
        args.lambda_phy,
        args.epochs,
        args.lr,
        device,
    )

    pred_pi_rnn, true_pi_rnn, raw_pi_rnn = predict_model(module, pi_rnn, test_loader, y_scaler, device)
    results = [{
        "model": f"PI_RNN_lambda_{args.lambda_phy}",
        **module.metrics_np(pred_pi_rnn, true_pi_rnn),
    }]
    df = pd.DataFrame(results)
    out_csv = out_dir / f"results_pi_rnn_only_{label}.csv"
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"[OK] saved {out_csv}")

    ckpt_path = out_dir / f"pi_rnn_only_{label}.pt"
    torch.save(pi_rnn.state_dict(), ckpt_path)
    print(f"[OK] saved {ckpt_path}")

    sample_idx = min(50, len(Y_test) - 1)
    plt.figure(figsize=(7, 6))
    hist = X_test[sample_idx, :, :2]
    gt = Y_test[sample_idx]
    plt.plot(hist[:, 0], hist[:, 1], "ko-", label="input history(nav)", linewidth=1)
    plt.plot(gt[:, 0], gt[:, 1], "g-", label="ground truth", linewidth=2)
    plt.plot(pred_pi_rnn[sample_idx, :, 0], pred_pi_rnn[sample_idx, :, 1], "--", label="PI-RNN")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(f"AUV PI-RNN prediction sample: {label}")
    fig_path = out_dir / f"fig_prediction_pi_rnn_only_{label}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] saved {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "all"], default="train")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_traj", type=int, default=800)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--input_len", type=int, default=50)
    parser.add_argument("--pred_len", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_phy", type=float, default=0.05)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    module = load_src()
    module.set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    no_current_path = out_dir / "dataset_no_current.npz"
    current_path = out_dir / "dataset_current.npz"

    if args.mode == "all":
        cfg0 = module.SimConfig(
            n_traj=args.n_traj,
            steps=args.steps,
            dt=args.dt,
            current_kind="none",
            current_strength=0.0,
        )
        module.generate_dataset(cfg0, no_current_path)

        cfg1 = module.SimConfig(
            n_traj=args.n_traj,
            steps=args.steps,
            dt=args.dt,
            current_kind="mixed",
            current_strength=0.35,
        )
        module.generate_dataset(cfg1, current_path)

    if not no_current_path.exists() or not current_path.exists():
        raise FileNotFoundError("Dataset files not found. Generate them first or use --mode all.")

    run_pi_rnn(module, no_current_path, out_dir, "no_current", args)
    run_pi_rnn(module, current_path, out_dir, "current", args)


if __name__ == "__main__":
    main()
