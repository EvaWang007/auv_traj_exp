#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "auv_trajectory_smoke_experiment.py"


def load_src():
    spec = importlib.util.spec_from_file_location("auv_smoke_encdec", SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EncoderDecoderLSTMPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, pred_len=25, decoder_input_dim=2):
        super().__init__()
        self.pred_len = pred_len
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(decoder_input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x, decoder_init, target_seq=None, teacher_forcing_ratio=0.0):
        _, (h, c) = self.encoder(x)

        decoder_input = decoder_init.unsqueeze(1)
        outputs = []
        for step_idx in range(self.pred_len):
            dec_out, (h, c) = self.decoder(decoder_input, (h, c))
            pred_step = self.head(dec_out.squeeze(1))
            outputs.append(pred_step.unsqueeze(1))

            if self.training and target_seq is not None and teacher_forcing_ratio > 0.0:
                teacher_mask = (torch.rand(x.size(0), device=x.device) < teacher_forcing_ratio).float().unsqueeze(1)
                next_input = teacher_mask * target_seq[:, step_idx, :] + (1.0 - teacher_mask) * pred_step
            else:
                next_input = pred_step
            decoder_input = next_input.unsqueeze(1)

        return torch.cat(outputs, dim=1)


def get_decoder_init_norm(xraw: torch.Tensor, y_scaler, device: str) -> torch.Tensor:
    y_mean = torch.tensor(y_scaler.mean[:2], dtype=torch.float32, device=device)
    y_std = torch.tensor(y_scaler.std[:2], dtype=torch.float32, device=device)
    last_nav = xraw[:, -1, :2].to(device)
    return (last_nav - y_mean) / y_std


def train_encdec_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_scaler,
    epochs: int,
    lr: float,
    teacher_forcing_ratio: float,
    device: str,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb, xraw, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xraw = xraw.to(device)
            dec0 = get_decoder_init_norm(xraw, y_scaler, device)

            pred = model(xb, dec0, target_seq=yb, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = torch.mean((pred - yb) ** 2)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += loss.item() * len(xb)

        val_loss = evaluate_encdec_loss(model, val_loader, y_scaler, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(
                f"[EncDec_LSTM] epoch={ep:03d} "
                f"train_loss={total/len(train_loader.dataset):.6f} "
                f"val_mse={val_loss:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_encdec_loss(model: nn.Module, loader: DataLoader, y_scaler, device: str) -> float:
    model.eval()
    total = 0.0
    for xb, yb, xraw, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        xraw = xraw.to(device)
        dec0 = get_decoder_init_norm(xraw, y_scaler, device)
        pred = model(xb, dec0)
        loss = torch.mean((pred - yb) ** 2)
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def predict_encdec_model(model: nn.Module, loader: DataLoader, y_scaler, device: str):
    model.eval()
    preds, trues, raws = [], [], []
    for xb, _, xraw, yraw in loader:
        xb = xb.to(device)
        xraw = xraw.to(device)
        dec0 = get_decoder_init_norm(xraw, y_scaler, device)
        pred_norm = model(xb, dec0).cpu().numpy()
        pred = y_scaler.inverse_transform(pred_norm)
        preds.append(pred)
        trues.append(yraw.numpy())
        raws.append(xraw.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(raws)


def run_experiment(module, npz_path: Path, out_dir: Path, label: str, args):
    print(f"\n=== Running EncDec-LSTM comparison dataset: {label} ===")
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

    results = []

    ekf_pred = module.ekf_like_predict(X_test, args.pred_len, args.dt)
    results.append({"model": "EKF_like_kinematic", **module.metrics_np(ekf_pred, Y_test)})

    lstm = module.RNNPredictor(input_dim=5, hidden_dim=args.hidden_dim, pred_len=args.pred_len, cell="lstm")
    lstm = module.train_model(
        "LSTM",
        lstm,
        train_loader,
        val_loader,
        y_scaler,
        args.dt,
        0.0,
        args.epochs,
        args.lr,
        device,
    )
    pred_lstm, true_lstm, _ = module.predict_model(lstm, test_loader, y_scaler, device)
    results.append({"model": "LSTM", **module.metrics_np(pred_lstm, true_lstm)})

    encdec = EncoderDecoderLSTMPredictor(
        input_dim=5,
        hidden_dim=args.hidden_dim,
        pred_len=args.pred_len,
        decoder_input_dim=2,
    )
    encdec = train_encdec_model(
        encdec,
        train_loader,
        val_loader,
        y_scaler,
        epochs=args.epochs,
        lr=args.lr,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        device=device,
    )
    pred_encdec, true_encdec, _ = predict_encdec_model(encdec, test_loader, y_scaler, device)
    results.append({"model": "EncDec_LSTM", **module.metrics_np(pred_encdec, true_encdec)})

    df = pd.DataFrame(results)
    out_csv = out_dir / f"results_encdec_{label}.csv"
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"[OK] saved {out_csv}")

    torch.save(lstm.state_dict(), out_dir / f"lstm_oneshot_{label}.pt")
    torch.save(encdec.state_dict(), out_dir / f"encdec_lstm_{label}.pt")

    sample_idx = min(50, len(Y_test) - 1)
    plt.figure(figsize=(7, 6))
    hist = X_test[sample_idx, :, :2]
    gt = Y_test[sample_idx]
    ekf = ekf_pred[sample_idx]
    plt.plot(hist[:, 0], hist[:, 1], "ko-", label="input history(nav)", linewidth=1)
    plt.plot(gt[:, 0], gt[:, 1], "g-", label="ground truth", linewidth=2)
    plt.plot(ekf[:, 0], ekf[:, 1], "--", label="EKF-like")
    plt.plot(pred_lstm[sample_idx, :, 0], pred_lstm[sample_idx, :, 1], "--", label="LSTM")
    plt.plot(pred_encdec[sample_idx, :, 0], pred_encdec[sample_idx, :, 1], "--", label="EncDec-LSTM")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(f"AUV EncDec-LSTM comparison: {label}")
    fig_path = out_dir / f"fig_prediction_encdec_{label}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] saved {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "train", "all"], default="train")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_traj", type=int, default=800)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--input_len", type=int, default=60)
    parser.add_argument("--pred_len", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--current_kind", type=str, default="mixed_hard")
    parser.add_argument("--current_strength", type=float, default=0.45)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    module = load_src()
    module.set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    no_current_path = out_dir / "dataset_no_current.npz"
    current_path = out_dir / "dataset_current.npz"

    if args.mode in ("generate", "all"):
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
            current_kind=args.current_kind,
            current_strength=args.current_strength,
        )
        module.generate_dataset(cfg1, current_path)

    if args.mode in ("train", "all"):
        if not no_current_path.exists() or not current_path.exists():
            raise FileNotFoundError("Dataset files not found. Generate them first or use --mode all.")
        run_experiment(module, no_current_path, out_dir, "no_current", args)
        run_experiment(module, current_path, out_dir, "current", args)


if __name__ == "__main__":
    main()
