import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / 'auv_trajectory_smoke_experiment.py'


def load_src():
    spec = importlib.util.spec_from_file_location('auv_smoke_base', SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CurrentWindowDataset(Dataset):
    def __init__(self, X_raw, Y_pos_raw, Y_cur_raw, x_scaler, cur_scaler):
        self.X_raw = X_raw.astype(np.float32)
        self.Y_pos_raw = Y_pos_raw.astype(np.float32)
        self.Y_cur_raw = Y_cur_raw.astype(np.float32)
        self.X = x_scaler.transform(self.X_raw).astype(np.float32)
        self.Y_cur = cur_scaler.transform(self.Y_cur_raw).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.Y_cur[idx]),
            torch.from_numpy(self.X_raw[idx]),
            torch.from_numpy(self.Y_pos_raw[idx]),
            torch.from_numpy(self.Y_cur_raw[idx]),
        )


def make_current_windows(data: np.ndarray, input_len: int, pred_len: int):
    """
    Input: [nav_x, nav_y, v, a, theta]
    Position target: future true position [true_x, true_y]
    Current target: future true current/disturbance [cx, cy]
    """
    Xs, Y_pos, Y_cur = [], [], []
    for traj in data:
        features = traj[:, [3, 4, 5, 6, 7]]
        target_pos = traj[:, [1, 2]]
        target_cur = traj[:, [8, 9]]
        max_i = len(traj) - input_len - pred_len
        for i in range(max_i):
            Xs.append(features[i:i + input_len])
            Y_pos.append(target_pos[i + input_len:i + input_len + pred_len])
            Y_cur.append(target_cur[i + input_len:i + input_len + pred_len])
    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(Y_pos, dtype=np.float32),
        np.asarray(Y_cur, dtype=np.float32),
    )


class LSTMCurrentPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, pred_len=30):
        super().__init__()
        self.pred_len = pred_len
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * 2),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).view(x.size(0), self.pred_len, 2)


def integrate_with_current(x_raw, cur_raw, dt: float):
    """
    Differentiable open-loop integration.
    x_raw last feature order: [nav_x, nav_y, v, a, theta].
    cur_raw: predicted future [cx, cy] in m/s, shape [B,K,2].
    """
    last = x_raw[:, -1, :]
    x0 = last[:, 0]
    y0 = last[:, 1]
    v0 = last[:, 2]
    a0 = last[:, 3]
    theta = last[:, 4]

    K = cur_raw.shape[1]
    steps = torch.arange(K, dtype=torch.float32, device=cur_raw.device).unsqueeze(0)
    v_i = torch.clamp(v0.unsqueeze(1) + a0.unsqueeze(1) * steps * dt, min=0.0)
    body_vel = torch.stack([
        v_i * torch.cos(theta).unsqueeze(1),
        v_i * torch.sin(theta).unsqueeze(1),
    ], dim=-1)

    delta = (body_vel + cur_raw) * dt
    start = torch.stack([x0, y0], dim=-1).unsqueeze(1)
    return start + torch.cumsum(delta, dim=1)


def current_inverse(cur_norm, cur_scaler, device):
    mean = torch.tensor(cur_scaler.mean, dtype=torch.float32, device=device)
    std = torch.tensor(cur_scaler.std, dtype=torch.float32, device=device)
    return cur_norm * std + mean


def smoothness_loss(cur_raw, current_scale: float):
    if cur_raw.shape[1] < 2:
        return torch.tensor(0.0, dtype=cur_raw.dtype, device=cur_raw.device)
    diff = cur_raw[:, 1:, :] - cur_raw[:, :-1, :]
    return torch.mean(diff ** 2) / (current_scale ** 2)


def run_epoch(model, loader, opt, cur_scaler, dt, pos_scale, current_scale, alpha_cur, beta_smooth, device):
    model.train()
    total = total_pos = total_cur = total_smooth = 0.0
    for xb, ycur, xraw, ypos, _ in loader:
        xb = xb.to(device)
        ycur = ycur.to(device)
        xraw = xraw.to(device)
        ypos = ypos.to(device)

        pred_cur_norm = model(xb)
        pred_cur_raw = current_inverse(pred_cur_norm, cur_scaler, device)
        pred_pos = integrate_with_current(xraw, pred_cur_raw, dt)

        loss_pos = torch.mean((pred_pos - ypos) ** 2) / (pos_scale ** 2)
        loss_cur = torch.mean((pred_cur_norm - ycur) ** 2)
        loss_smooth = smoothness_loss(pred_cur_raw, current_scale)
        loss = loss_pos + alpha_cur * loss_cur + beta_smooth * loss_smooth

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        n = len(xb)
        total += loss.item() * n
        total_pos += loss_pos.item() * n
        total_cur += loss_cur.item() * n
        total_smooth += loss_smooth.item() * n

    denom = len(loader.dataset)
    return total / denom, total_pos / denom, total_cur / denom, total_smooth / denom


@torch.no_grad()
def evaluate_losses(model, loader, cur_scaler, dt, pos_scale, current_scale, alpha_cur, beta_smooth, device):
    model.eval()
    total = total_pos = total_cur = total_smooth = 0.0
    for xb, ycur, xraw, ypos, _ in loader:
        xb = xb.to(device)
        ycur = ycur.to(device)
        xraw = xraw.to(device)
        ypos = ypos.to(device)

        pred_cur_norm = model(xb)
        pred_cur_raw = current_inverse(pred_cur_norm, cur_scaler, device)
        pred_pos = integrate_with_current(xraw, pred_cur_raw, dt)

        loss_pos = torch.mean((pred_pos - ypos) ** 2) / (pos_scale ** 2)
        loss_cur = torch.mean((pred_cur_norm - ycur) ** 2)
        loss_smooth = smoothness_loss(pred_cur_raw, current_scale)
        loss = loss_pos + alpha_cur * loss_cur + beta_smooth * loss_smooth

        n = len(xb)
        total += loss.item() * n
        total_pos += loss_pos.item() * n
        total_cur += loss_cur.item() * n
        total_smooth += loss_smooth.item() * n

    denom = len(loader.dataset)
    return total / denom, total_pos / denom, total_cur / denom, total_smooth / denom


@torch.no_grad()
def predict_all(model, loader, cur_scaler, dt, device):
    model.eval()
    pred_pos_all, true_pos_all = [], []
    pred_cur_all, true_cur_all = [], []
    xraw_all = []
    for xb, _, xraw, ypos, ycur_raw in loader:
        xb = xb.to(device)
        xraw_dev = xraw.to(device)
        pred_cur_norm = model(xb)
        pred_cur_raw = current_inverse(pred_cur_norm, cur_scaler, device)
        pred_pos = integrate_with_current(xraw_dev, pred_cur_raw, dt)

        pred_pos_all.append(pred_pos.cpu().numpy())
        true_pos_all.append(ypos.numpy())
        pred_cur_all.append(pred_cur_raw.cpu().numpy())
        true_cur_all.append(ycur_raw.numpy())
        xraw_all.append(xraw.numpy())

    return (
        np.concatenate(pred_pos_all),
        np.concatenate(true_pos_all),
        np.concatenate(pred_cur_all),
        np.concatenate(true_cur_all),
        np.concatenate(xraw_all),
    )


def current_metrics(pred_cur, true_cur):
    err = pred_cur - true_cur
    return {
        'Current_RMSE': float(np.sqrt(np.mean(err ** 2))),
        'Current_MAE': float(np.mean(np.abs(err))),
        'Current_MSE': float(np.mean(err ** 2)),
        'Current_Corr_cx': float(np.corrcoef(pred_cur[..., 0].ravel(), true_cur[..., 0].ravel())[0, 1]),
        'Current_Corr_cy': float(np.corrcoef(pred_cur[..., 1].ravel(), true_cur[..., 1].ravel())[0, 1]),
    }


def save_plots(out_dir, label, sample_idx, X_raw, true_pos, pred_pos, true_cur, pred_cur):
    sample_idx = int(np.clip(sample_idx, 0, len(true_pos) - 1))

    fig_path = out_dir / f'fig_prediction_lstm_true_current_integral_{label}.png'
    plt.figure(figsize=(7, 6))
    hist = X_raw[sample_idx, :, :2]
    plt.plot(hist[:, 0], hist[:, 1], 'ko-', linewidth=1, label='input history(nav)')
    plt.plot(true_pos[sample_idx, :, 0], true_pos[sample_idx, :, 1], 'g-', linewidth=2, label='ground truth')
    plt.plot(pred_pos[sample_idx, :, 0], pred_pos[sample_idx, :, 1], '--', linewidth=2, label='LSTM current-integral')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f'LSTM true-current integral trajectory: {label}')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f'[OK] saved {fig_path}')

    cur_fig_path = out_dir / f'fig_current_lstm_true_current_integral_{label}.png'
    k = np.arange(true_cur.shape[1])
    plt.figure(figsize=(8, 5))
    plt.plot(k, true_cur[sample_idx, :, 0], 'b-', label='true cx')
    plt.plot(k, pred_cur[sample_idx, :, 0], 'b--', label='pred cx')
    plt.plot(k, true_cur[sample_idx, :, 1], 'r-', label='true cy')
    plt.plot(k, pred_cur[sample_idx, :, 1], 'r--', label='pred cy')
    plt.grid(True)
    plt.xlabel('future step')
    plt.ylabel('current / disturbance (m/s)')
    plt.legend()
    plt.title(f'LSTM true-current prediction: {label}')
    plt.tight_layout()
    plt.savefig(cur_fig_path, dpi=200)
    plt.close()
    print(f'[OK] saved {cur_fig_path}')

    sample_csv = out_dir / f'current_prediction_sample_lstm_true_current_integral_{label}.csv'
    pd.DataFrame({
        'step': k,
        'true_cx': true_cur[sample_idx, :, 0],
        'pred_cx': pred_cur[sample_idx, :, 0],
        'true_cy': true_cur[sample_idx, :, 1],
        'pred_cy': pred_cur[sample_idx, :, 1],
        'true_x': true_pos[sample_idx, :, 0],
        'true_y': true_pos[sample_idx, :, 1],
        'pred_x': pred_pos[sample_idx, :, 0],
        'pred_y': pred_pos[sample_idx, :, 1],
    }).to_csv(sample_csv, index=False)
    print(f'[OK] saved {sample_csv}')


def run(args):
    module = load_src()
    module.set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / f'dataset_{args.dataset_label}.npz'
    if not dataset_path.exists():
        raise FileNotFoundError(f'{dataset_path} not found. Generate or copy the dataset first.')

    data = np.load(dataset_path)['data']
    train_traj, val_traj, test_traj = module.split_by_trajectory(data, 0.7, 0.1)
    X_train, Yp_train, Yc_train = make_current_windows(train_traj, args.input_len, args.pred_len)
    X_val, Yp_val, Yc_val = make_current_windows(val_traj, args.input_len, args.pred_len)
    X_test, Yp_test, Yc_test = make_current_windows(test_traj, args.input_len, args.pred_len)

    x_scaler = module.Standardizer().fit(X_train)
    cur_scaler = module.Standardizer().fit(Yc_train)
    pos_scaler = module.Standardizer().fit(Yp_train)
    pos_scale = float(np.mean(pos_scaler.std))
    current_scale = float(np.mean(cur_scaler.std))

    train_ds = CurrentWindowDataset(X_train, Yp_train, Yc_train, x_scaler, cur_scaler)
    val_ds = CurrentWindowDataset(X_val, Yp_val, Yc_val, x_scaler, cur_scaler)
    test_ds = CurrentWindowDataset(X_test, Yp_test, Yc_test, x_scaler, cur_scaler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(
        f'device={device}, dataset={args.dataset_label}, '
        f'train_windows={len(X_train)}, val_windows={len(X_val)}, test_windows={len(X_test)}'
    )
    print(
        f'loss = pos_loss + {args.alpha_cur} * current_loss + '
        f'{args.beta_smooth} * smoothness_loss'
    )

    model = LSTMCurrentPredictor(input_dim=5, hidden_dim=args.hidden_dim, pred_len=args.pred_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float('inf')
    best_state = None
    history = []

    for ep in range(1, args.epochs + 1):
        train_losses = run_epoch(
            model, train_loader, opt, cur_scaler, args.dt, pos_scale, current_scale,
            args.alpha_cur, args.beta_smooth, device,
        )
        val_losses = evaluate_losses(
            model, val_loader, cur_scaler, args.dt, pos_scale, current_scale,
            args.alpha_cur, args.beta_smooth, device,
        )
        row = {
            'epoch': ep,
            'train_total': train_losses[0],
            'train_pos': train_losses[1],
            'train_current': train_losses[2],
            'train_smooth': train_losses[3],
            'val_total': val_losses[0],
            'val_pos': val_losses[1],
            'val_current': val_losses[2],
            'val_smooth': val_losses[3],
        }
        history.append(row)
        if val_losses[0] < best_val:
            best_val = val_losses[0]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
            print(
                f'[LSTM_TRUE_CURRENT] epoch={ep:03d} '
                f'train_total={train_losses[0]:.6f} val_total={val_losses[0]:.6f} '
                f'val_pos={val_losses[1]:.6f} val_current={val_losses[2]:.6f}'
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_pos, true_pos, pred_cur, true_cur, X_raw_test = predict_all(model, test_loader, cur_scaler, args.dt, device)
    result = {
        'model': 'LSTM_TrueCurrent_Integral',
        **module.metrics_np(pred_pos, true_pos),
        **current_metrics(pred_cur, true_cur),
        'hidden_dim': args.hidden_dim,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'alpha_cur': args.alpha_cur,
        'beta_smooth': args.beta_smooth,
        'epochs': args.epochs,
    }
    result_df = pd.DataFrame([result])
    print(result_df)

    result_csv = out_dir / f'results_lstm_true_current_integral_{args.dataset_label}.csv'
    history_csv = out_dir / f'train_history_lstm_true_current_integral_{args.dataset_label}.csv'
    model_path = out_dir / f'lstm_true_current_integral_{args.dataset_label}.pt'
    result_df.to_csv(result_csv, index=False)
    pd.DataFrame(history).to_csv(history_csv, index=False)
    torch.save(model.state_dict(), model_path)
    print(f'[OK] saved {result_csv}')
    print(f'[OK] saved {history_csv}')
    print(f'[OK] saved {model_path}')

    save_plots(out_dir, args.dataset_label, args.sample_idx, X_raw_test, true_pos, pred_pos, true_cur, pred_cur)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='./auv_exp_traj24_extreme')
    parser.add_argument('--dataset_label', type=str, default='current')
    parser.add_argument('--input_len', type=int, default=80)
    parser.add_argument('--pred_len', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dt', type=float, default=0.2)
    parser.add_argument('--alpha_cur', type=float, default=0.1)
    parser.add_argument('--beta_smooth', type=float, default=0.01)
    parser.add_argument('--sample_idx', type=int, default=50)
    parser.add_argument('--log_every', type=int, default=5)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()


