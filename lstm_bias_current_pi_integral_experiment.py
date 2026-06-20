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
    spec = importlib.util.spec_from_file_location('auv_smoke_bias_current_pi', SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BiasCurrentWindowDataset(Dataset):
    def __init__(self, X_raw, Y_pos_raw, Y_cur_raw, Y_bias_raw, x_scaler, cur_scaler, bias_scaler):
        self.X_raw = X_raw.astype(np.float32)
        self.Y_pos_raw = Y_pos_raw.astype(np.float32)
        self.Y_cur_raw = Y_cur_raw.astype(np.float32)
        self.Y_bias_raw = Y_bias_raw.astype(np.float32)
        self.X = x_scaler.transform(self.X_raw).astype(np.float32)
        self.Y_cur = cur_scaler.transform(self.Y_cur_raw).astype(np.float32)
        self.Y_bias = bias_scaler.transform(self.Y_bias_raw).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.Y_cur[idx]),
            torch.from_numpy(self.Y_bias[idx]),
            torch.from_numpy(self.X_raw[idx]),
            torch.from_numpy(self.Y_pos_raw[idx]),
            torch.from_numpy(self.Y_cur_raw[idx]),
            torch.from_numpy(self.Y_bias_raw[idx]),
        )


def make_bias_current_windows(data: np.ndarray, input_len: int, pred_len: int):
    """
    Input: [nav_x, nav_y, v, a, theta]
    Position target: future true position [true_x, true_y]
    Current target: future true current/disturbance [cx, cy]
    Bias target: current accumulated offset [true_last - nav_last]
    """
    Xs, Y_pos, Y_cur, Y_bias = [], [], [], []
    for traj in data:
        features = traj[:, [3, 4, 5, 6, 7]]
        target_pos = traj[:, [1, 2]]
        target_cur = traj[:, [8, 9]]
        true_xy = traj[:, [1, 2]]
        nav_xy = traj[:, [3, 4]]
        max_i = len(traj) - input_len - pred_len
        for i in range(max_i):
            last_idx = i + input_len - 1
            Xs.append(features[i:i + input_len])
            Y_pos.append(target_pos[i + input_len:i + input_len + pred_len])
            Y_cur.append(target_cur[i + input_len:i + input_len + pred_len])
            Y_bias.append(true_xy[last_idx] - nav_xy[last_idx])
    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(Y_pos, dtype=np.float32),
        np.asarray(Y_cur, dtype=np.float32),
        np.asarray(Y_bias, dtype=np.float32),
    )


class BiasCurrentPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, pred_len=30, cell='rnn'):
        super().__init__()
        self.pred_len = pred_len
        self.cell = cell
        if cell == 'rnn':
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=1, batch_first=True, nonlinearity='tanh')
        elif cell == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        else:
            raise ValueError(cell)
        self.shared = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU())
        self.current_head = nn.Linear(64, pred_len * 2)
        self.bias_head = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        h = self.shared(out[:, -1, :])
        cur = self.current_head(h).view(x.size(0), self.pred_len, 2)
        bias = self.bias_head(h)
        return cur, bias


def inverse_norm(x_norm, scaler, device):
    mean = torch.tensor(scaler.mean, dtype=torch.float32, device=device)
    std = torch.tensor(scaler.std, dtype=torch.float32, device=device)
    return x_norm * std + mean


def integrate_with_bias_current(x_raw, bias_raw, cur_raw, dt: float):
    """
    x_raw last feature order: [nav_x, nav_y, v, a, theta].
    bias_raw: [B,2] accumulated position offset in meters.
    cur_raw: [B,K,2] future current/disturbance velocity in m/s.
    """
    last = x_raw[:, -1, :]
    nav0 = last[:, :2]
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
    start = (nav0 + bias_raw).unsqueeze(1)
    delta = (body_vel + cur_raw) * dt
    return start + torch.cumsum(delta, dim=1)


def effective_current_from_true(x_raw, y_pos_raw, y_bias_raw, dt: float):
    """
    Build a deployment-consistent equivalent current target from true trajectory finite differences.
    Uses only the current observed v0, a0 and theta0 for self-motion approximation.
    """
    last = x_raw[:, -1, :]
    nav0 = last[:, :2]
    v0 = last[:, 2]
    a0 = last[:, 3]
    theta = last[:, 4]

    K = y_pos_raw.shape[1]
    steps = torch.arange(K, dtype=torch.float32, device=y_pos_raw.device).unsqueeze(0)
    v_i = torch.clamp(v0.unsqueeze(1) + a0.unsqueeze(1) * steps * dt, min=0.0)
    self_motion = torch.stack([
        v_i * torch.cos(theta).unsqueeze(1),
        v_i * torch.sin(theta).unsqueeze(1),
    ], dim=-1)

    true_start = nav0 + y_bias_raw
    prev_true = torch.cat([true_start.unsqueeze(1), y_pos_raw[:, :-1, :]], dim=1)
    true_velocity = (y_pos_raw - prev_true) / dt
    return true_velocity - self_motion


def smoothness_loss(cur_raw, current_scale: float):
    if cur_raw.shape[1] < 2:
        return torch.tensor(0.0, dtype=cur_raw.dtype, device=cur_raw.device)
    diff = cur_raw[:, 1:, :] - cur_raw[:, :-1, :]
    return torch.mean(diff ** 2) / (current_scale ** 2)


def unpack_batch(batch, device):
    xb, ycur, ybias, xraw, ypos, ycur_raw, ybias_raw = batch
    return (
        xb.to(device),
        ycur.to(device),
        ybias.to(device),
        xraw.to(device),
        ypos.to(device),
        ycur_raw.to(device),
        ybias_raw.to(device),
    )


def run_epoch(model, loader, opt, cur_scaler, bias_scaler, dt, pos_scale, current_scale,
              alpha_cur, alpha_bias, alpha_pi, beta_smooth, device):
    model.train()
    totals = np.zeros(6, dtype=np.float64)
    for batch in loader:
        xb, ycur, ybias, xraw, ypos, ycur_raw, ybias_raw = unpack_batch(batch, device)
        pred_cur_norm, pred_bias_norm = model(xb)
        pred_cur_raw = inverse_norm(pred_cur_norm, cur_scaler, device)
        pred_bias_raw = inverse_norm(pred_bias_norm, bias_scaler, device)
        pred_pos = integrate_with_bias_current(xraw, pred_bias_raw, pred_cur_raw, dt)

        loss_pos = torch.mean((pred_pos - ypos) ** 2) / (pos_scale ** 2)
        loss_cur = torch.mean((pred_cur_norm - ycur) ** 2)
        loss_bias = torch.mean((pred_bias_norm - ybias) ** 2)
        loss_pi = torch.mean((pred_cur_raw - effective_current_from_true(xraw, ypos, ybias_raw, dt)) ** 2) / (current_scale ** 2)
        loss_smooth = smoothness_loss(pred_cur_raw, current_scale)
        loss = loss_pos + alpha_cur * loss_cur + alpha_bias * loss_bias + alpha_pi * loss_pi + beta_smooth * loss_smooth

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        n = len(xb)
        totals += np.array([loss.item(), loss_pos.item(), loss_cur.item(), loss_bias.item(), loss_pi.item(), loss_smooth.item()]) * n
    return totals / len(loader.dataset)


@torch.no_grad()
def evaluate_losses(model, loader, cur_scaler, bias_scaler, dt, pos_scale, current_scale,
                    alpha_cur, alpha_bias, alpha_pi, beta_smooth, device):
    model.eval()
    totals = np.zeros(6, dtype=np.float64)
    for batch in loader:
        xb, ycur, ybias, xraw, ypos, ycur_raw, ybias_raw = unpack_batch(batch, device)
        pred_cur_norm, pred_bias_norm = model(xb)
        pred_cur_raw = inverse_norm(pred_cur_norm, cur_scaler, device)
        pred_bias_raw = inverse_norm(pred_bias_norm, bias_scaler, device)
        pred_pos = integrate_with_bias_current(xraw, pred_bias_raw, pred_cur_raw, dt)

        loss_pos = torch.mean((pred_pos - ypos) ** 2) / (pos_scale ** 2)
        loss_cur = torch.mean((pred_cur_norm - ycur) ** 2)
        loss_bias = torch.mean((pred_bias_norm - ybias) ** 2)
        loss_pi = torch.mean((pred_cur_raw - effective_current_from_true(xraw, ypos, ybias_raw, dt)) ** 2) / (current_scale ** 2)
        loss_smooth = smoothness_loss(pred_cur_raw, current_scale)
        loss = loss_pos + alpha_cur * loss_cur + alpha_bias * loss_bias + alpha_pi * loss_pi + beta_smooth * loss_smooth

        n = len(xb)
        totals += np.array([loss.item(), loss_pos.item(), loss_cur.item(), loss_bias.item(), loss_pi.item(), loss_smooth.item()]) * n
    return totals / len(loader.dataset)


@torch.no_grad()
def predict_all(model, loader, cur_scaler, bias_scaler, dt, device):
    model.eval()
    pred_pos_all, true_pos_all = [], []
    pred_cur_all, true_cur_all = [], []
    pred_bias_all, true_bias_all = [], []
    xraw_all = []
    for batch in loader:
        xb, _, _, xraw, ypos, ycur_raw, ybias_raw = batch
        xb = xb.to(device)
        xraw_dev = xraw.to(device)
        pred_cur_norm, pred_bias_norm = model(xb)
        pred_cur_raw = inverse_norm(pred_cur_norm, cur_scaler, device)
        pred_bias_raw = inverse_norm(pred_bias_norm, bias_scaler, device)
        pred_pos = integrate_with_bias_current(xraw_dev, pred_bias_raw, pred_cur_raw, dt)

        pred_pos_all.append(pred_pos.cpu().numpy())
        true_pos_all.append(ypos.numpy())
        pred_cur_all.append(pred_cur_raw.cpu().numpy())
        true_cur_all.append(ycur_raw.numpy())
        pred_bias_all.append(pred_bias_raw.cpu().numpy())
        true_bias_all.append(ybias_raw.numpy())
        xraw_all.append(xraw.numpy())

    return (
        np.concatenate(pred_pos_all),
        np.concatenate(true_pos_all),
        np.concatenate(pred_cur_all),
        np.concatenate(true_cur_all),
        np.concatenate(pred_bias_all),
        np.concatenate(true_bias_all),
        np.concatenate(xraw_all),
    )


def vector_metrics(prefix, pred, true):
    err = pred - true
    out = {
        f'{prefix}_RMSE': float(np.sqrt(np.mean(err ** 2))),
        f'{prefix}_MAE': float(np.mean(np.abs(err))),
        f'{prefix}_MSE': float(np.mean(err ** 2)),
    }
    flat_pred_x = pred[..., 0].ravel()
    flat_true_x = true[..., 0].ravel()
    flat_pred_y = pred[..., 1].ravel()
    flat_true_y = true[..., 1].ravel()
    out[f'{prefix}_Corr_x'] = float(np.corrcoef(flat_pred_x, flat_true_x)[0, 1])
    out[f'{prefix}_Corr_y'] = float(np.corrcoef(flat_pred_y, flat_true_y)[0, 1])
    return out


def save_plots(out_dir, label, model_tag, sample_idx, X_raw, true_pos, pred_pos, true_cur, pred_cur, true_bias, pred_bias):
    sample_idx = int(np.clip(sample_idx, 0, len(true_pos) - 1))

    fig_path = out_dir / f'fig_prediction_{model_tag}_{label}.png'
    plt.figure(figsize=(7, 6))
    hist = X_raw[sample_idx, :, :2]
    corrected_start = hist[-1] + pred_bias[sample_idx]
    true_start = hist[-1] + true_bias[sample_idx]
    plt.plot(hist[:, 0], hist[:, 1], 'ko-', linewidth=1, label='input history(nav)')
    plt.plot(true_pos[sample_idx, :, 0], true_pos[sample_idx, :, 1], 'g-', linewidth=2, label='ground truth')
    plt.plot(pred_pos[sample_idx, :, 0], pred_pos[sample_idx, :, 1], '--', linewidth=2, label=model_tag)
    plt.scatter([corrected_start[0]], [corrected_start[1]], marker='x', s=60, label='pred corrected start')
    plt.scatter([true_start[0]], [true_start[1]], marker='+', s=60, label='true corrected start')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f'{model_tag} bias-current integral trajectory: {label}')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f'[OK] saved {fig_path}')

    cur_fig_path = out_dir / f'fig_current_{model_tag}_{label}.png'
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
    plt.title(f'{model_tag} future current prediction: {label}')
    plt.tight_layout()
    plt.savefig(cur_fig_path, dpi=200)
    plt.close()
    print(f'[OK] saved {cur_fig_path}')

    sample_csv = out_dir / f'prediction_sample_{model_tag}_{label}.csv'
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


def pi_metrics(pred_cur, x_raw, true_pos, true_bias, dt, current_scale):
    xraw_t = torch.from_numpy(x_raw.astype(np.float32))
    ypos_t = torch.from_numpy(true_pos.astype(np.float32))
    ybias_t = torch.from_numpy(true_bias.astype(np.float32))
    eff = effective_current_from_true(xraw_t, ypos_t, ybias_t, dt).numpy()
    err = pred_cur - eff
    return {
        'DatasetPI_RMSE': float(np.sqrt(np.mean(err ** 2))),
        'DatasetPI_MAE': float(np.mean(np.abs(err))),
        'DatasetPI_MSE': float(np.mean(err ** 2)),
    }


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
    X_train, Yp_train, Yc_train, Yb_train = make_bias_current_windows(train_traj, args.input_len, args.pred_len)
    X_val, Yp_val, Yc_val, Yb_val = make_bias_current_windows(val_traj, args.input_len, args.pred_len)
    X_test, Yp_test, Yc_test, Yb_test = make_bias_current_windows(test_traj, args.input_len, args.pred_len)

    x_scaler = module.Standardizer().fit(X_train)
    cur_scaler = module.Standardizer().fit(Yc_train)
    bias_scaler = module.Standardizer().fit(Yb_train)
    pos_scaler = module.Standardizer().fit(Yp_train)
    pos_scale = float(np.mean(pos_scaler.std))
    current_scale = float(np.mean(cur_scaler.std))

    train_ds = BiasCurrentWindowDataset(X_train, Yp_train, Yc_train, Yb_train, x_scaler, cur_scaler, bias_scaler)
    val_ds = BiasCurrentWindowDataset(X_val, Yp_val, Yc_val, Yb_val, x_scaler, cur_scaler, bias_scaler)
    test_ds = BiasCurrentWindowDataset(X_test, Yp_test, Yc_test, Yb_test, x_scaler, cur_scaler, bias_scaler)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model_tag = 'lstm_bias_current_pi_integral'
    print(
        f'device={device}, model={model_tag}, dataset={args.dataset_label}, '
        f'train_windows={len(X_train)}, val_windows={len(X_val)}, test_windows={len(X_test)}'
    )
    print(
        f'loss = pos_loss + {args.alpha_cur} * current_loss + '
        f'{args.alpha_bias} * bias_loss + {args.alpha_pi} * dataset_pi_loss + '
        f'{args.beta_smooth} * smoothness_loss'
    )

    model = BiasCurrentPredictor(input_dim=5, hidden_dim=args.hidden_dim, pred_len=args.pred_len, cell='lstm').to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float('inf')
    best_state = None
    history = []

    for ep in range(1, args.epochs + 1):
        train_losses = run_epoch(
            model, train_loader, opt, cur_scaler, bias_scaler, args.dt, pos_scale, current_scale,
            args.alpha_cur, args.alpha_bias, args.alpha_pi, args.beta_smooth, device,
        )
        val_losses = evaluate_losses(
            model, val_loader, cur_scaler, bias_scaler, args.dt, pos_scale, current_scale,
            args.alpha_cur, args.alpha_bias, args.alpha_pi, args.beta_smooth, device,
        )
        row = {
            'epoch': ep,
            'train_total': train_losses[0],
            'train_pos': train_losses[1],
            'train_current': train_losses[2],
            'train_bias': train_losses[3],
            'train_pi': train_losses[4],
            'train_smooth': train_losses[5],
            'val_total': val_losses[0],
            'val_pos': val_losses[1],
            'val_current': val_losses[2],
            'val_bias': val_losses[3],
            'val_pi': val_losses[4],
            'val_smooth': val_losses[5],
        }
        history.append(row)
        if val_losses[0] < best_val:
            best_val = val_losses[0]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
            print(
                f'[{model_tag}] epoch={ep:03d} train_total={train_losses[0]:.6f} '
                f'val_total={val_losses[0]:.6f} val_pos={val_losses[1]:.6f} '
                f'val_current={val_losses[2]:.6f} val_bias={val_losses[3]:.6f} val_pi={val_losses[4]:.6f}'
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_pos, true_pos, pred_cur, true_cur, pred_bias, true_bias, X_raw_test = predict_all(
        model, test_loader, cur_scaler, bias_scaler, args.dt, device,
    )
    result = {
        'model': model_tag.upper(),
        **module.metrics_np(pred_pos, true_pos),
        **vector_metrics('Current', pred_cur, true_cur),
        **vector_metrics('Bias', pred_bias, true_bias),
        **pi_metrics(pred_cur, X_raw_test, true_pos, true_bias, args.dt, current_scale),
        'hidden_dim': args.hidden_dim,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'alpha_cur': args.alpha_cur,
        'alpha_bias': args.alpha_bias,
        'alpha_pi': args.alpha_pi,
        'beta_smooth': args.beta_smooth,
        'epochs': args.epochs,
    }
    result_df = pd.DataFrame([result])
    print(result_df)

    result_csv = out_dir / f'results_{model_tag}_{args.dataset_label}.csv'
    history_csv = out_dir / f'train_history_{model_tag}_{args.dataset_label}.csv'
    model_path = out_dir / f'{model_tag}_{args.dataset_label}.pt'
    result_df.to_csv(result_csv, index=False)
    pd.DataFrame(history).to_csv(history_csv, index=False)
    torch.save(model.state_dict(), model_path)
    print(f'[OK] saved {result_csv}')
    print(f'[OK] saved {history_csv}')
    print(f'[OK] saved {model_path}')

    save_plots(out_dir, args.dataset_label, model_tag, args.sample_idx, X_raw_test,
               true_pos, pred_pos, true_cur, pred_cur, true_bias, pred_bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='./auv_exp_traj24_extreme')
    parser.add_argument('--dataset_label', type=str, default='current')
    parser.add_argument('--input_len', type=int, default=80)
    parser.add_argument('--pred_len', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dt', type=float, default=0.2)
    parser.add_argument('--alpha_cur', type=float, default=0.1)
    parser.add_argument('--alpha_bias', type=float, default=0.2)
    parser.add_argument('--alpha_pi', type=float, default=0.02)
    parser.add_argument('--beta_smooth', type=float, default=0.01)
    parser.add_argument('--sample_idx', type=int, default=50)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.cell = 'lstm'
    run(args)


if __name__ == '__main__':
    main()


