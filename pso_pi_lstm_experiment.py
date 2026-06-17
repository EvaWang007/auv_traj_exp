#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / 'auv_trajectory_smoke_experiment.py'


def load_src():
    spec = importlib.util.spec_from_file_location('auv_smoke_pso_pi', SRC)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LSTMHyperPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, pred_len=30, num_layers=1, dropout=0.0):
        super().__init__()
        self.pred_len = pred_len
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * 2),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).view(x.size(0), self.pred_len, 2)


@dataclass
class SearchSpace:
    hidden_dim: tuple = (64, 96, 128, 192, 256)
    num_layers: tuple = (1, 2, 3)
    dropout: tuple = (0.0, 0.1, 0.2, 0.3, 0.4)
    lr: tuple = (1e-4, 3e-4, 5e-4, 1e-3, 3e-3)
    batch_size: tuple = (32, 64, 128)
    lambda_phy: tuple = (0.01, 0.02, 0.05, 0.1)

    def lists(self):
        return [
            list(self.hidden_dim),
            list(self.num_layers),
            list(self.dropout),
            list(self.lr),
            list(self.batch_size),
            list(self.lambda_phy),
        ]


def decode_position(position: np.ndarray, space: SearchSpace):
    choices = space.lists()
    names = ['hidden_dim', 'num_layers', 'dropout', 'lr', 'batch_size', 'lambda_phy']
    idxs = []
    params = {}
    for i, vals in enumerate(choices):
        idx = int(np.clip(np.round(position[i]), 0, len(vals) - 1))
        idxs.append(idx)
        params[names[i]] = vals[idx]
    return tuple(idxs), params


def sample_subset(X, Y, limit: int, seed: int):
    if limit <= 0 or len(X) <= limit:
        return X, Y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=limit, replace=False)
    return X[idx], Y[idx]


def build_loaders(module, X_train, Y_train, X_val, Y_val, batch_size):
    x_scaler = module.Standardizer().fit(X_train)
    y_scaler = module.Standardizer().fit(Y_train)
    train_ds = module.AUVDataset(X_train, Y_train, x_scaler, y_scaler)
    val_ds = module.AUVDataset(X_val, Y_val, x_scaler, y_scaler)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, x_scaler, y_scaler


def train_model(module, model, train_loader, val_loader, y_scaler, dt, lambda_phy, epochs, lr, device, name):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    best_state = None
    position_scale = float(np.mean(y_scaler.std))

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb, xraw, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xraw = xraw.to(device)
            pred = model(xb)
            data_loss = torch.mean((pred - yb) ** 2)
            phy = module.physics_loss_lstm(pred, xraw, y_scaler, dt=dt, position_scale=position_scale)
            loss = data_loss + lambda_phy * phy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += loss.item() * len(xb)

        val_loss = evaluate_data_loss(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep == epochs:
            print(f'[{name}] epoch={ep:03d} train_loss={total/len(train_loader.dataset):.6f} val_mse={val_loss:.6f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_data_loss(model, loader, device):
    model.eval()
    total = 0.0
    for xb, yb, _, _ in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = torch.mean((pred - yb) ** 2)
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def predict_model(module, model, loader, y_scaler, device):
    model.eval()
    preds, trues = [], []
    for xb, _, _, yraw in loader:
        xb = xb.to(device)
        pred_norm = model(xb).cpu().numpy()
        pred = y_scaler.inverse_transform(pred_norm)
        preds.append(pred)
        trues.append(yraw.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def evaluate_candidate(module, params, X_train_s, Y_train_s, X_val_s, Y_val_s, pred_len, dt, search_epochs, device):
    train_loader, val_loader, _, y_scaler = build_loaders(module, X_train_s, Y_train_s, X_val_s, Y_val_s, params['batch_size'])
    model = LSTMHyperPredictor(
        input_dim=5,
        hidden_dim=params['hidden_dim'],
        pred_len=pred_len,
        num_layers=params['num_layers'],
        dropout=params['dropout'],
    )
    model = train_model(
        module,
        model,
        train_loader,
        val_loader,
        y_scaler,
        dt,
        params['lambda_phy'],
        epochs=search_epochs,
        lr=params['lr'],
        device=device,
        name=f'PSO-PI-candidate-{params}',
    )
    pred_val, true_val = predict_model(module, model, val_loader, y_scaler, device)
    return module.metrics_np(pred_val, true_val)['RMSE']


def run_pso_pi(module, args, out_dir: Path):
    dataset_path = out_dir / f'dataset_{args.dataset_label}.npz'
    if not dataset_path.exists():
        raise FileNotFoundError(f'{dataset_path} not found')

    data = np.load(dataset_path)['data']
    train_traj, val_traj, test_traj = module.split_by_trajectory(data, 0.7, 0.1)
    X_train, Y_train = module.make_windows(train_traj, args.input_len, args.pred_len)
    X_val, Y_val = module.make_windows(val_traj, args.input_len, args.pred_len)
    X_test, Y_test = module.make_windows(test_traj, args.input_len, args.pred_len)

    X_train_s, Y_train_s = sample_subset(X_train, Y_train, args.search_train_limit, args.seed + 101)
    X_val_s, Y_val_s = sample_subset(X_val, Y_val, args.search_val_limit, args.seed + 202)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(
        f"device={device}, dataset={args.dataset_label}, "
        f"train_windows={len(X_train)}, val_windows={len(X_val)}, test_windows={len(X_test)}, "
        f"search_train_windows={len(X_train_s)}, search_val_windows={len(X_val_s)}"
    )

    space = SearchSpace()
    dims = [len(v) - 1 for v in space.lists()]
    rng = np.random.default_rng(args.seed)

    positions = np.stack([rng.uniform(0, dim, size=args.population) for dim in dims], axis=1)
    velocities = np.zeros_like(positions)
    pbest_pos = positions.copy()
    pbest_fit = np.full(args.population, np.inf, dtype=np.float64)
    gbest_pos = None
    gbest_fit = np.inf
    cache = {}
    history_rows = []

    for it in range(1, args.iterations + 1):
        print(f'\n=== PSO-PI iteration {it}/{args.iterations} ===')
        for p in range(args.population):
            key, params = decode_position(positions[p], space)
            if key in cache:
                fitness = cache[key]
                from_cache = True
            else:
                fitness = evaluate_candidate(
                    module, params, X_train_s, Y_train_s, X_val_s, Y_val_s,
                    args.pred_len, args.dt, args.search_epochs, device,
                )
                cache[key] = fitness
                from_cache = False

            history_rows.append({
                'iteration': it,
                'particle': p,
                'fitness_rmse': fitness,
                'from_cache': from_cache,
                **params,
            })
            print(f'iter={it:02d} particle={p:02d} fitness={fitness:.6f} params={params} cache={from_cache}')

            if fitness < pbest_fit[p]:
                pbest_fit[p] = fitness
                pbest_pos[p] = positions[p].copy()
            if fitness < gbest_fit:
                gbest_fit = fitness
                gbest_pos = positions[p].copy()

        w = args.inertia_max - (args.inertia_max - args.inertia_min) * (it - 1) / max(args.iterations - 1, 1)
        for p in range(args.population):
            r1 = rng.random(len(dims))
            r2 = rng.random(len(dims))
            velocities[p] = (
                w * velocities[p]
                + args.c1 * r1 * (pbest_pos[p] - positions[p])
                + args.c2 * r2 * (gbest_pos - positions[p])
            )
            positions[p] = positions[p] + velocities[p]
            for d, dim in enumerate(dims):
                positions[p, d] = np.clip(positions[p, d], 0, dim)

        _, best_params = decode_position(gbest_pos, space)
        print(f'[PSO-PI] gbest_rmse={gbest_fit:.6f} best_params={best_params}')

    history_df = pd.DataFrame(history_rows)
    history_path = out_dir / f'pso_search_history_pi_lstm_{args.dataset_label}.csv'
    history_df.to_csv(history_path, index=False)

    _, best_params = decode_position(gbest_pos, space)
    best_params['fitness_rmse'] = float(gbest_fit)
    best_path = out_dir / f'best_hparams_pso_pi_lstm_{args.dataset_label}.json'
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2)

    print(f'[OK] saved {history_path}')
    print(f'[OK] saved {best_path}')

    train_loader, val_loader, _, y_scaler = build_loaders(module, X_train, Y_train, X_val, Y_val, best_params['batch_size'])
    test_x_scaler = module.Standardizer().fit(X_train)
    test_ds = module.AUVDataset(X_test, Y_test, test_x_scaler, y_scaler)
    test_loader = DataLoader(test_ds, batch_size=best_params['batch_size'], shuffle=False)

    final_model = LSTMHyperPredictor(
        input_dim=5,
        hidden_dim=best_params['hidden_dim'],
        pred_len=args.pred_len,
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout'],
    )
    final_model = train_model(
        module,
        final_model,
        train_loader,
        val_loader,
        y_scaler,
        args.dt,
        best_params['lambda_phy'],
        epochs=args.final_epochs,
        lr=best_params['lr'],
        device=device,
        name='PSO_PI_LSTM_final',
    )

    pred_test, true_test = predict_model(module, final_model, test_loader, y_scaler, device)
    metrics = module.metrics_np(pred_test, true_test)
    results_df = pd.DataFrame([{'model': 'PSO_PI_LSTM', **metrics, **best_params}])
    results_path = out_dir / f'results_pso_pi_lstm_{args.dataset_label}.csv'
    results_df.to_csv(results_path, index=False)
    print(results_df)
    print(f'[OK] saved {results_path}')

    model_path = out_dir / f'pso_pi_lstm_{args.dataset_label}.pt'
    torch.save(final_model.state_dict(), model_path)
    print(f'[OK] saved {model_path}')

    gbest_curve = history_df.groupby('iteration')['fitness_rmse'].min().cummin().reset_index()
    plt.figure(figsize=(6, 4))
    plt.plot(gbest_curve['iteration'], gbest_curve['fitness_rmse'], marker='o')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('Best validation RMSE')
    plt.title(f'PSO-PI search convergence: {args.dataset_label}')
    conv_path = out_dir / f'fig_pso_pi_convergence_{args.dataset_label}.png'
    plt.tight_layout()
    plt.savefig(conv_path, dpi=200)
    plt.close()
    print(f'[OK] saved {conv_path}')

    sample_idx = min(50, len(Y_test) - 1)
    plt.figure(figsize=(7, 6))
    hist = X_test[sample_idx, :, :2]
    gt = Y_test[sample_idx]
    pred = pred_test[sample_idx]
    plt.plot(hist[:, 0], hist[:, 1], 'ko-', label='input history(nav)', linewidth=1)
    plt.plot(gt[:, 0], gt[:, 1], 'g-', label='ground truth', linewidth=2)
    plt.plot(pred[:, 0], pred[:, 1], '--', label='PSO-PI-LSTM')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f'AUV PSO-PI-LSTM prediction sample: {args.dataset_label}')
    pred_fig_path = out_dir / f'fig_prediction_pso_pi_lstm_{args.dataset_label}.png'
    plt.tight_layout()
    plt.savefig(pred_fig_path, dpi=200)
    plt.close()
    print(f'[OK] saved {pred_fig_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--dataset_label', choices=['current', 'no_current'], default='current')
    parser.add_argument('--dt', type=float, default=0.2)
    parser.add_argument('--input_len', type=int, default=60)
    parser.add_argument('--pred_len', type=int, default=30)
    parser.add_argument('--population', type=int, default=6)
    parser.add_argument('--iterations', type=int, default=6)
    parser.add_argument('--search_epochs', type=int, default=8)
    parser.add_argument('--final_epochs', type=int, default=50)
    parser.add_argument('--search_train_limit', type=int, default=20000)
    parser.add_argument('--search_val_limit', type=int, default=4000)
    parser.add_argument('--inertia_max', type=float, default=0.9)
    parser.add_argument('--inertia_min', type=float, default=0.4)
    parser.add_argument('--c1', type=float, default=1.5)
    parser.add_argument('--c2', type=float, default=1.5)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    module = load_src()
    module.set_seed(args.seed)
    run_pso_pi(module, args, Path(args.out_dir))


if __name__ == '__main__':
    main()
