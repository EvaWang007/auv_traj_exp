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
PI_SRC = REPO_ROOT / 'lstm_bias_current_pi_integral_experiment.py'
BASE_SRC = REPO_ROOT / 'auv_trajectory_smoke_experiment.py'


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BCPILSTMHyperPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, pred_len=40, dropout=0.0):
        super().__init__()
        self.pred_len = pred_len
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.current_head = nn.Linear(64, pred_len * 2)
        self.bias_head = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        h = self.shared(out[:, -1, :])
        cur = self.current_head(h).view(x.size(0), self.pred_len, 2)
        bias = self.bias_head(h)
        return cur, bias


@dataclass
class SearchSpace:
    hidden_dim: tuple = (64, 96, 128, 192)
    dropout: tuple = (0.0, 0.1, 0.2)
    lr: tuple = (3e-4, 5e-4, 1e-3)
    batch_size: tuple = (64, 128)
    alpha_cur: tuple = (0.05, 0.1, 0.2)
    alpha_bias: tuple = (0.1, 0.2, 0.3)
    alpha_pi: tuple = (0.005, 0.01, 0.02, 0.05)
    beta_smooth: tuple = (0.005, 0.01)

    def lists(self):
        return [
            list(self.hidden_dim),
            list(self.dropout),
            list(self.lr),
            list(self.batch_size),
            list(self.alpha_cur),
            list(self.alpha_bias),
            list(self.alpha_pi),
            list(self.beta_smooth),
        ]


def decode_position(position: np.ndarray, space: SearchSpace):
    names = ['hidden_dim', 'dropout', 'lr', 'batch_size', 'alpha_cur', 'alpha_bias', 'alpha_pi', 'beta_smooth']
    choices = space.lists()
    idxs = []
    params = {}
    for i, vals in enumerate(choices):
        idx = int(np.clip(np.round(position[i]), 0, len(vals) - 1))
        idxs.append(idx)
        params[names[i]] = vals[idx]
    return tuple(idxs), params


def sample_subset(*arrays, limit: int, seed: int):
    if limit <= 0 or len(arrays[0]) <= limit:
        return arrays
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(arrays[0]), size=limit, replace=False)
    return tuple(a[idx] for a in arrays)


def build_loaders(pi_module, X_train, Yp_train, Yc_train, Yb_train, X_val, Yp_val, Yc_val, Yb_val, batch_size):
    x_scaler = pi_module.load_src().Standardizer().fit(X_train)
    cur_scaler = pi_module.load_src().Standardizer().fit(Yc_train)
    bias_scaler = pi_module.load_src().Standardizer().fit(Yb_train)
    train_ds = pi_module.BiasCurrentWindowDataset(X_train, Yp_train, Yc_train, Yb_train, x_scaler, cur_scaler, bias_scaler)
    val_ds = pi_module.BiasCurrentWindowDataset(X_val, Yp_val, Yc_val, Yb_val, x_scaler, cur_scaler, bias_scaler)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, cur_scaler, bias_scaler


def train_candidate(pi_module, model, train_loader, val_loader, cur_scaler, bias_scaler, y_pos_train, params, args, device, name):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])
    pos_scaler = pi_module.load_src().Standardizer().fit(y_pos_train)
    pos_scale = float(np.mean(pos_scaler.std))
    current_scale = float(np.mean(cur_scaler.std))
    best_val = float('inf')
    best_state = None

    for ep in range(1, args.search_epochs + 1):
        pi_module.run_epoch(
            model, train_loader, opt, cur_scaler, bias_scaler, args.dt, pos_scale, current_scale,
            params['alpha_cur'], params['alpha_bias'], params['alpha_pi'], params['beta_smooth'], device,
        )
        val_losses = pi_module.evaluate_losses(
            model, val_loader, cur_scaler, bias_scaler, args.dt, pos_scale, current_scale,
            params['alpha_cur'], params['alpha_bias'], params['alpha_pi'], params['beta_smooth'], device,
        )
        if val_losses[0] < best_val:
            best_val = val_losses[0]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    pred_pos, true_pos, _, _, _, _, _ = pi_module.predict_all(model, val_loader, cur_scaler, bias_scaler, args.dt, device)
    err = pred_pos - true_pos
    dist = np.linalg.norm(err, axis=-1)
    ade = float(np.mean(dist))
    fde = float(np.mean(dist[:, -1]))
    return ade + 0.5 * fde


def final_train(pi_module, base_module, params, X_train, Yp_train, Yc_train, Yb_train, X_val, Yp_val, Yc_val, Yb_val, args, device):
    train_loader, val_loader, cur_scaler, bias_scaler = build_loaders(
        pi_module, X_train, Yp_train, Yc_train, Yb_train, X_val, Yp_val, Yc_val, Yb_val, params['batch_size']
    )
    pos_scaler = base_module.Standardizer().fit(Yp_train)
    pos_scale = float(np.mean(pos_scaler.std))
    current_scale = float(np.mean(cur_scaler.std))
    model = BCPILSTMHyperPredictor(hidden_dim=params['hidden_dim'], pred_len=args.pred_len, dropout=params['dropout']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])
    best_val = float('inf')
    best_state = None
    history = []
    for ep in range(1, args.final_epochs + 1):
        train_losses = pi_module.run_epoch(
            model, train_loader, opt, cur_scaler, bias_scaler, args.dt, pos_scale, current_scale,
            params['alpha_cur'], params['alpha_bias'], params['alpha_pi'], params['beta_smooth'], device,
        )
        val_losses = pi_module.evaluate_losses(
            model, val_loader, cur_scaler, bias_scaler, args.dt, pos_scale, current_scale,
            params['alpha_cur'], params['alpha_bias'], params['alpha_pi'], params['beta_smooth'], device,
        )
        history.append({
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
        })
        if val_losses[0] < best_val:
            best_val = val_losses[0]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % args.log_every == 0 or ep == args.final_epochs:
            print(
                f"[PSO_BCPI_LSTM_ADEFDE_final] epoch={ep:03d} train_total={train_losses[0]:.6f} "
                f"val_total={val_losses[0]:.6f} val_pos={val_losses[1]:.6f} "
                f"val_current={val_losses[2]:.6f} val_bias={val_losses[3]:.6f} val_pi={val_losses[4]:.6f}"
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_loader, val_loader, cur_scaler, bias_scaler, history


def run(args):
    pi_module = load_module(PI_SRC, 'bcpi_pi_module')
    base_module = load_module(BASE_SRC, 'bcpi_base_module')
    base_module.set_seed(args.seed)
    out_dir = Path(args.out_dir)
    dataset_path = out_dir / f'dataset_{args.dataset_label}.npz'
    if not dataset_path.exists():
        raise FileNotFoundError(f'{dataset_path} not found')

    data = np.load(dataset_path)['data']
    train_traj, val_traj, test_traj = base_module.split_by_trajectory(data, 0.7, 0.1)
    X_train, Yp_train, Yc_train, Yb_train = pi_module.make_bias_current_windows(train_traj, args.input_len, args.pred_len)
    X_val, Yp_val, Yc_val, Yb_val = pi_module.make_bias_current_windows(val_traj, args.input_len, args.pred_len)
    X_test, Yp_test, Yc_test, Yb_test = pi_module.make_bias_current_windows(test_traj, args.input_len, args.pred_len)

    search_train = sample_subset(X_train, Yp_train, Yc_train, Yb_train, limit=args.search_train_limit, seed=args.seed + 101)
    search_val = sample_subset(X_val, Yp_val, Yc_val, Yb_val, limit=args.search_val_limit, seed=args.seed + 202)
    X_train_s, Yp_train_s, Yc_train_s, Yb_train_s = search_train
    X_val_s, Yp_val_s, Yc_val_s, Yb_val_s = search_val

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(
        f"device={device}, dataset={args.dataset_label}, train_windows={len(X_train)}, val_windows={len(X_val)}, "
        f"test_windows={len(X_test)}, search_train_windows={len(X_train_s)}, search_val_windows={len(X_val_s)}"
    )

    space = SearchSpace()
    dims = [len(v) - 1 for v in space.lists()]
    rng = np.random.default_rng(args.seed)
    positions = np.stack([rng.uniform(0, dim, size=args.population) for dim in dims], axis=1)
    velocities = np.zeros_like(positions)
    pbest_pos = positions.copy()
    pbest_fit = np.full(args.population, np.inf)
    gbest_pos = None
    gbest_fit = np.inf
    cache = {}
    history_rows = []

    for it in range(1, args.iterations + 1):
        print(f'\n=== PSO-BCPI-ADEFDE iteration {it}/{args.iterations} ===')
        for p in range(args.population):
            key, params = decode_position(positions[p], space)
            if key in cache:
                fitness = cache[key]
                from_cache = True
            else:
                train_loader, val_loader, cur_scaler, bias_scaler = build_loaders(
                    pi_module, X_train_s, Yp_train_s, Yc_train_s, Yb_train_s,
                    X_val_s, Yp_val_s, Yc_val_s, Yb_val_s, params['batch_size']
                )
                model = BCPILSTMHyperPredictor(hidden_dim=params['hidden_dim'], pred_len=args.pred_len, dropout=params['dropout'])
                fitness = train_candidate(pi_module, model, train_loader, val_loader, cur_scaler, bias_scaler, Yp_train_s, params, args, device, f'candidate-{p}')
                cache[key] = fitness
                from_cache = False
            if fitness < pbest_fit[p]:
                pbest_fit[p] = fitness
                pbest_pos[p] = positions[p].copy()
            if fitness < gbest_fit:
                gbest_fit = fitness
                gbest_pos = positions[p].copy()
            row = {'iteration': it, 'particle': p, 'fitness_ade_fde': fitness, 'from_cache': from_cache, **params}
            history_rows.append(row)
            print(f"particle={p} fitness={fitness:.6f} cache={from_cache} params={params}")

        assert gbest_pos is not None
        for p in range(args.population):
            r1 = rng.random(len(dims))
            r2 = rng.random(len(dims))
            velocities[p] = args.w * velocities[p] + args.c1 * r1 * (pbest_pos[p] - positions[p]) + args.c2 * r2 * (gbest_pos - positions[p])
            positions[p] = np.clip(positions[p] + velocities[p], 0, dims)
        _, best_params = decode_position(gbest_pos, space)
        print(f'[PSO-BCPI-ADEFDE] best_fitness={gbest_fit:.6f} best_params={best_params}')

    _, best_params = decode_position(gbest_pos, space)
    history_csv = out_dir / f'pso_search_history_bcpi_lstm_{args.dataset_label}.csv'
    best_json = out_dir / f'best_hparams_pso_bcpi_lstm_adefde_{args.dataset_label}.json'
    pd.DataFrame(history_rows).to_csv(history_csv, index=False)
    with open(best_json, 'w', encoding='utf-8') as f:
        json.dump({'best_params': best_params, 'best_fitness_ade_fde': gbest_fit}, f, indent=2)
    print(f'[OK] saved {history_csv}')
    print(f'[OK] saved {best_json}')

    final_model, _, _, cur_scaler, bias_scaler, final_history = final_train(
        pi_module, base_module, best_params, X_train, Yp_train, Yc_train, Yb_train,
        X_val, Yp_val, Yc_val, Yb_val, args, device,
    )

    x_scaler = base_module.Standardizer().fit(X_train)
    test_ds = pi_module.BiasCurrentWindowDataset(X_test, Yp_test, Yc_test, Yb_test, x_scaler, cur_scaler, bias_scaler)
    test_loader = DataLoader(test_ds, batch_size=best_params['batch_size'], shuffle=False)
    pred_pos, true_pos, pred_cur, true_cur, pred_bias, true_bias, X_raw_test = pi_module.predict_all(
        final_model, test_loader, cur_scaler, bias_scaler, args.dt, device,
    )

    result = {
        'model': 'PSO_BCPI_LSTM_ADEFDE',
        **base_module.metrics_np(pred_pos, true_pos),
        **pi_module.vector_metrics('Current', pred_cur, true_cur),
        **pi_module.vector_metrics('Bias', pred_bias, true_bias),
        **pi_module.pi_metrics(pred_cur, X_raw_test, true_pos, true_bias, args.dt, float(np.mean(cur_scaler.std))),
        **best_params,
        'fitness_ade_fde': gbest_fit,
        'final_epochs': args.final_epochs,
    }
    result_df = pd.DataFrame([result])
    print(result_df)

    result_csv = out_dir / f'results_pso_bcpi_lstm_adefde_{args.dataset_label}.csv'
    final_history_csv = out_dir / f'train_history_pso_bcpi_lstm_adefde_{args.dataset_label}.csv'
    model_path = out_dir / f'pso_bcpi_lstm_adefde_{args.dataset_label}.pt'
    result_df.to_csv(result_csv, index=False)
    pd.DataFrame(final_history).to_csv(final_history_csv, index=False)
    torch.save(final_model.state_dict(), model_path)
    print(f'[OK] saved {result_csv}')
    print(f'[OK] saved {final_history_csv}')
    print(f'[OK] saved {model_path}')

    pi_module.save_plots(
        out_dir, args.dataset_label, 'pso_bcpi_lstm_adefde', args.sample_idx,
        X_raw_test, true_pos, pred_pos, true_cur, pred_cur, true_bias, pred_bias,
    )

    conv_fig = out_dir / f'fig_pso_bcpi_convergence_{args.dataset_label}.png'
    hist_df = pd.DataFrame(history_rows)
    best_curve = hist_df.groupby('iteration')['fitness_ade_fde'].min().cummin()
    plt.figure(figsize=(7, 4))
    plt.plot(best_curve.index, best_curve.values, marker='o')
    plt.grid(True)
    plt.xlabel('PSO iteration')
    plt.ylabel('best ADE + 0.5*FDE')
    plt.title(f'PSO-BCPI-ADEFDE-LSTM convergence: {args.dataset_label}')
    plt.tight_layout()
    plt.savefig(conv_fig, dpi=200)
    plt.close()
    print(f'[OK] saved {conv_fig}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='./auv_exp_traj24_extreme')
    parser.add_argument('--dataset_label', type=str, default='current')
    parser.add_argument('--input_len', type=int, default=80)
    parser.add_argument('--pred_len', type=int, default=40)
    parser.add_argument('--population', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--search_epochs', type=int, default=6)
    parser.add_argument('--final_epochs', type=int, default=30)
    parser.add_argument('--search_train_limit', type=int, default=20000)
    parser.add_argument('--search_val_limit', type=int, default=4000)
    parser.add_argument('--dt', type=float, default=0.2)
    parser.add_argument('--w', type=float, default=0.55)
    parser.add_argument('--c1', type=float, default=1.4)
    parser.add_argument('--c2', type=float, default=1.4)
    parser.add_argument('--sample_idx', type=int, default=50)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()


