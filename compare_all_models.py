#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
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
    spec = importlib.util.spec_from_file_location('auv_smoke_compare', SRC)
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
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=effective_dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * 2),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).view(x.size(0), self.pred_len, 2)


class EncoderDecoderLSTMPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, pred_len=30, decoder_input_dim=2):
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
            decoder_input = pred_step.unsqueeze(1)
        return torch.cat(outputs, dim=1)


def get_decoder_init_norm(xraw: torch.Tensor, y_scaler, device: str) -> torch.Tensor:
    y_mean = torch.tensor(y_scaler.mean[:2], dtype=torch.float32, device=device)
    y_std = torch.tensor(y_scaler.std[:2], dtype=torch.float32, device=device)
    last_nav = xraw[:, -1, :2].to(device)
    return (last_nav - y_mean) / y_std


@torch.no_grad()
def predict_oneshot(model, loader, y_scaler, device):
    model.eval()
    preds = []
    for xb, _, _, _ in loader:
        xb = xb.to(device)
        pred_norm = model(xb).cpu().numpy()
        preds.append(y_scaler.inverse_transform(pred_norm))
    return np.concatenate(preds)


@torch.no_grad()
def predict_encdec(model, loader, y_scaler, device):
    model.eval()
    preds = []
    for xb, _, xraw, _ in loader:
        xb = xb.to(device)
        xraw = xraw.to(device)
        dec0 = get_decoder_init_norm(xraw, y_scaler, device)
        pred_norm = model(xb, dec0).cpu().numpy()
        preds.append(y_scaler.inverse_transform(pred_norm))
    return np.concatenate(preds)


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def maybe_add_results(results_frames, path: Path):
    if path.exists():
        results_frames.append(pd.read_csv(path))
    else:
        print(f'[WARN] missing results file: {path.name}')


def run_compare(module, args, out_dir: Path):
    dataset_path = out_dir / f'dataset_{args.dataset_label}.npz'
    if not dataset_path.exists():
        raise FileNotFoundError(f'{dataset_path} not found')

    data = np.load(dataset_path)['data']
    module.set_seed(args.seed)
    train_traj, val_traj, test_traj = module.split_by_trajectory(data, 0.7, 0.1)
    X_train, Y_train = module.make_windows(train_traj, args.input_len, args.pred_len)
    X_test, Y_test = module.make_windows(test_traj, args.input_len, args.pred_len)

    x_scaler = module.Standardizer().fit(X_train)
    y_scaler = module.Standardizer().fit(Y_train)
    test_ds = module.AUVDataset(X_test, Y_test, x_scaler, y_scaler)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f'device={device}, test_windows={len(test_ds)}')

    results_frames = []
    maybe_add_results(results_frames, out_dir / f'results_{args.dataset_label}.csv')
    maybe_add_results(results_frames, out_dir / f'results_pso_lstm_{args.dataset_label}.csv')
    maybe_add_results(results_frames, out_dir / f'results_encdec_only_{args.dataset_label}.csv')
    maybe_add_results(results_frames, out_dir / f'results_pso_pi_lstm_{args.dataset_label}.csv')

    if not results_frames:
        raise FileNotFoundError('No result csv files found to compare.')

    merged = pd.concat(results_frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=['model'], keep='last')
    merged = merged.sort_values('RMSE').reset_index(drop=True)
    results_out = out_dir / f'results_all_{args.dataset_label}.csv'
    merged.to_csv(results_out, index=False)
    print(merged[['model', 'ADE', 'FDE', 'RMSE', 'MAE']])
    print(f'[OK] saved {results_out}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ['ADE', 'FDE', 'RMSE']):
        ax.bar(merged['model'], merged[metric])
        ax.set_title(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    metrics_fig = out_dir / f'fig_metrics_all_{args.dataset_label}.png'
    plt.savefig(metrics_fig, dpi=200)
    plt.close()
    print(f'[OK] saved {metrics_fig}')

    predictions = {}
    true = Y_test
    raw_hist = X_test

    # EKF-like
    predictions['EKF_like_kinematic'] = module.ekf_like_predict(X_test, args.pred_len, args.dt)

    def load_state(model, path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model

    file_map = {
        'RNN': out_dir / f'rnn_{args.dataset_label}.pt',
        'PI_RNN_lambda_0.05': out_dir / f'pi_rnn_{args.dataset_label}.pt',
        'LSTM': out_dir / f'lstm_{args.dataset_label}.pt',
        'PI_LSTM_lambda_0.05': out_dir / f'pi_lstm_{args.dataset_label}.pt',
    }
    for model_name, path in file_map.items():
        if path.exists():
            cell = 'rnn' if 'RNN' in model_name and 'LSTM' not in model_name else 'lstm'
            model = module.RNNPredictor(input_dim=5, hidden_dim=args.base_hidden_dim, pred_len=args.pred_len, cell=cell)
            load_state(model, path)
            predictions[model_name] = predict_oneshot(model, test_loader, y_scaler, device)

    # PSO-LSTM
    pso_json = out_dir / f'best_hparams_pso_lstm_{args.dataset_label}.json'
    pso_ckpt = out_dir / f'pso_lstm_{args.dataset_label}.pt'
    if pso_json.exists() and pso_ckpt.exists():
        hp = load_json(pso_json)
        model = LSTMHyperPredictor(input_dim=5, hidden_dim=int(hp['hidden_dim']), pred_len=args.pred_len, num_layers=int(hp['num_layers']), dropout=float(hp['dropout']))
        load_state(model, pso_ckpt)
        predictions['PSO_LSTM'] = predict_oneshot(model, test_loader, y_scaler, device)

    # EncDec models
    encdec_files = {
        'EncDec_LSTM': out_dir / f'encdec_lstm_only_{args.dataset_label}.pt',
        'PI_EncDec_LSTM_lambda_0.05': out_dir / f'pi_encdec_lstm_only_{args.dataset_label}.pt',
    }
    for model_name, path in encdec_files.items():
        if path.exists():
            model = EncoderDecoderLSTMPredictor(input_dim=5, hidden_dim=args.encdec_hidden_dim, pred_len=args.pred_len, decoder_input_dim=2)
            load_state(model, path)
            predictions[model_name] = predict_encdec(model, test_loader, y_scaler, device)

    # PSO-PI-LSTM
    pso_pi_json = out_dir / f'best_hparams_pso_pi_lstm_{args.dataset_label}.json'
    pso_pi_ckpt = out_dir / f'pso_pi_lstm_{args.dataset_label}.pt'
    if pso_pi_json.exists() and pso_pi_ckpt.exists():
        hp = load_json(pso_pi_json)
        model = LSTMHyperPredictor(input_dim=5, hidden_dim=int(hp['hidden_dim']), pred_len=args.pred_len, num_layers=int(hp['num_layers']), dropout=float(hp['dropout']))
        load_state(model, pso_pi_ckpt)
        predictions['PSO_PI_LSTM'] = predict_oneshot(model, test_loader, y_scaler, device)

    sample_idx = min(args.sample_idx, len(Y_test) - 1)
    plt.figure(figsize=(8, 7))
    hist = raw_hist[sample_idx, :, :2]
    gt = true[sample_idx]
    plt.plot(hist[:, 0], hist[:, 1], 'ko-', label='input history(nav)', linewidth=1)
    plt.plot(gt[:, 0], gt[:, 1], 'g-', label='ground truth', linewidth=2)
    for model_name, pred in predictions.items():
        plt.plot(pred[sample_idx, :, 0], pred[sample_idx, :, 1], '--', label=model_name)
    plt.axis('equal')
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.title(f'All-model trajectory comparison: {args.dataset_label}')
    pred_fig = out_dir / f'fig_prediction_all_models_{args.dataset_label}.png'
    plt.tight_layout()
    plt.savefig(pred_fig, dpi=220)
    plt.close()
    print(f'[OK] saved {pred_fig}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--dataset_label', choices=['current', 'no_current'], default='current')
    parser.add_argument('--dt', type=float, default=0.2)
    parser.add_argument('--input_len', type=int, default=60)
    parser.add_argument('--pred_len', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--base_hidden_dim', type=int, default=128)
    parser.add_argument('--encdec_hidden_dim', type=int, default=128)
    parser.add_argument('--sample_idx', type=int, default=50)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    module = load_src()
    run_compare(module, args, Path(args.out_dir))


if __name__ == '__main__':
    main()
