#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUV trajectory prediction smoke experiment:
1) Generate simulated AUV datasets with/without ocean-current disturbance.
2) Compare EKF-like kinematic predictor, vanilla RNN, vanilla LSTM, and Physics-informed LSTM.

Run:
    python auv_trajectory_smoke_experiment.py --mode all --out_dir ./auv_exp --epochs 40 --n_traj 200

Outputs:
    ./auv_exp/dataset_no_current.npz
    ./auv_exp/dataset_current.npz
    ./auv_exp/results_no_current.csv
    ./auv_exp/results_current.csv
    ./auv_exp/fig_*.png

Dependencies:
    pip install numpy pandas matplotlib torch
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# 0. Reproducibility
# ----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# 1. Ocean-current models
# ----------------------------

def ocean_current(x: float, y: float, t: float, kind: str = "none", strength: float = 0.25) -> Tuple[float, float]:
    """
    Returns current velocity [cx, cy] in m/s.
    x,y are local coordinates in meters.
    """
    if kind == "none":
        return 0.0, 0.0

    if kind == "constant":
        return strength, 0.4 * strength

    if kind == "spatial":
        cx = strength * math.sin(0.025 * y)
        cy = strength * math.cos(0.025 * x)
        return cx, cy

    if kind == "vortex":
        xc, yc = 500.0, 500.0
        dx, dy = x - xc, y - yc
        r2 = dx * dx + dy * dy
        R = 350.0
        alpha = strength / 160.0
        decay = math.exp(-r2 / (R * R))
        cx = -alpha * dy * decay
        cy =  alpha * dx * decay
        return cx, cy

    if kind == "time_varying":
        base_x = strength * math.sin(0.02 * y + 0.01 * t)
        base_y = strength * math.cos(0.02 * x - 0.01 * t)
        return base_x, base_y

    if kind == "mixed":
        # Combination of spatial and vortex components.
        c1x, c1y = ocean_current(x, y, t, "spatial", strength * 0.55)
        c2x, c2y = ocean_current(x, y, t, "vortex", strength * 0.75)
        c3x = 0.08 * strength * math.sin(0.04 * t)
        c3y = 0.08 * strength * math.cos(0.03 * t)
        return c1x + c2x + c3x, c1y + c2y + c3y

    if kind == "mixed_hard":
        # Multi-scale field with regime-switch-like jumps and burst perturbations.
        c1x = 0.42 * strength * math.sin(0.014 * y + 0.004 * t)
        c1y = 0.38 * strength * math.cos(0.017 * x - 0.003 * t)

        dx1, dy1 = x - 320.0, y - 420.0
        r1 = dx1 * dx1 + dy1 * dy1
        swirl1 = math.exp(-r1 / (230.0 * 230.0))
        c2x = -0.55 * strength * dy1 / 120.0 * swirl1
        c2y =  0.55 * strength * dx1 / 120.0 * swirl1

        dx2, dy2 = x - 760.0, y - 720.0
        r2 = dx2 * dx2 + dy2 * dy2
        swirl2 = math.exp(-r2 / (180.0 * 180.0))
        c3x = 0.42 * strength * dy2 / 100.0 * swirl2
        c3y = -0.42 * strength * dx2 / 100.0 * swirl2

        shear_x = 0.28 * strength * math.tanh((y - 520.0) / 120.0)
        shear_y = -0.22 * strength * math.tanh((x - 470.0) / 150.0)

        regime = math.floor(t / 8.0)
        jump_x = 0.35 * strength * math.sin(1.7 * regime + 0.3)
        jump_y = 0.35 * strength * math.cos(1.3 * regime - 0.5)

        pulse_phase = (t % 18.0) - 9.0
        burst = math.exp(-(pulse_phase * pulse_phase) / 7.0)
        burst_x = 0.22 * strength * math.sin(0.06 * x + 0.09 * t) * burst
        burst_y = 0.22 * strength * math.cos(0.05 * y - 0.07 * t) * burst

        return c1x + c2x + c3x + shear_x + jump_x + burst_x, c1y + c2y + c3y + shear_y + jump_y + burst_y

    raise ValueError(f"Unknown current kind: {kind}")


# ----------------------------
# 2. AUV trajectory simulation
# ----------------------------

@dataclass
class SimConfig:
    n_traj: int = 200
    steps: int = 260
    dt: float = 0.2
    map_size: float = 1000.0
    current_kind: str = "mixed"
    current_strength: float = 0.30
    process_noise_pos: float = 0.02
    meas_noise_pos: float = 0.5
    meas_noise_v: float = 0.03
    meas_noise_a: float = 0.02
    meas_noise_theta: float = 0.01


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def simulate_one_trajectory(cfg: SimConfig, traj_id: int) -> np.ndarray:
    """
    Returns array shape [steps, 11]:
    [t, true_x, true_y, nav_x, nav_y, v, a, theta, cx, cy, traj_id]

    true_x,true_y: current-disturbed ground truth
    nav_x, nav_y: dead-reckoning/noisy navigation position used as model input
    """
    dt = cfg.dt

    # Random start and target waypoints.
    x = np.random.uniform(80, 180)
    y = np.random.uniform(80, 180)
    theta = np.random.uniform(-0.2, 0.5)
    v = np.random.uniform(1.2, 2.0)
    a = 0.0

    nav_x = x + np.random.normal(0, cfg.meas_noise_pos)
    nav_y = y + np.random.normal(0, cfg.meas_noise_pos)

    hard_mode = cfg.current_kind == "mixed_hard"

    # Curved reference path; use more aggressive waypoint changes for the hard-current setting.
    if hard_mode:
        x_centers = np.linspace(210, 930, 6)
        y_ranges = [(180, 320), (680, 860), (220, 380), (650, 840), (260, 430), (760, 930)]
        waypoints = np.array([
            [
                np.clip(xc + np.random.uniform(-45, 45), 120, 950),
                np.random.uniform(y_lo, y_hi),
            ]
            for xc, (y_lo, y_hi) in zip(x_centers, y_ranges)
        ], dtype=np.float32)
    else:
        waypoints = np.array([
            [np.random.uniform(250, 450), np.random.uniform(200, 450)],
            [np.random.uniform(450, 700), np.random.uniform(450, 700)],
            [np.random.uniform(800, 930), np.random.uniform(800, 930)],
        ], dtype=np.float32)
    wp_idx = 0

    rows = []
    prev_v = v

    for k in range(cfg.steps):
        t = k * dt
        target = waypoints[min(wp_idx, len(waypoints) - 1)]
        dx_t, dy_t = target[0] - x, target[1] - y
        dist = math.hypot(dx_t, dy_t)
        switch_radius = 18 if hard_mode else 25
        if dist < switch_radius and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            target = waypoints[wp_idx]
            dx_t, dy_t = target[0] - x, target[1] - y

        desired_theta = math.atan2(dy_t, dx_t)
        heading_error = wrap_angle(desired_theta - theta)

        # LOS/PID-like heading and speed control.
        if hard_mode:
            omega = np.clip(1.9 * heading_error + 0.12 * math.sin(0.11 * k + 0.5 * traj_id), -0.75, 0.75)
            v_des = 1.85 + 0.35 * math.sin(0.045 * k + 0.9 * traj_id) + 0.18 * math.sin(0.12 * k)
            a_cmd = np.clip(0.95 * (v_des - v), -0.55, 0.55)
            a = 0.78 * a + 0.22 * a_cmd + np.random.normal(0, 0.03)
        else:
            omega = np.clip(1.4 * heading_error, -0.45, 0.45)
            v_des = 1.7 + 0.25 * math.sin(0.03 * k + 0.8 * traj_id)
            a_cmd = np.clip(0.7 * (v_des - v), -0.35, 0.35)
            a = 0.85 * a + 0.15 * a_cmd + np.random.normal(0, 0.015)

        # Ocean current affects true position only.
        cx, cy = ocean_current(x, y, t, cfg.current_kind, cfg.current_strength)

        v = np.clip(v + a * dt, 0.4, 2.8)
        theta = wrap_angle(theta + omega * dt + np.random.normal(0, 0.002))

        x = x + (v * math.cos(theta) + cx) * dt + np.random.normal(0, cfg.process_noise_pos)
        y = y + (v * math.sin(theta) + cy) * dt + np.random.normal(0, cfg.process_noise_pos)

        # Dead-reckoning navigation ignores current; this creates drift.
        # This mimics low-cost underwater navigation without direct current observation.
        nav_x = nav_x + v * math.cos(theta) * dt + np.random.normal(0, 0.03)
        nav_y = nav_y + v * math.sin(theta) * dt + np.random.normal(0, 0.03)

        # Measured kinematics.
        v_meas = v + np.random.normal(0, cfg.meas_noise_v)
        a_meas = (v - prev_v) / dt + np.random.normal(0, cfg.meas_noise_a)
        th_meas = wrap_angle(theta + np.random.normal(0, cfg.meas_noise_theta))
        prev_v = v

        rows.append([t, x, y, nav_x, nav_y, v_meas, a_meas, th_meas, cx, cy, traj_id])

    return np.array(rows, dtype=np.float32)


def generate_dataset(cfg: SimConfig, out_path: Path) -> None:
    all_rows = [simulate_one_trajectory(cfg, i) for i in range(cfg.n_traj)]
    data = np.stack(all_rows, axis=0)  # [n_traj, steps, 11]
    np.savez_compressed(out_path, data=data, columns=np.array([
        "t", "true_x", "true_y", "nav_x", "nav_y", "v", "a", "theta", "cx", "cy", "traj_id"
    ]))
    print(f"[OK] saved {out_path}, shape={data.shape}")


# ----------------------------
# 3. Sliding-window dataset
# ----------------------------

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
        self.std = x.reshape(-1, x.shape[-1]).std(axis=0) + 1e-6
        return self

    def transform(self, x: np.ndarray):
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray):
        return x * self.std + self.mean


class AUVDataset(Dataset):
    def __init__(self, X_raw, Y_raw, x_scaler: Standardizer, y_scaler: Standardizer):
        self.X_raw = X_raw.astype(np.float32)
        self.Y_raw = Y_raw.astype(np.float32)
        self.X = x_scaler.transform(self.X_raw).astype(np.float32)
        self.Y = y_scaler.transform(self.Y_raw).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.Y[idx]),
            torch.from_numpy(self.X_raw[idx]),
            torch.from_numpy(self.Y_raw[idx]),
        )


def make_windows(data: np.ndarray, input_len: int = 10, pred_len: int = 10):
    """
    Input features: [nav_x, nav_y, v, a, theta]
    Target: true future positions [true_x, true_y]
    """
    Xs, Ys = [], []
    for traj in data:
        features = traj[:, [3, 4, 5, 6, 7]]
        target = traj[:, [1, 2]]
        max_i = len(traj) - input_len - pred_len
        for i in range(max_i):
            Xs.append(features[i:i + input_len])
            Ys.append(target[i + input_len:i + input_len + pred_len])
    return np.asarray(Xs, dtype=np.float32), np.asarray(Ys, dtype=np.float32)


def split_by_trajectory(data: np.ndarray, train_ratio=0.7, val_ratio=0.1):
    n = data.shape[0]
    idx = np.random.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return data[idx[:n_train]], data[idx[n_train:n_train+n_val]], data[idx[n_train+n_val:]]


# ----------------------------
# 4. Models
# ----------------------------

class RNNPredictor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, pred_len=10, cell: Literal["rnn", "lstm"] = "rnn"):
        super().__init__()
        self.pred_len = pred_len
        self.cell_type = cell
        if cell == "rnn":
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=1, batch_first=True, nonlinearity="tanh")
        elif cell == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        else:
            raise ValueError(cell)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * 2),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        y = self.head(last).view(x.size(0), self.pred_len, 2)
        return y


# ----------------------------
# 5. Loss and metrics物理损失项
# ----------------------------

def physics_loss_lstm(
    pred_norm: torch.Tensor,
    x_raw: torch.Tensor,
    y_scaler: Standardizer,
    dt: float,
    position_scale: float,
):
    """
    Finite-difference kinematic physics loss:
    Compare predicted displacement with body-motion expected displacement.

    x_raw last feature order: [nav_x, nav_y, v, a, theta].
    pred_norm: normalized predicted true positions [B,K,2].
    """
    device = pred_norm.device
    mean = torch.tensor(y_scaler.mean, dtype=torch.float32, device=device)
    std = torch.tensor(y_scaler.std, dtype=torch.float32, device=device)
    pred = pred_norm * std + mean  # [B,K,2], raw meters

    last = x_raw[:, -1, :].to(device)
    x0 = last[:, 0]
    y0 = last[:, 1]
    v0 = last[:, 2]
    a0 = last[:, 3]
    th = last[:, 4]

    prev_pos = torch.stack([x0, y0], dim=-1).unsqueeze(1)
    prev = torch.cat([prev_pos, pred[:, :-1, :]], dim=1)
    delta_pred = pred - prev

    K = pred.shape[1]
    steps = torch.arange(K, dtype=torch.float32, device=device)
    # velocity at the beginning of each future interval
    v_i = torch.clamp(v0.unsqueeze(1) + a0.unsqueeze(1) * steps.unsqueeze(0) * dt, min=0.0)
    dx_exp = (v_i * dt + 0.5 * a0.unsqueeze(1) * dt * dt) * torch.cos(th).unsqueeze(1)
    dy_exp = (v_i * dt + 0.5 * a0.unsqueeze(1) * dt * dt) * torch.sin(th).unsqueeze(1)
    delta_exp = torch.stack([dx_exp, dy_exp], dim=-1)

    return ((delta_pred - delta_exp) ** 2).mean() / (position_scale ** 2)


def metrics_np(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """
    pred,true: [N,K,2] in meters
    """
    err = pred - true
    dist = np.linalg.norm(err, axis=-1)
    return {
        "ADE": float(np.mean(dist)),
        "FDE": float(np.mean(dist[:, -1])),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MSE": float(np.mean(err ** 2)),
    }


def ekf_like_predict(X_raw: np.ndarray, pred_len: int, dt: float) -> np.ndarray:
    """
    EKF-like open-loop prediction from the last observed state.
    In a pure prediction horizon without future measurements, EKF forecast is equivalent
    to repeatedly applying the nonlinear motion model.
    """
    B = X_raw.shape[0]
    last = X_raw[:, -1, :]
    x = last[:, 0].copy()
    y = last[:, 1].copy()
    v = last[:, 2].copy()
    a = last[:, 3].copy()
    theta = last[:, 4].copy()

    preds = np.zeros((B, pred_len, 2), dtype=np.float32)
    for k in range(pred_len):
        v = np.clip(v + a * dt, 0.0, 5.0)
        x = x + (v * np.cos(theta) + 0.5 * a * np.cos(theta) * dt) * dt
        y = y + (v * np.sin(theta) + 0.5 * a * np.sin(theta) * dt) * dt
        preds[:, k, 0] = x
        preds[:, k, 1] = y
    return preds


# ----------------------------
# 6. Training and evaluation
# ----------------------------

def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_scaler: Standardizer,
    dt: float,
    lambda_phy: float = 0.0,
    epochs: int = 40,
    lr: float = 1e-3,
    device: str = "cpu",
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    position_scale = float(np.mean(y_scaler.std))

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb, xraw, yraw in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xraw = xraw.to(device)
            pred = model(xb)

            data_loss = torch.mean((pred - yb) ** 2)
            if lambda_phy > 0:
                phy = physics_loss_lstm(pred, xraw, y_scaler, dt=dt, position_scale=position_scale)
                loss = data_loss + lambda_phy * phy
            else:
                loss = data_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += loss.item() * len(xb)

        val_loss = evaluate_loss(model, val_loader, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"[{name}] epoch={ep:03d} train_loss={total/len(train_loader.dataset):.6f} val_mse={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: str) -> float:
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
def predict_model(model: nn.Module, loader: DataLoader, y_scaler: Standardizer, device: str):
    model.eval()
    preds = []
    trues = []
    raws = []
    for xb, yb, xraw, yraw in loader:
        xb = xb.to(device)
        pred_norm = model(xb).cpu().numpy()
        pred = y_scaler.inverse_transform(pred_norm)
        preds.append(pred)
        trues.append(yraw.numpy())
        raws.append(xraw.numpy())
    return np.concatenate(preds), np.concatenate(trues), np.concatenate(raws)


def run_experiment(npz_path: Path, out_dir: Path, label: str, args):
    print(f"\n=== Running dataset: {label} ===")
    loaded = np.load(npz_path)
    data = loaded["data"]

    train_traj, val_traj, test_traj = split_by_trajectory(data, 0.7, 0.1)
    X_train, Y_train = make_windows(train_traj, args.input_len, args.pred_len)
    X_val, Y_val = make_windows(val_traj, args.input_len, args.pred_len)
    X_test, Y_test = make_windows(test_traj, args.input_len, args.pred_len)

    x_scaler = Standardizer().fit(X_train)
    y_scaler = Standardizer().fit(Y_train)

    train_ds = AUVDataset(X_train, Y_train, x_scaler, y_scaler)
    val_ds = AUVDataset(X_val, Y_val, x_scaler, y_scaler)
    test_ds = AUVDataset(X_test, Y_test, x_scaler, y_scaler)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"device={device}, train_windows={len(train_ds)}, val_windows={len(val_ds)}, test_windows={len(test_ds)}")

    results = []

    # EKF-like baseline
    ekf_pred = ekf_like_predict(X_test, args.pred_len, args.dt)
    m = metrics_np(ekf_pred, Y_test)
    results.append({"model": "EKF_like_kinematic", **m})

    # Vanilla RNN
    rnn = RNNPredictor(input_dim=5, hidden_dim=args.hidden_dim, pred_len=args.pred_len, cell="rnn")
    rnn = train_model("RNN", rnn, train_loader, val_loader, y_scaler, args.dt, 0.0, args.epochs, args.lr, device)
    pred, true, raw = predict_model(rnn, test_loader, y_scaler, device)
    results.append({"model": "RNN", **metrics_np(pred, true)})

    # Physics-informed RNN
    pi_rnn = RNNPredictor(input_dim=5, hidden_dim=args.hidden_dim, pred_len=args.pred_len, cell="rnn")
    pi_rnn = train_model("PI_RNN", pi_rnn, train_loader, val_loader, y_scaler, args.dt, args.lambda_phy, args.epochs, args.lr, device)
    pred_pi_rnn, true_pi_rnn, raw_pi_rnn = predict_model(pi_rnn, test_loader, y_scaler, device)
    results.append({"model": f"PI_RNN_lambda_{args.lambda_phy}", **metrics_np(pred_pi_rnn, true_pi_rnn)})

    # Vanilla LSTM
    lstm = RNNPredictor(input_dim=5, hidden_dim=args.hidden_dim, pred_len=args.pred_len, cell="lstm")
    lstm = train_model("LSTM", lstm, train_loader, val_loader, y_scaler, args.dt, 0.0, args.epochs, args.lr, device)
    pred_lstm, true_lstm, raw_lstm = predict_model(lstm, test_loader, y_scaler, device)
    results.append({"model": "LSTM", **metrics_np(pred_lstm, true_lstm)})

    # Physics-informed LSTM
    pi_lstm = RNNPredictor(input_dim=5, hidden_dim=args.hidden_dim, pred_len=args.pred_len, cell="lstm")
    pi_lstm = train_model("PI_LSTM", pi_lstm, train_loader, val_loader, y_scaler, args.dt, args.lambda_phy, args.epochs, args.lr, device)
    pred_pi, true_pi, raw_pi = predict_model(pi_lstm, test_loader, y_scaler, device)
    results.append({"model": f"PI_LSTM_lambda_{args.lambda_phy}", **metrics_np(pred_pi, true_pi)})

    df = pd.DataFrame(results)
    out_csv = out_dir / f"results_{label}.csv"
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"[OK] saved {out_csv}")

    # Plot one sample
    sample_idx = min(50, len(Y_test)-1)
    plt.figure(figsize=(7, 6))
    hist = X_test[sample_idx, :, :2]
    gt = Y_test[sample_idx]
    ekf = ekf_pred[sample_idx]
    plt.plot(hist[:, 0], hist[:, 1], "ko-", label="input history(nav)", linewidth=1)
    plt.plot(gt[:, 0], gt[:, 1], "g-", label="ground truth", linewidth=2)
    plt.plot(ekf[:, 0], ekf[:, 1], "--", label="EKF-like")
    plt.plot(pred[sample_idx, :, 0], pred[sample_idx, :, 1], "--", label="RNN")
    plt.plot(pred_pi_rnn[sample_idx, :, 0], pred_pi_rnn[sample_idx, :, 1], "--", label="PI-RNN")
    plt.plot(pred_lstm[sample_idx, :, 0], pred_lstm[sample_idx, :, 1], "--", label="LSTM")
    plt.plot(pred_pi[sample_idx, :, 0], pred_pi[sample_idx, :, 1], "--", label="PI-LSTM")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(f"AUV trajectory prediction sample: {label}")
    fig_path = out_dir / f"fig_prediction_{label}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] saved {fig_path}")


# ----------------------------
# 7. CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "train", "all"], default="all")
    parser.add_argument("--out_dir", type=str, default="./auv_exp")
    parser.add_argument("--n_traj", type=int, default=200)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--input_len", type=int, default=10)
    parser.add_argument("--pred_len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_phy", type=float, default=0.2)
    parser.add_argument("--current_kind", type=str, default="mixed")
    parser.add_argument("--current_strength", type=float, default=0.35)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    no_current_path = out_dir / "dataset_no_current.npz"
    current_path = out_dir / "dataset_current.npz"

    if args.mode in ("generate", "all"):
        cfg0 = SimConfig(n_traj=args.n_traj, steps=args.steps, dt=args.dt, current_kind="none", current_strength=0.0)
        generate_dataset(cfg0, no_current_path)

        cfg1 = SimConfig(
            n_traj=args.n_traj,
            steps=args.steps,
            dt=args.dt,
            current_kind=args.current_kind,
            current_strength=args.current_strength,
        )
        generate_dataset(cfg1, current_path)

    if args.mode in ("train", "all"):
        if not no_current_path.exists() or not current_path.exists():
            raise FileNotFoundError("Dataset files not found. Run with --mode generate first.")
        run_experiment(no_current_path, out_dir, "no_current", args)
        run_experiment(current_path, out_dir, "current", args)


if __name__ == "__main__":
    main()
