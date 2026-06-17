#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a current-only dataset using traj2-like and traj4-like route families."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from auv_trajectory_smoke_experiment import SimConfig, set_seed, wrap_angle


def ocean_current_extreme(x: float, y: float, t: float, strength: float = 0.72) -> tuple[float, float]:
    """Stronger multi-scale current field, matching the approved preview style."""
    bg_x = 0.42 * strength * math.sin(0.010 * y + 0.005 * t) + 0.22 * strength * math.cos(0.006 * x)
    bg_y = 0.38 * strength * math.cos(0.012 * x - 0.004 * t) - 0.18 * strength * math.sin(0.007 * y)

    def vortex(xc: float, yc: float, scale: float, radius: float, clockwise: bool) -> tuple[float, float]:
        dx, dy = x - xc, y - yc
        r2 = dx * dx + dy * dy
        swirl = math.exp(-r2 / (radius * radius))
        sign = -1.0 if clockwise else 1.0
        return sign * scale * strength * dy / 120.0 * swirl, -sign * scale * strength * dx / 120.0 * swirl

    v1x, v1y = vortex(290.0, 760.0, 0.92, 210.0, True)
    v2x, v2y = vortex(760.0, 320.0, 0.82, 180.0, False)
    v3x, v3y = vortex(610.0, 610.0, 0.56, 120.0, True)

    zone_x = 0.0
    zone_y = 0.0
    if x < 330.0:
        zone_x += 0.34 * strength
        zone_y += 0.14 * strength
    elif x > 720.0:
        zone_x -= 0.28 * strength
        zone_y += 0.24 * strength

    if y > 690.0:
        zone_y -= 0.34 * strength
    elif y < 260.0:
        zone_y += 0.28 * strength

    shear_x = 0.44 * strength * math.tanh((y - 520.0) / 85.0)
    shear_y = -0.36 * strength * math.tanh((x - 500.0) / 95.0)

    regime = math.floor(t / 4.5)
    jump_x = 0.46 * strength * math.sin(1.4 * regime + 0.35)
    jump_y = 0.42 * strength * math.cos(1.2 * regime - 0.55)

    corridor = math.exp(-((x - 540.0) ** 2) / (2 * 110.0 ** 2)) * math.exp(-((y - 470.0) ** 2) / (2 * 70.0 ** 2))
    anomaly_x = 0.56 * strength * corridor * math.sin(0.09 * t + 0.02 * x)
    anomaly_y = 0.40 * strength * corridor * math.cos(0.07 * t - 0.015 * y)

    pulse_phase = (t % 11.0) - 5.5
    burst = math.exp(-(pulse_phase * pulse_phase) / 2.6)
    burst_x = 0.52 * strength * math.sin(0.06 * x + 0.11 * t) * burst
    burst_y = 0.48 * strength * math.cos(0.05 * y - 0.08 * t) * burst

    cx = bg_x + v1x + v2x + v3x + zone_x + shear_x + jump_x + anomaly_x + burst_x
    cy = bg_y + v1y + v2y + v3y + zone_y + shear_y + jump_y + anomaly_y + burst_y
    return cx, cy


STYLE2_BASE = np.array([
    [120, 110],
    [190, 150],
    [250, 180],
    [210, 300],
    [150, 385],
    [235, 420],
    [325, 455],
    [455, 620],
    [380, 780],
    [560, 915],
], dtype=np.float32)

STYLE4_BASE = np.array([
    [125, 105],
    [100, 145],
    [205, 185],
    [275, 195],
    [235, 305],
    [315, 335],
    [355, 410],
    [490, 560],
    [640, 735],
    [520, 900],
], dtype=np.float32)


def build_waypoints(style: str, traj_id: int) -> np.ndarray:
    base = STYLE2_BASE if style == "traj2_like" else STYLE4_BASE
    rng = np.random.default_rng(5000 + traj_id)
    jitter = np.column_stack([
        rng.uniform(-45, 45, size=len(base)),
        rng.uniform(-35, 35, size=len(base)),
    ]).astype(np.float32)

    if style == "traj2_like":
        drift = np.column_stack([
            np.linspace(0, rng.uniform(80, 160), len(base)),
            np.linspace(0, rng.uniform(-10, 25), len(base)),
        ]).astype(np.float32)
    else:
        drift = np.column_stack([
            np.linspace(0, rng.uniform(110, 220), len(base)),
            np.linspace(0, rng.uniform(-25, 15), len(base)),
        ]).astype(np.float32)

    waypoints = np.clip(base + jitter + drift, [90, 90], [920, 950]).astype(np.float32)
    return waypoints


def simulate_one(cfg: SimConfig, traj_id: int, style: str) -> np.ndarray:
    dt = cfg.dt
    waypoints = build_waypoints(style, traj_id)
    x = float(waypoints[0, 0] + np.random.uniform(-8, 8))
    y = float(waypoints[0, 1] + np.random.uniform(-8, 8))
    theta = np.random.uniform(-0.5, 0.5)
    v = np.random.uniform(1.1, 1.9)
    a = 0.0
    nav_x = x + np.random.normal(0, cfg.meas_noise_pos)
    nav_y = y + np.random.normal(0, cfg.meas_noise_pos)
    wp_idx = 1
    prev_v = v
    rows = []

    for k in range(cfg.steps):
        t = k * dt
        target = waypoints[min(wp_idx, len(waypoints) - 1)]
        dx_t, dy_t = target[0] - x, target[1] - y
        dist = math.hypot(dx_t, dy_t)
        if dist < 16 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            target = waypoints[min(wp_idx, len(waypoints) - 1)]
            dx_t, dy_t = target[0] - x, target[1] - y

        desired_theta = math.atan2(dy_t, dx_t)
        heading_error = wrap_angle(desired_theta - theta)

        omega = np.clip(
            2.35 * heading_error
            + 0.22 * math.sin(0.073 * k + 0.31 * traj_id)
            + 0.09 * math.sin(0.191 * k + 0.7)
            + 0.06 * math.cos(0.113 * k + 0.8 * traj_id)
            + np.random.normal(0, 0.035),
            -1.05,
            1.05,
        )
        v_des = (
            1.75
            + 0.26 * math.sin(0.041 * k + 0.7 * traj_id)
            + 0.19 * math.sin(0.133 * k + 0.3)
            + 0.12 * math.cos(0.227 * k + 0.2)
        )
        a_cmd = np.clip(1.16 * (v_des - v), -0.82, 0.82)
        a = 0.68 * a + 0.32 * a_cmd + np.random.normal(0, 0.042)

        cx, cy = ocean_current_extreme(x, y, t, cfg.current_strength)
        v = np.clip(v + a * dt, 0.35, 3.0)
        theta = wrap_angle(theta + omega * dt + np.random.normal(0, 0.003))

        x = x + (v * math.cos(theta) + cx) * dt + np.random.normal(0, cfg.process_noise_pos)
        y = y + (v * math.sin(theta) + cy) * dt + np.random.normal(0, cfg.process_noise_pos)

        nav_x = nav_x + v * math.cos(theta) * dt + np.random.normal(0, 0.04)
        nav_y = nav_y + v * math.sin(theta) * dt + np.random.normal(0, 0.04)

        v_meas = v + np.random.normal(0, cfg.meas_noise_v)
        a_meas = (v - prev_v) / dt + np.random.normal(0, cfg.meas_noise_a)
        th_meas = wrap_angle(theta + np.random.normal(0, cfg.meas_noise_theta))
        prev_v = v
        rows.append([t, x, y, nav_x, nav_y, v_meas, a_meas, th_meas, cx, cy, traj_id])

    return np.array(rows, dtype=np.float32)


def generate_dataset(out_path: Path, n_traj: int, steps: int, dt: float, current_strength: float, seed: int) -> None:
    set_seed(seed)
    cfg = SimConfig(n_traj=n_traj, steps=steps, dt=dt, current_kind="mixed_hard", current_strength=current_strength)
    rows = []
    styles = []
    for traj_id in range(n_traj):
        style = "traj2_like" if traj_id % 2 == 0 else "traj4_like"
        styles.append(style)
        rows.append(simulate_one(cfg, traj_id, style))
    data = np.stack(rows, axis=0)
    np.savez_compressed(
        out_path,
        data=data,
        columns=np.array(["t", "true_x", "true_y", "nav_x", "nav_y", "v", "a", "theta", "cx", "cy", "traj_id"]),
        styles=np.array(styles),
    )
    print(f"[OK] saved {out_path}, shape={data.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_traj", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--current_strength", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    current_path = out_dir / "dataset_current.npz"
    generate_dataset(current_path, args.n_traj, args.steps, args.dt, args.current_strength, args.seed)


if __name__ == "__main__":
    main()
