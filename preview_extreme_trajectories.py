#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preview a more complex trajectory/current scenario before full integration."""

from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from auv_trajectory_smoke_experiment import SimConfig, set_seed, wrap_angle


def ocean_current_extreme(x: float, y: float, t: float, strength: float = 0.72) -> tuple[float, float]:
    """More complex current field than mixed_hard, with zone switches and bursts."""
    # Large-scale background flow.
    bg_x = 0.42 * strength * math.sin(0.010 * y + 0.005 * t) + 0.22 * strength * math.cos(0.006 * x)
    bg_y = 0.38 * strength * math.cos(0.012 * x - 0.004 * t) - 0.18 * strength * math.sin(0.007 * y)

    # Two vortices plus one smaller local eddy.
    def vortex(xc: float, yc: float, scale: float, radius: float, clockwise: bool) -> tuple[float, float]:
        dx, dy = x - xc, y - yc
        r2 = dx * dx + dy * dy
        swirl = math.exp(-r2 / (radius * radius))
        sign = -1.0 if clockwise else 1.0
        return sign * scale * strength * dy / 120.0 * swirl, -sign * scale * strength * dx / 120.0 * swirl

    v1x, v1y = vortex(290.0, 760.0, 0.92, 210.0, True)
    v2x, v2y = vortex(760.0, 320.0, 0.82, 180.0, False)
    v3x, v3y = vortex(610.0, 610.0, 0.56, 120.0, True)

    # Spatial zone switches: different regions bias flow differently.
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

    # Shear layers.
    shear_x = 0.44 * strength * math.tanh((y - 520.0) / 85.0)
    shear_y = -0.36 * strength * math.tanh((x - 500.0) / 95.0)

    # Frequent regime switches.
    regime = math.floor(t / 4.5)
    jump_x = 0.46 * strength * math.sin(1.4 * regime + 0.35)
    jump_y = 0.42 * strength * math.cos(1.2 * regime - 0.55)

    # Local anomaly corridor.
    corridor = math.exp(-((x - 540.0) ** 2) / (2 * 110.0 ** 2)) * math.exp(-((y - 470.0) ** 2) / (2 * 70.0 ** 2))
    anomaly_x = 0.56 * strength * corridor * math.sin(0.09 * t + 0.02 * x)
    anomaly_y = 0.40 * strength * corridor * math.cos(0.07 * t - 0.015 * y)

    # Short bursts.
    pulse_phase = (t % 11.0) - 5.5
    burst = math.exp(-(pulse_phase * pulse_phase) / 2.6)
    burst_x = 0.52 * strength * math.sin(0.06 * x + 0.11 * t) * burst
    burst_y = 0.48 * strength * math.cos(0.05 * y - 0.08 * t) * burst

    cx = bg_x + v1x + v2x + v3x + zone_x + shear_x + jump_x + anomaly_x + burst_x
    cy = bg_y + v1y + v2y + v3y + zone_y + shear_y + jump_y + anomaly_y + burst_y
    return cx, cy


def simulate_extreme_preview(cfg: SimConfig, traj_id: int) -> np.ndarray:
    """Preview-only simulator with longer serpentine motion and stronger turns."""
    dt = cfg.dt
    x = np.random.uniform(90, 170)
    y = np.random.uniform(90, 190)
    theta = np.random.uniform(-0.4, 0.6)
    v = np.random.uniform(1.15, 1.95)
    a = 0.0
    nav_x = x + np.random.normal(0, cfg.meas_noise_pos)
    nav_y = y + np.random.normal(0, cfg.meas_noise_pos)

    # Spread the route across more flow regions while keeping an overall upward mission.
    base_points = np.array([
        [120, 140],
        [300, 220],
        [180, 360],
        [430, 470],
        [260, 610],
        [560, 720],
        [380, 820],
        [700, 900],
        [520, 945],
    ], dtype=np.float32)
    # Per-trajectory irregular perturbations so turns are less rhythmic and less symmetric.
    rng = np.random.default_rng(1000 + traj_id)
    jitter = np.column_stack([
        rng.uniform(-70, 70, size=len(base_points)),
        rng.uniform(-45, 45, size=len(base_points)),
    ]).astype(np.float32)
    drift = np.column_stack([
        np.linspace(0, rng.uniform(80, 180), len(base_points)),
        np.linspace(0, rng.uniform(-20, 30), len(base_points)),
    ]).astype(np.float32)
    waypoints = np.clip(base_points + jitter + drift, [90, 90], [920, 950]).astype(np.float32)
    wp_idx = 0
    rows = []
    prev_v = v

    for k in range(cfg.steps):
        t = k * dt
        target = waypoints[min(wp_idx, len(waypoints) - 1)]
        dx_t, dy_t = target[0] - x, target[1] - y
        dist = math.hypot(dx_t, dy_t)
        if dist < 14 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            target = waypoints[wp_idx]
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


def build_preview_figure(out_path: Path) -> None:
    set_seed(42)
    cfg = SimConfig(n_traj=4, steps=1500, dt=0.2, current_kind="mixed_hard", current_strength=0.72)
    trajectories = [simulate_extreme_preview(cfg, i) for i in range(cfg.n_traj)]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background current field at a representative time.
    gx = np.linspace(80, 940, 18)
    gy = np.linspace(80, 940, 18)
    X, Y = np.meshgrid(gx, gy)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            cx, cy = ocean_current_extreme(float(X[i, j]), float(Y[i, j]), 32.0, cfg.current_strength)
            U[i, j] = cx
            V[i, j] = cy
    speed = np.hypot(U, V)
    ax.contourf(X, Y, speed, levels=16, cmap="Blues", alpha=0.35)
    ax.quiver(X, Y, U, V, color="#355C7D", alpha=0.55, scale=10)

    colors = ["#D7263D", "#1B998B", "#2E294E", "#F46036"]
    for idx, traj in enumerate(trajectories):
        true_xy = traj[:, 1:3]
        nav_xy = traj[:, 3:5]
        ax.plot(true_xy[:, 0], true_xy[:, 1], color=colors[idx], linewidth=2.2, label=f"true traj {idx + 1}")
        ax.plot(nav_xy[:, 0], nav_xy[:, 1], color=colors[idx], linewidth=1.1, alpha=0.55, linestyle="--", label=f"nav drift {idx + 1}")
        ax.scatter(true_xy[0, 0], true_xy[0, 1], color=colors[idx], s=24, marker="o")
        ax.scatter(true_xy[-1, 0], true_xy[-1, 1], color=colors[idx], s=36, marker="X")

    ax.set_title("Preview: proposed mixed_extreme trajectories and drifted navigation")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(40, 980)
    ax.set_ylim(40, 980)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    out_path = repo_root / "previews" / "fig_trajectory_preview_mixed_extreme.png"
    build_preview_figure(out_path)
    print(f"[OK] saved preview to {out_path}")


if __name__ == "__main__":
    main()
