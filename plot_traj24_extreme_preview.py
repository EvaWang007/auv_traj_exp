#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preview the traj2/traj4 extreme-current training dataset trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from generate_traj24_extreme_dataset import generate_dataset


def ensure_dataset_preview(tmp_path: Path, n_traj: int, steps: int, dt: float, current_strength: float, seed: int) -> Path:
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(tmp_path, n_traj=n_traj, steps=steps, dt=dt, current_strength=current_strength, seed=seed)
    return tmp_path


def plot_dataset(npz_path: Path, out_path: Path, max_traj: int = 12) -> None:
    loaded = np.load(npz_path, allow_pickle=True)
    data = loaded["data"]
    styles = loaded["styles"]

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab20", min(max_traj, len(data)))

    for i in range(min(max_traj, len(data))):
        traj = data[i]
        true_xy = traj[:, 1:3]
        nav_xy = traj[:, 3:5]
        style = styles[i]
        color = cmap(i)
        label_true = f"{style} true {i+1}"
        label_nav = f"{style} nav {i+1}"
        ax.plot(true_xy[:, 0], true_xy[:, 1], color=color, linewidth=2.0, label=label_true)
        ax.plot(nav_xy[:, 0], nav_xy[:, 1], color=color, linewidth=1.1, linestyle="--", alpha=0.55, label=label_nav)
        ax.scatter(true_xy[0, 0], true_xy[0, 1], color=color, s=18)
        ax.scatter(true_xy[-1, 0], true_xy[-1, 1], color=color, s=28, marker="X")

    ax.set_title("Traj2/Traj4 extreme-current training trajectory preview")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--n_traj", type=int, default=12)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--current_strength", type=float, default=0.72)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    preview_dir = repo_root / "previews"
    dataset_path = preview_dir / "traj24_extreme_preview_dataset.npz"
    figure_path = Path(args.out_path) if args.out_path else preview_dir / "fig_traj24_extreme_training_preview.png"

    ensure_dataset_preview(dataset_path, args.n_traj, args.steps, args.dt, args.current_strength, args.seed)
    plot_dataset(dataset_path, figure_path, max_traj=args.n_traj)
    print(f"[OK] saved {figure_path}")


if __name__ == "__main__":
    main()
