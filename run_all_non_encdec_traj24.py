#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate/train/compare all non-EncDec models on the traj2+traj4 extreme-current dataset."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_generate(out_dir: Path, args) -> None:
    gen = load_module("traj24_gen", REPO_ROOT / "generate_traj24_extreme_dataset.py")
    gen.generate_dataset(
        out_dir / "dataset_current.npz",
        n_traj=args.n_traj,
        steps=args.steps,
        dt=args.dt,
        current_strength=args.current_strength,
        seed=args.seed,
    )


def run_train_all(out_dir: Path, args) -> None:
    smoke = load_module("traj24_smoke", REPO_ROOT / "auv_trajectory_smoke_experiment.py")
    pso = load_module("traj24_pso", REPO_ROOT / "pso_lstm_experiment.py")
    pso_pi = load_module("traj24_pso_pi", REPO_ROOT / "pso_pi_lstm_experiment.py")
    compare = load_module("traj24_compare", REPO_ROOT / "compare_all_models.py")

    current_path = out_dir / "dataset_current.npz"
    if not current_path.exists():
        raise FileNotFoundError(f"{current_path} not found. Run with --mode generate first.")

    common_seed = args.seed

    # Base non-EncDec models: EKF, RNN, PI-RNN, LSTM, PI-LSTM
    smoke.set_seed(common_seed)
    smoke_args = SimpleNamespace(
        input_len=args.input_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dt=args.dt,
        lambda_phy=args.lambda_phy,
        epochs=args.epochs,
        lr=args.lr,
        cpu=args.cpu,
    )
    smoke.run_experiment(current_path, out_dir, "current", smoke_args)

    # PSO-LSTM
    smoke.set_seed(common_seed)
    pso_args = SimpleNamespace(
        dataset_label="current",
        input_len=args.input_len,
        pred_len=args.pred_len,
        dt=args.dt,
        population=args.population,
        iterations=args.iterations,
        search_epochs=args.search_epochs,
        final_epochs=args.final_epochs,
        search_train_limit=args.search_train_limit,
        search_val_limit=args.search_val_limit,
        inertia_max=args.inertia_max,
        inertia_min=args.inertia_min,
        c1=args.c1,
        c2=args.c2,
        cpu=args.cpu,
        seed=common_seed,
    )
    pso.run_pso(smoke, pso_args, out_dir)

    # PSO-PI-LSTM
    smoke.set_seed(common_seed)
    pso_pi_args = SimpleNamespace(
        dataset_label="current",
        input_len=args.input_len,
        pred_len=args.pred_len,
        dt=args.dt,
        population=args.population,
        iterations=args.iterations,
        search_epochs=args.search_epochs,
        final_epochs=args.final_epochs,
        search_train_limit=args.search_train_limit,
        search_val_limit=args.search_val_limit,
        inertia_max=args.inertia_max,
        inertia_min=args.inertia_min,
        c1=args.c1,
        c2=args.c2,
        cpu=args.cpu,
        seed=common_seed,
    )
    pso_pi.run_pso_pi(smoke, pso_pi_args, out_dir)

    # Unified comparison (non-EncDec models will be auto-detected).
    compare_args = SimpleNamespace(
        dataset_label="current",
        dt=args.dt,
        input_len=args.input_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        base_hidden_dim=args.hidden_dim,
        encdec_hidden_dim=args.hidden_dim,
        sample_idx=args.sample_idx,
        cpu=args.cpu,
        seed=common_seed,
    )
    compare.run_compare(smoke, compare_args, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "train", "all"], default="all")
    parser.add_argument("--out_dir", type=str, required=True)

    # Dataset generation
    parser.add_argument("--n_traj", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--current_strength", type=float, default=0.72)

    # Base training
    parser.add_argument("--input_len", type=int, default=80)
    parser.add_argument("--pred_len", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_phy", type=float, default=0.05)

    # PSO / PSO-PI
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--search_epochs", type=int, default=8)
    parser.add_argument("--final_epochs", type=int, default=60)
    parser.add_argument("--search_train_limit", type=int, default=20000)
    parser.add_argument("--search_val_limit", type=int, default=4000)
    parser.add_argument("--inertia_max", type=float, default=0.9)
    parser.add_argument("--inertia_min", type=float, default=0.4)
    parser.add_argument("--c1", type=float, default=1.5)
    parser.add_argument("--c2", type=float, default=1.5)

    parser.add_argument("--sample_idx", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("generate", "all"):
        run_generate(out_dir, args)
    if args.mode in ("train", "all"):
        run_train_all(out_dir, args)


if __name__ == "__main__":
    main()
