# AUV Trajectory Experiment Release

这个仓库包含一套可直接复现的 AUV 轨迹预测实验代码与 `hardcurrent` 数据集，主要用于比较：

- `RNN / PI-RNN`
- `LSTM / PI-LSTM`
- `EncDec_LSTM / PI_EncDec_LSTM`
- `PSO-LSTM`
- `EKF_like_kinematic`

## 目录结构

```text
.
├── auv_trajectory_smoke_experiment.py
├── pso_lstm_experiment.py
├── encdec_lstm_experiment.py
├── encdec_lstm_physics_experiment.py
├── run_encdec_only.py
├── run_pi_rnn_only.py
└── auv_exp_lstm_hardcurrent/
    ├── dataset_current.npz
    └── dataset_no_current.npz
```

## 环境准备

建议使用 Python 3.10。

### 方式 1：使用 conda 一键复现

```bash
conda env create -f environment.yml
conda activate auv_traj_exp
```

### 方式 2：手动创建环境 + 固定版本 requirements

```bash
conda create -n auv_traj_exp python=3.10 -y
conda activate auv_traj_exp
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果服务器有 NVIDIA GPU，建议根据 CUDA 版本将 `torch==2.12.0` 替换为对应的 GPU 版 PyTorch 安装方式。

## 数据集

已经包含：

- `auv_exp_lstm_hardcurrent/dataset_current.npz`
- `auv_exp_lstm_hardcurrent/dataset_no_current.npz`

## 脚本与模型对应关系

### `auv_trajectory_smoke_experiment.py`
会训练并评估以下模型：

- `EKF_like_kinematic`
- `RNN`
- `PI_RNN`
- `LSTM`
- `PI_LSTM`

### `run_pi_rnn_only.py`
只训练并评估：

- `PI_RNN`

### `run_encdec_only.py`
只训练并评估：

- `EncDec_LSTM`
- `PI_EncDec_LSTM`

### `encdec_lstm_experiment.py`
用于比较：

- `EKF_like_kinematic`
- `LSTM`
- `EncDec_LSTM`

### `encdec_lstm_physics_experiment.py`
用于比较：

- `EKF_like_kinematic`
- `LSTM`
- `PI_LSTM`
- `EncDec_LSTM`
- `PI_EncDec_LSTM`

### `pso_lstm_experiment.py`
用于执行：

- `PSO-LSTM`

其中：

- `PSO` 搜索超参数
- `Adam` 训练网络权重
- 搜索阶段使用验证集 `RMSE` 作为 fitness
- 找到最优超参数后，再重新训练最终 `LSTM`

## 训练命令

### 1. 原始主实验脚本（会跑 EKF/RNN/PI-RNN/LSTM/PI-LSTM）

```bash
python auv_trajectory_smoke_experiment.py \
  --mode train \
  --out_dir ./auv_exp_lstm_hardcurrent \
  --epochs 50 \
  --input_len 60 \
  --pred_len 30 \
  --batch_size 64 \
  --hidden_dim 128 \
  --lambda_phy 0.05
```

### 2. 只跑 EncDec_LSTM 和 PI_EncDec_LSTM

```bash
python run_encdec_only.py \
  --out_dir ./auv_exp_lstm_hardcurrent \
  --epochs 50 \
  --input_len 60 \
  --pred_len 30 \
  --batch_size 64 \
  --hidden_dim 128 \
  --lambda_phy 0.05 \
  --teacher_forcing_ratio 0.5
```

### 3. PSO-LSTM（先搜索超参数，再正式训练）

```bash
python pso_lstm_experiment.py \
  --out_dir ./auv_exp_lstm_hardcurrent \
  --dataset_label current \
  --input_len 60 \
  --pred_len 30 \
  --population 6 \
  --iterations 6 \
  --search_epochs 8 \
  --final_epochs 50 \
  --search_train_limit 20000 \
  --search_val_limit 4000
```

## 常见输出文件

### 主实验脚本输出
- `results_current.csv`
- `results_no_current.csv`
- `fig_prediction_current.png`
- `fig_prediction_no_current.png`

### EncDec-only 输出
- `results_encdec_only_current.csv`
- `results_encdec_only_no_current.csv`
- `fig_prediction_encdec_only_current.png`
- `fig_prediction_encdec_only_no_current.png`

### PI-RNN-only 输出
- `results_pi_rnn_only_current.csv`
- `results_pi_rnn_only_no_current.csv`
- `fig_prediction_pi_rnn_only_current.png`
- `fig_prediction_pi_rnn_only_no_current.png`

### PSO-LSTM 输出
- `best_hparams_pso_lstm_current.json`
- `pso_search_history_current.csv`
- `results_pso_lstm_current.csv`
- `fig_pso_convergence_current.png`
- `fig_prediction_pso_lstm_current.png`

## 说明

- 所有辅助脚本已经改成相对路径，不再依赖 `/home/evawang/...`。
- 默认 `out_dir` 指向当前仓库内的 `auv_exp_lstm_hardcurrent/`。
- 结果文件会生成在 `out_dir` 目录下。
