from bias_current_integral_experiment import run
import argparse


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
    parser.add_argument('--beta_smooth', type=float, default=0.01)
    parser.add_argument('--sample_idx', type=int, default=50)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.cell = 'rnn'
    run(args)


if __name__ == '__main__':
    main()
