import argparse
import torch as tc
import utils

from bptt import bptt_algorithm

from bptt.rnn_model import RNN

tc.set_num_threads(1)


def get_parser():
    parser = argparse.ArgumentParser(description="Estimate Dynamical System")  # 创建一个命令行解析器对象
    # parser.add_argument('--experiment', type=str, default='GradientTests')
    parser.add_argument('--experiment', type=str, default='BVPTests')
    # parser.add_argument('--name', type=str, default='Lorenz')
    parser.add_argument('--name', type=str, default='bvp_ode')
    parser.add_argument('--run', type=int, default=None)
    # general settings
    parser.add_argument('--no_printing', type=bool, default=True)
    parser.add_argument('--use_tb', type=bool, default=True)
    parser.add_argument('--metrics', type=list, default=['klx'])
    # dataset
    # parser.add_argument('--data_path', type=str, default='datasets/Lorenz/lorenz_data_chaos.npy')
    # parser.add_argument('--data_path', type=str, default='datasets/bvp_data/T1/bvp_1_T1.npy')
    parser.add_argument('--data_path', type=str, default='datasets/bvp_data/UBFC_bvp_all_1_train.npy')

    # parser.add_argument('--data_path', type=str, default='datasets/two_cycle.npy')

    parser.add_argument('--inputs_path', type=str, default=None)
    parser.add_argument('--load_model_path', type=str, default=None)
    # model
    parser.add_argument('--dim_z', type=int, default=64)  # latent model dimensionality ppg-64 lorenz-30
    parser.add_argument('--n_bases', '-nb', type=int, default=1)
    parser.add_argument('--clip_range', '-clip', type=float, default=None)

    parser.add_argument('--time_step', type=int, default=5)
    # choices: LSTM/RNN/ODELSTM/node
    parser.add_argument('--model', '-m', type=str, default='LSTMODE')  # choose from  LSTM, RNN, LSTMODE, NODE

    parser.add_argument('--layer_norm', '-ln', type=int, default=0)  # 1:True , 0 False

    # Training parameter
    parser.add_argument('--windowing', '-win', type=int, default=0)  # 1:True , 0 False
    parser.add_argument('--random', '-rand', type=int, default=0)  # 1:True , 0 False
    parser.add_argument('--deltaTau', '-dT', type=int, default=10)  # 1:True , 0 False

    parser.add_argument('--n_interleave', '-ni', type=int, default=12)  # forcing/learning interval, called tau in the paper ppg-12 lorenz-30
    parser.add_argument('--batch_size', '-bs', type=int, default=32)  # 96  #32
    # parser.add_argument('--batches_per_epoch', '-bpi', type=int, default=300)  # 10
    parser.add_argument('--batches_per_epoch', '-bpi', type=int, default=10)  # 10
    parser.add_argument('--seq_len', '-sl', type=int, default=32)  # choose depending on sweeping range of n_interleave ppg-32 lorenz-100
    parser.add_argument('--fix_obs_model', '-fo', type=bool, default=False)

    parser.add_argument('--gradient_clipping', '-gc', type=float, default=None)

    # optimization
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', '-n', type=int, default=600)

    parser.add_argument('--use_reg', '-r', type=bool, default=False)
    parser.add_argument('--reg_ratios', '-rr', nargs='*', type=float, default=[.3])
    parser.add_argument('--reg_alphas', '-ra', nargs='*', type=float, default=[.5])
    parser.add_argument('--reg_norm', '-rn', type=str, choices=['l2', 'l1'], default='l2')
    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def train(args):
    writer, save_path = utils.init_writer(args)
    args, data_set = utils.load_dataset(args)

    utils.check_args(args)
    utils.save_args(args, save_path, writer)

    training_algorithm = bptt_algorithm.BPTT(args, data_set, writer, save_path)
    training_algorithm.train()
    return save_path


def attractor_extract(args):
    train(args)


attractor_extract(get_args())
# python attractor_extract.py -bpi=10 -sl=32 -lr=0.0001  -n=300 -ni=12 -ln=1 --dim_z=64