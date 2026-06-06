import argparse
import os
import random
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')

PAPER_BACKBONES = [
    # Full plug-in (Section 5.2, Table 2)
    'anomalytransformer', 'memto', 'sub_adjacent_transformer',
    # Loss-only plug-in (Section 5.2, Table 1)
    'dagmm', 'dtaad', 'lstm_autoencoder', 'npsr', 'tranad',
]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_args():
    parser = argparse.ArgumentParser(
        description='Physics-Informed Plug-ins for Satellite Orbit Anomaly Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Basic settings ───────────────────────────────────────────────────
    parser.add_argument('--task_name', type=str, default='anomaly_detection')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='kompsat_ad')
    parser.add_argument(
        '--model', type=str, default='anomalytransformer',
        choices=PAPER_BACKBONES,
        help='AD backbone to use')

    # ── Data settings ────────────────────────────────────────────────────
    parser.add_argument('--data', type=str, default='KAFASATAnomalyDataset')
    parser.add_argument('--root_path', type=str,
                        default='./data/KOMPSAT-3A_multiclass',
                        help='root directory containing the dataset CSV files')
    parser.add_argument('--data_path', type=str,
                        default='kompsat3a_train_multiclass.csv')
    parser.add_argument('--train_data_path', type=str,
                        default='kompsat3a_train_multiclass.csv')
    parser.add_argument('--test_data_path', type=str,
                        default='kompsat3a_test_multiclass.csv')
    parser.add_argument('--val_data_path', type=str, default=None)

    # ── Sequence settings ────────────────────────────────────────────────
    parser.add_argument('--seq_len', type=int, default=48,
                        help='window length L (paper default 48)')
    parser.add_argument('--input_token_len', type=int, default=48)
    parser.add_argument('--output_token_len', type=int, default=48)
    parser.add_argument('--test_seq_len', type=int, default=48)
    parser.add_argument('--test_pred_len', type=int, default=48)
    parser.add_argument('--stride', type=int, default=None,
                        help='sliding-window stride '
                             '(None = auto: seq_len for train, 1 for test)')

    # ── Generic architecture ─────────────────────────────────────────────
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', type=bool, default=True)

    # ── Loss selection (paper contribution) ──────────────────────────────
    parser.add_argument(
        '--loss', type=str, default='MSE',
        choices=['MSE', 'Physics', 'SATLoss'],
        help='"MSE": standard reconstruction. '
             '"Physics": Physics-informed Loss (Section 4.3). '
             '"SATLoss": original SAT objective (only when --model sub_adjacent_transformer)')

    # Physics-informed Loss weights (Section 4.3, Equation 10)
    parser.add_argument('--physics_lambda_recon',   type=float, default=1.0,
                        help='reconstruction weight lambda_r')
    parser.add_argument('--physics_lambda_smooth',  type=float, default=0.1,
                        help='temporal smoothness weight lambda_s')
    parser.add_argument('--physics_lambda_angular', type=float, default=0.1,
                        help='angular momentum weight lambda_a')
    parser.add_argument('--physics_lambda_bound',   type=float, default=0.01,
                        help='physical bounds weight (auxiliary)')

    # ── OPE plug-in (paper contribution, Section 4.2) ────────────────────
    parser.add_argument('--use_ope',         type=int,   default=0,
                        help='enable Adaptive Orbital Period Embedding (1 = on)')
    parser.add_argument('--ope_auto_period', type=int,   default=1,
                        help='estimate T_orb from data via Kepler third law')
    parser.add_argument('--ope_period',      type=float, default=97.0,
                        help='fallback T_orb (minutes) if --ope_auto_period 0')
    parser.add_argument('--ope_n_harmonics', type=int,   default=4,
                        help='number of harmonics K (paper default 4)')

    # ── Training ─────────────────────────────────────────────────────────
    parser.add_argument('--train_epochs',   type=int,   default=50)
    parser.add_argument('--batch_size',     type=int,   default=60)
    parser.add_argument('--learning_rate',  type=float, default=1e-3)
    parser.add_argument('--weight_decay',   type=float, default=0.0)
    parser.add_argument('--patience',       type=int,   default=10)
    parser.add_argument('--use_early_stop', type=bool,  default=True)
    parser.add_argument('--lradj',          type=str,   default='type1')
    parser.add_argument('--no_scheduler',   action='store_true',
                        help='Disable the learning-rate scheduler.')
    parser.add_argument('--cosine',         action='store_true',
                        help='Use cosine annealing scheduler (SAT only).')
    parser.add_argument('--tmax',           type=int,   default=50,
                        help='T_max for cosine annealing (defaults to train_epochs).')
    parser.add_argument('--samples_per_file', type=int, default=None,
                        help='Optional cap on samples per file (None = use all).')

    # ── Dataset loader options ───────────────────────────────────────────
    parser.add_argument('--nonautoregressive', action='store_true',
                        help='Use non-autoregressive sample construction.')
    parser.add_argument('--test_flag',        type=str,   default='T',
                        help='Test split flag used by the dataset loader.')
    parser.add_argument('--subset_rand_ratio', type=float, default=1.0,
                        help='Random subset ratio of training data (1.0 = use all).')

    # ── Hardware ─────────────────────────────────────────────────────────
    parser.add_argument('--use_gpu',       type=bool, default=True)
    parser.add_argument('--gpu',           type=int,  default=0)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--devices',       type=str,  default='0,1,2,3')
    parser.add_argument('--num_workers',   type=int,  default=4)

    # ── Reproducibility & output ─────────────────────────────────────────
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--checkpoints',    type=str, default='./checkpoints/')
    parser.add_argument('--target_feature', type=int, default=None,
                        help='None = multivariate (all 6 elements)')

    # ── Backbone-specific hyperparameters ────────────────────────────────
    # TranAD
    parser.add_argument('--tranad_k',             type=float, default=3.0)
    parser.add_argument('--tranad_phase_switch',  type=int,   default=25)
    # DAGMM
    parser.add_argument('--n_gmm',          type=int,   default=6)
    parser.add_argument('--hidden_dim1',    type=int,   default=128)
    parser.add_argument('--hidden_dim2',    type=int,   default=64)
    parser.add_argument('--hidden_dim3',    type=int,   default=32)
    parser.add_argument('--lambda_energy',  type=float, default=0.1)
    parser.add_argument('--lambda_cov',     type=float, default=0.005)
    # Anomaly Transformer
    parser.add_argument('--lambda_association', type=float, default=1.0)
    # DTAAD
    parser.add_argument('--dtaad_window_size', type=int,   default=10)
    parser.add_argument('--dtaad_d_model',     type=int,   default=None)
    parser.add_argument('--dtaad_n_heads',     type=int,   default=None)
    parser.add_argument('--dtaad_d_ff',        type=int,   default=16)
    parser.add_argument('--dtaad_lambda',      type=float, default=0.8)
    # MEMTO
    parser.add_argument('--memto_n_memory',     type=int,   default=100)
    parser.add_argument('--memto_shrink_thres', type=float, default=0.0025)
    parser.add_argument('--memto_d_model',      type=int,   default=512)
    parser.add_argument('--memto_n_heads',      type=int,   default=8)
    parser.add_argument('--memto_e_layers',     type=int,   default=3)
    parser.add_argument('--memto_d_ff',         type=int,   default=512)
    parser.add_argument('--memto_lambda_entropy', type=float, default=0.01)
    parser.add_argument('--memto_temperature',  type=float, default=0.1)
    parser.add_argument('--memto_phase_type',   type=str,   default='train',
                        choices=['train', 'second_train', 'test'])
    # NPSR
    parser.add_argument('--npsr_d_model',  type=int, default=256)
    parser.add_argument('--npsr_n_heads',  type=int, default=8)
    parser.add_argument('--npsr_e_layers', type=int, default=3)
    parser.add_argument('--npsr_type',     type=str, default='autoencoder',
                        choices=['autoencoder', 'squeezing'])
    # Sub-Adjacent Transformer
    parser.add_argument('--sat_linear_attn',   type=int,   default=1)
    parser.add_argument('--sat_mapping_fun',   type=str,   default='ours')
    parser.add_argument('--sat_span',          type=int,   nargs=2, default=[10, 20])
    parser.add_argument('--sat_one_side',      type=int,   default=1)
    parser.add_argument('--sat_k',             type=float, default=3.0)
    parser.add_argument('--sat_temperature',   type=float, default=50.0)
    parser.add_argument('--sat_softmax_span',  type=int,   default=None)

    return parser

def build_setting_tag(args):
    s = f'{args.model_id}_{args.model}_{args.seq_len}_{args.learning_rate}_{args.train_epochs}'

    # Backbone-specific suffix
    m = args.model
    if m == 'lstm_autoencoder':
        s += f'_dmodel{args.d_model}'
    elif m == 'tranad':
        s += f'_dmodel{args.d_model}'
    elif m == 'dagmm':
        s += f'_h{args.hidden_dim1}_{args.hidden_dim2}_{args.hidden_dim3}'
    elif m == 'anomalytransformer':
        s += f'_dmodel{args.d_model}_dff{args.d_ff}'
    elif m == 'dtaad':
        s += f'_dff{args.dtaad_d_ff}'
    elif m == 'memto':
        s += f'_dmodel{args.memto_d_model}_dff{args.memto_d_ff}'
    elif m == 'npsr':
        s += f'_dmodel{args.npsr_d_model}'
    elif m == 'sub_adjacent_transformer':
        s += f'_dmodel{args.d_model}_el{args.e_layers}_span{args.sat_span[0]}_{args.sat_span[1]}'

    if args.target_feature is not None:
        s += f'_feat{args.target_feature}'
    else:
        s += '_multivariate'

    # Loss tag — distinguishes MSE / Physics / SATLoss checkpoints
    s += f'_loss{args.loss}'

    # OPE tag — distinguishes ope0 / ope1 checkpoints
    from utils.ope_patch import ope_setting_tag
    s += f'_{ope_setting_tag(args)}'

    # Seed tag — separates seed runs
    s += f'_seed{args.seed}'
    return s


def main():
    args = create_args().parse_args()
    set_seed(args.seed)

    # SAT uses a dedicated experiment class
    if args.model == 'sub_adjacent_transformer':
        from exp.exp_ad_sat import Exp_AD_SAT
        Exp = Exp_AD_SAT
    else:
        from exp.exp_ad import Exp_AD
        Exp = Exp_AD

    # Banner
    print('=' * 60)
    print('Physics-Informed Plug-ins for Satellite Orbit AD')
    print('=' * 60)
    print(f'Backbone        : {args.model}')
    print(f'Loss            : {args.loss}')
    print(f'OPE plug-in     : {"on" if args.use_ope else "off"}')
    print(f'Seed            : {args.seed}')
    print(f'Window length L : {args.seq_len}')
    if args.target_feature is None:
        print('Mode            : Multivariate (six orbital elements)')
    else:
        print(f'Mode            : Univariate (feature {args.target_feature})')
    print('=' * 60)

    setting = build_setting_tag(args)
    exp = Exp(args)

    # Apply OPE plug-in if enabled
    from utils.ope_patch import maybe_patch_with_ope
    maybe_patch_with_ope(exp.model, args)

    # Parameter count
    total     = sum(p.numel() for p in exp.model.parameters())
    trainable = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
    print(f'\nTotal parameters     : {total:,}')
    print(f'Trainable parameters : {trainable:,}\n')

    # Train + test
    if args.is_training:
        print(f'Setting: {setting}\n')
        print('Training...')
        exp.train(setting)
        print('Training completed.\n')

        print('Testing...')
        exp.test(setting)
        print('Testing completed.')
    else:
        print(f'Setting: {setting}')
        print('Testing only (loading checkpoint)...')
        exp.test(setting)
        print('Testing completed.')


if __name__ == '__main__':
    main()
