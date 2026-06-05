"""
Batch runner to reproduce all eight backbones in the paper.

Usage
-----
# Reproduce Table 1 (Loss-only plug-in baselines) on KOMPSAT-3:
python run_all.py --dataset K3 --group loss_only

# Reproduce Table 2 (Full plug-in models) on KOMPSAT-3A:
python run_all.py --dataset K3A --group full_plugin

# Reproduce everything:
python run_all.py --dataset K3 --group all
python run_all.py --dataset K3A --group all
"""

import argparse
import subprocess
import sys


# =============================================================================
# Backbone groups
# =============================================================================
FULL_PLUGIN = {
    'anomalytransformer':       {'epochs': 50, 'lr': 1e-4, 'batch_size': 60},
    'memto':                    {'epochs': 50, 'lr': 1e-4, 'batch_size': 60},
    'sub_adjacent_transformer': {'epochs': 50, 'lr': 1e-4, 'batch_size': 60},
}

LOSS_ONLY = {
    'dagmm':            {'epochs': 50, 'lr': 1e-3, 'batch_size': 60},
    'dtaad':            {'epochs': 50, 'lr': 1e-3, 'batch_size': 60},
    'lstm_autoencoder': {'epochs': 50, 'lr': 1e-3, 'batch_size': 60},
    'npsr':             {'epochs': 50, 'lr': 1e-3, 'batch_size': 60},
    'tranad':           {'epochs': 50, 'lr': 1e-3, 'batch_size': 60},
}


# =============================================================================
# Dataset config
# =============================================================================
DATASETS = {
    'K3': {
        'root_path':       './data/KOMPSAT-3_multiclass',
        'train_data_path': 'kompsat3_train_multiclass.csv',
        'test_data_path':  'kompsat3_test_multiclass.csv',
    },
    'K3A': {
        'root_path':       './data/KOMPSAT-3A_multiclass',
        'train_data_path': 'kompsat3a_train_multiclass.csv',
        'test_data_path':  'kompsat3a_test_multiclass.csv',
    },
}


# =============================================================================
# Runner
# =============================================================================
def run_one(model, cfg, ds_cfg, loss, use_ope, seed, gpu):
    cmd = [
        sys.executable, 'main.py',
        '--model',         model,
        '--train_epochs',  str(cfg['epochs']),
        '--learning_rate', str(cfg['lr']),
        '--batch_size',    str(cfg['batch_size']),
        '--loss',          loss,
        '--use_ope',       str(use_ope),
        '--seed',          str(seed),
        '--gpu',           str(gpu),
        '--root_path',       ds_cfg['root_path'],
        '--train_data_path', ds_cfg['train_data_path'],
        '--test_data_path',  ds_cfg['test_data_path'],
        '--data_path',       ds_cfg['train_data_path'],
    ]

    # Physics weights (paper default)
    if loss == 'Physics':
        cmd.extend([
            '--physics_lambda_smooth',  '0.1',
            '--physics_lambda_angular', '0.1',
        ])

    # OPE harmonics
    if use_ope:
        cmd.extend(['--ope_n_harmonics', '4'])

    print('\n' + '=' * 70)
    print(f'>>> Running {model:<26} loss={loss:<8} ope={use_ope} seed={seed}')
    print('=' * 70)
    subprocess.run(cmd, check=False)


def main():
    p = argparse.ArgumentParser(description='Batch runner for the paper experiments')
    p.add_argument('--dataset', choices=['K3', 'K3A'], required=True)
    p.add_argument('--group',   choices=['full_plugin', 'loss_only', 'all'], default='all')
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--gpu',     type=int, default=0)
    args = p.parse_args()

    ds_cfg = DATASETS[args.dataset]

    if args.group in ('loss_only', 'all'):
        # Loss-only plug-in: MSE and Physics for each backbone
        for model, cfg in LOSS_ONLY.items():
            for loss in ('MSE', 'Physics'):
                run_one(model, cfg, ds_cfg, loss=loss, use_ope=0,
                        seed=args.seed, gpu=args.gpu)

    if args.group in ('full_plugin', 'all'):
        # Full plug-in: 4 variants per backbone
        for model, cfg in FULL_PLUGIN.items():
            # base
            run_one(model, cfg, ds_cfg, loss='MSE',     use_ope=0,
                    seed=args.seed, gpu=args.gpu)
            # + OPE
            run_one(model, cfg, ds_cfg, loss='MSE',     use_ope=1,
                    seed=args.seed, gpu=args.gpu)
            # + Physics
            run_one(model, cfg, ds_cfg, loss='Physics', use_ope=0,
                    seed=args.seed, gpu=args.gpu)
            # + Both
            run_one(model, cfg, ds_cfg, loss='Physics', use_ope=1,
                    seed=args.seed, gpu=args.gpu)


if __name__ == '__main__':
    main()
