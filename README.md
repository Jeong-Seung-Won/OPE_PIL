# Orbital Period Embedding and Physics-Informed Loss for Satellite Anomaly Detection

Official implementation of our CIKM 2026 short paper:

> **Orbital Period Embedding and Physics-Informed Loss for Satellite Anomaly Detection**
> *CIKM 2026 (Short Paper Track)*

We introduce two model-agnostic plug-in components for satellite orbit anomaly detection:

- **Adaptive Orbital Period Embedding (AOPE)**: a positional embedding whose fundamental period is automatically derived from training data via Kepler's third law.
- **Physics-informed Loss**: a reconstruction objective augmented with temporal smoothness and angular momentum conservation.

Both components attach to existing reconstruction-based AD backbones with a single-module substitution or an additive loss term. We evaluate on a new KOMPSAT-3 / KOMPSAT-3A benchmark constructed via SACM-based anomaly injection across eight state-of-the-art backbones.

---

## Quick start

```bash
# Default backbone (Anomaly Transformer), MSE loss, no OPE:
python main.py

# Physics-informed Loss on the default backbone:
python main.py --loss Physics

# Both plug-in components on MEMTO:
python main.py --loss Physics --use_ope 1 --model memto

# Sub-Adjacent Transformer with both components:
python main.py --loss Physics --use_ope 1 --model sub_adjacent_transformer
```

## Backbones

| Backbone | Group | Venue | `--model` |
|----------|-------|-------|-----------|
| Anomaly Transformer    | Full plug-in       | ICLR 2022    | `anomalytransformer` |
| MEMTO                  | Full plug-in       | NeurIPS 2023 | `memto`              |
| Sub-Adjacent Transformer | Full plug-in     | IJCAI 2024   | `sub_adjacent_transformer` |
| DAGMM                  | Loss-only plug-in  | ICLR 2018    | `dagmm`              |
| DTAAD                  | Loss-only plug-in  | Knowledge-Based Systems 2024 | `dtaad` |
| LSTM-AE                | Loss-only plug-in  | ICML 2015 WS | `lstm_autoencoder`   |
| NPSR                   | Loss-only plug-in  | NeurIPS 2023 | `npsr`               |
| TranAD                 | Loss-only plug-in  | VLDB 2022    | `tranad`             |

- **Full plug-in models** support both OPE (`--use_ope 1`) and Physics Loss (`--loss Physics`).
- **Loss-only plug-in models** use no positional embedding (or one incompatible with direct OPE substitution); only Physics Loss applies.

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--model`             | `anomalytransformer` | AD backbone (see table above) |
| `--loss`              | `MSE`     | `MSE`, `Physics`, or `SATLoss` (SAT only) |
| `--use_ope`           | `0`       | Set to `1` to enable Adaptive OPE |
| `--ope_n_harmonics`   | `4`       | Number of harmonics K |
| `--ope_auto_period`   | `1`       | Estimate T_orb via Kepler third law from data |
| `--physics_lambda_smooth`  | `0.1` | Temporal smoothness weight λ_s |
| `--physics_lambda_angular` | `0.1` | Angular momentum weight λ_a |
| `--seq_len`           | `48`      | Sliding window length L |
| `--seed`              | `42`      | Random seed |

## Repository structure

```
.
├── main.py                  # Unified entry point for all eight backbones
├── requirements.txt
│
├── data_provider/
│   ├── data_factory.py      # Builds the train/test DataLoader
│   └── data_loader.py       # KAFASATAnomalyDataset (sliding-window TLE)
│
├── exp/                     # Experiment classes
│   ├── exp_basic.py         # Backbone registry
│   ├── exp_ad.py            # Standard reconstruction training/evaluation loop
│   └── exp_ad_sat.py        # SAT-specific training loop
│
├── models/                  # Backbone implementations (eight paper models)
│   ├── anomalytransformer.py
│   ├── memto.py
│   ├── sub_adjacent_transformer.py
│   ├── dagmm.py
│   ├── dtaad.py
│   ├── lstm_autoencoder.py
│   ├── npsr.py
│   └── tranad.py
│
└── utils/
    ├── ope_module.py        # OrbitalPeriodEmbedding (Section 4.2)
    ├── ope_patch.py         # Plug-in substitution into Transformer backbones
    ├── physics_loss.py      # Physics-informed Loss (Section 4.3)
    ├── orbital_period.py    # T_orb estimation via Kepler third law
    ├── metrics.py           # Reconstruction-error metrics (MAE / MSE / RMSE)
    └── tools.py             # EarlyStopping, learning-rate schedule
```

## License

MIT License — see [LICENSE](LICENSE).
