from __future__ import annotations
import os
import torch.nn as nn

from utils.ope_module import OrbitalPeriodEmbedding
from utils.orbital_period import auto_detect_period_and_dt


# Map from model name (lowercase, as in args.model) to the dotted path of the
# positional-embedding attribute inside the model module.
_PE_ATTR_PATHS = {
    'anomalytransformer':        'embedding.position_embedding',
    'memto':                     'memto_model.embedding.pos_embedding',
    'sub_adjacent_transformer':  'core.embedding.position_embedding',
    'sat':                       'core.embedding.position_embedding',  # alias
}


def _get_nested_attr(obj, dotted: str):
    cur = obj
    for part in dotted.split('.'):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _set_nested_attr(obj, dotted: str, new_value):
    parts = dotted.split('.')
    parent = obj
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_value)


def _infer_d_model_from_pe(old_pe: nn.Module) -> int:
    if hasattr(old_pe, 'pe') and hasattr(old_pe.pe, 'shape'):
        return int(old_pe.pe.shape[-1])
    # Fallback: search any buffer with 3-d shape
    for _, buf in old_pe.named_buffers():
        if buf.dim() == 3:
            return int(buf.shape[-1])
    raise RuntimeError(
        "Could not infer d_model from existing positional embedding module."
    )


def _resolve_period_and_dt(args, default=97.0, default_dt=1.0):
    manual_T  = getattr(args, 'ope_period', None) or getattr(args, 'orb_period', None)
    manual_dt = getattr(args, 'ope_dt_minutes', None) or getattr(args, 'orb_dt_minutes', None)
    force_auto = bool(
        getattr(args, 'ope_auto_period', 0) or getattr(args, 'orb_auto_period', 0)
    )
    train_csv = (
        getattr(args, 'ope_train_csv_path', None)
        or getattr(args, 'orb_train_csv_path', None)
    )
    sma_col = getattr(args, 'sma_column', 'Semi_major_axis')
    ts_col  = getattr(args, 'timestamp_column', 'timestamp')

    # Auto-resolve CSV path from root_path + train_data_path if not set
    if not train_csv:
        root = getattr(args, 'root_path', None)
        train_name = (
            getattr(args, 'train_data_path', None)
            or getattr(args, 'data_path', None)
        )
        if root and train_name:
            candidate = os.path.join(root, train_name)
            if os.path.exists(candidate):
                train_csv = candidate

    # 1) manual override
    if (not force_auto) and (manual_T is not None) and (float(manual_T) > 0):
        T = float(manual_T)
        dt = float(manual_dt) if (manual_dt and float(manual_dt) > 0) else default_dt
        return T, dt, 'manual'

    # 2) auto via Kepler
    if train_csv:
        try:
            T, dt = auto_detect_period_and_dt(
                train_csv, sma_column=sma_col, timestamp_column=ts_col,
            )
            if manual_dt and float(manual_dt) > 0:
                dt = float(manual_dt)
            return T, dt, 'auto'
        except Exception as ex:
            print(f"[OPE patch] auto period detection failed ({ex}); "
                  f"falling back to default T={default}min.")

    # 3) default
    return float(default), float(default_dt), 'default'


def maybe_patch_with_ope(model: nn.Module, args) -> bool:
    if not bool(getattr(args, 'use_ope', 0)):
        return False

    model_name = str(getattr(args, 'model', '')).lower()
    if model_name not in _PE_ATTR_PATHS:
        print(f"[OPE patch] WARNING: '{model_name}' not in supported list "
              f"{list(_PE_ATTR_PATHS.keys())}. Skipping OPE patch.")
        return False

    pe_path = _PE_ATTR_PATHS[model_name]
    old_pe = _get_nested_attr(model, pe_path)
    if old_pe is None:
        print(f"[OPE patch] WARNING: attribute '{pe_path}' not found on "
              f"{type(model).__name__}. Skipping OPE patch.")
        return False

    d_model = _infer_d_model_from_pe(old_pe)
    T_orb, dt, source = _resolve_period_and_dt(args)
    n_harm = int(getattr(args, 'ope_n_harmonics', 4))

    new_pe = OrbitalPeriodEmbedding(
        d_model=d_model,
        n_harmonics=n_harm,
        period=T_orb,
        dt_minutes=dt,
    )

    # Move new PE to the same device as the existing module
    try:
        device = next(model.parameters()).device
        new_pe = new_pe.to(device)
    except StopIteration:
        pass

    _set_nested_attr(model, pe_path, new_pe)

    print(f"[OPE patch] {model_name}: replaced PE at '{pe_path}'  "
          f"d_model={d_model}, T_orb={T_orb:.4f} min, dt={dt:.4f} min, "
          f"source={source}, n_harmonics={n_harm}")
    return True


def ope_setting_tag(args) -> str:
    if not bool(getattr(args, 'use_ope', 0)):
        return 'ope0'
    auto = bool(
        getattr(args, 'ope_auto_period', 0) or getattr(args, 'orb_auto_period', 0)
    )
    if auto:
        return 'ope1_autoT'
    period = (
        getattr(args, 'ope_period', None)
        or getattr(args, 'orb_period', None)
        or 97.0
    )
    return f"ope1_T{int(period)}"
