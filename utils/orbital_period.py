from __future__ import annotations
import math
import os
from typing import Optional, Iterable

import numpy as np
import pandas as pd


# Earth standard gravitational parameter (km^3 / s^2)
MU_EARTH_KM3_S2 = 398600.4418


def kepler_period_minutes(semi_major_axis_km: float) -> float:
    a = float(semi_major_axis_km)
    if a <= 0:
        raise ValueError(f"semi_major_axis must be positive, got {a}")
    T_sec = 2.0 * math.pi * math.sqrt(a ** 3 / MU_EARTH_KM3_S2)
    return T_sec / 60.0


def estimate_orbital_period_from_sma(
    sma_values: Iterable[float],
    reducer: str = "median",
) -> float:
    arr = np.asarray(list(sma_values), dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        raise ValueError("No valid SMA values to estimate orbital period.")

    if reducer == "median":
        a_repr = float(np.median(arr))
    elif reducer == "mean":
        a_repr = float(np.mean(arr))
    else:
        raise ValueError(f"Unknown reducer '{reducer}' (use 'median' or 'mean')")

    return kepler_period_minutes(a_repr)


def estimate_dt_minutes_from_timestamps(
    timestamps: Iterable,
    reducer: str = "median",
) -> float:
    ts = pd.to_datetime(list(timestamps))
    if len(ts) < 2:
        raise ValueError("Need at least 2 timestamps to estimate dt.")
    deltas = np.diff(ts.values).astype("timedelta64[s]").astype(np.float64) / 60.0
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        raise ValueError("No positive timestamp deltas found.")
    if reducer == "median":
        return float(np.median(deltas))
    elif reducer == "mean":
        return float(np.mean(deltas))
    else:
        raise ValueError(f"Unknown reducer '{reducer}'")


def auto_detect_period_and_dt(
    train_csv_path: str,
    sma_column: str = "Semi_major_axis",
    timestamp_column: str = "timestamp",
    reducer: str = "median",
) -> tuple[float, float]:
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")

    df = pd.read_csv(train_csv_path)

    if sma_column not in df.columns:
        raise KeyError(
            f"SMA column '{sma_column}' not in CSV columns: {list(df.columns)}"
        )
    T_orb = estimate_orbital_period_from_sma(df[sma_column].values, reducer=reducer)

    if timestamp_column is not None and timestamp_column in df.columns:
        dt = estimate_dt_minutes_from_timestamps(
            df[timestamp_column].values, reducer=reducer
        )
    else:
        dt = 1.0

    return T_orb, dt


if __name__ == "__main__":
    # Quick sanity check with the sample row from the dataset:
    #   a = 7068.735 km  -> expected T ≈ 98.58 min
    T = kepler_period_minutes(7068.735)
    print(f"a = 7068.735 km -> T = {T:.4f} min")