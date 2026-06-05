"""
Orbital period auto-detection utilities for OrbFormer.

Kepler's third law (two-body problem):
    T = 2π * sqrt(a^3 / μ)
where
    a : semi-major axis (km)
    μ : Earth gravitational parameter = 398600.4418 km^3/s^2
    T : orbital period (seconds)

This module computes the orbital period from the training data's SMA column,
enabling Orbital Period Embedding (OPE) to be dataset-adaptive instead of
relying on a fixed magic number (e.g., 97.0 minutes).
"""

from __future__ import annotations
import math
import os
from typing import Optional, Iterable

import numpy as np
import pandas as pd


# Earth standard gravitational parameter (km^3 / s^2)
MU_EARTH_KM3_S2 = 398600.4418


def kepler_period_minutes(semi_major_axis_km: float) -> float:
    """
    Compute orbital period in minutes from a single semi-major axis value.

    Args:
        semi_major_axis_km: Semi-major axis in kilometers.

    Returns:
        Orbital period in minutes.

    Example:
        >>> kepler_period_minutes(7068.735)
        98.57...
    """
    a = float(semi_major_axis_km)
    if a <= 0:
        raise ValueError(f"semi_major_axis must be positive, got {a}")
    T_sec = 2.0 * math.pi * math.sqrt(a ** 3 / MU_EARTH_KM3_S2)
    return T_sec / 60.0


def estimate_orbital_period_from_sma(
    sma_values: Iterable[float],
    reducer: str = "median",
) -> float:
    """
    Estimate the orbital period (in minutes) from a series of SMA observations.

    Args:
        sma_values: 1D iterable of semi-major axis values (km).
                    May come from a CSV column or numpy array.
        reducer:    How to aggregate SMA across the training set.
                    'median' (default, robust to anomalies/outliers) or 'mean'.

    Returns:
        Orbital period in minutes.

    Notes:
        Median is preferred over mean because the training set is
        anomaly-free in our semi-supervised setting (2016-2019 normal data),
        but residual orbital drift / station-keeping maneuvers can still
        produce SMA outliers. Median is robust to those.
    """
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
    """
    Estimate the sampling interval (dt, in minutes) from a timestamp column.

    Args:
        timestamps: Iterable of timestamps (parseable by pandas.to_datetime).
        reducer:    'median' (robust to gaps) or 'mean'.

    Returns:
        Sampling interval in minutes.
    """
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
    """
    Convenience: load training CSV, return (T_orb_minutes, dt_minutes).

    Args:
        train_csv_path:    Path to training CSV.
        sma_column:        Column name for semi-major axis (km).
        timestamp_column:  Column name for timestamp. Set to None to skip
                           dt estimation (returns dt=1.0).
        reducer:           'median' or 'mean'.

    Returns:
        (T_orb_minutes, dt_minutes)
    """
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