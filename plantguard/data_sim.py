"""
data_sim.py — Synthetic sensor data generator for a plant asset (centrifugal pump).

Simulates three sensor channels a real plant historian would log:
  - bearing_temp_f   : bearing temperature (deg F)
  - vibration_ips     : vibration velocity (inches/sec)
  - discharge_psi     : discharge pressure (psi)

Most of the run is healthy, noisy-but-stable operation. Partway through,
we inject a slow degradation trend (e.g. bearing wear) that drifts the
readings toward failure thresholds, followed by a short spike right
before a simulated trip. This mimics the "slow creep, then fast failure"
pattern that predictive maintenance is built to catch early.
"""

import numpy as np
import pandas as pd


def generate_asset_run(
    n_points: int = 2000,
    seed: int = 42,
    degrade_start_frac: float = 0.65,
    fail_near_end: bool = True,
) -> pd.DataFrame:
    """Generate one simulated run (time series) for a single asset.

    Returns a DataFrame with columns:
        timestamp, bearing_temp_f, vibration_ips, discharge_psi, label
    label = 1 marks points inside the degradation/failure window (for
    supervised training / evaluation), 0 = normal operation.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)
    timestamps = pd.date_range("2026-01-01", periods=n_points, freq="min")

    # --- Healthy baseline signals (steady-state with realistic noise) ---
    bearing_temp = 145 + rng.normal(0, 1.2, n_points)          # deg F
    vibration = 0.12 + rng.normal(0, 0.015, n_points)           # in/sec
    discharge_psi = 220 + rng.normal(0, 3.0, n_points)          # psi

    # --- Inject a slow degradation trend in the back part of the run ---
    degrade_start = int(n_points * degrade_start_frac)
    label = np.zeros(n_points, dtype=int)

    if degrade_start < n_points:
        ramp_len = n_points - degrade_start
        ramp = np.linspace(0, 1, ramp_len) ** 1.8  # accelerating creep

        bearing_temp[degrade_start:] += ramp * 35        # up to +35F
        vibration[degrade_start:] += ramp * 0.28          # up to +0.28 ips
        discharge_psi[degrade_start:] -= ramp * 18         # pressure sags

        label[degrade_start:] = 1

        if fail_near_end:
            # sharp spike in the final ~2% of points, right before "trip"
            spike_len = max(5, int(n_points * 0.02))
            spike = np.linspace(0, 1, spike_len) ** 3
            bearing_temp[-spike_len:] += spike * 20
            vibration[-spike_len:] += spike * 0.15

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "bearing_temp_f": bearing_temp.round(2),
            "vibration_ips": vibration.round(4),
            "discharge_psi": discharge_psi.round(2),
            "label": label,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_asset_run()
    print(df.head())
    print("...")
    print(df.tail())
    print(f"\nRows: {len(df)}  |  Degraded/at-risk rows: {int(df['label'].sum())}")
