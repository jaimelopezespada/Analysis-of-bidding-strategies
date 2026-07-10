"""
Generate a synthetic OMIE-style price CSV for development and tests.

Produces 90 days (2025-01-01 to 2025-03-31) x 24 hourly periods with
realistic Spanish day-ahead price patterns: morning peak, solar midday
valley, evening peak, and day-to-day stochastic variation.

Run once:
    python data/generate_example.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_DAYS = 90
START_DATE = "2025-01-01"
OUT_FILE = Path(__file__).parent / "example_omie.csv"

# ------------------------------------------------------------------
# Base hourly pattern (€/MWh) — typical Spanish OMIE profile
# ------------------------------------------------------------------
BASE_PATTERN = np.array([
    42, 40, 38, 37, 38, 43,   # 00–05h  off-peak night
    54, 66, 76, 83, 86, 73,   # 06–11h  morning ramp + peak
    63, 58, 61, 69, 76, 81,   # 12–17h  solar valley + afternoon
    89, 93, 91, 83, 71, 53,   # 18–23h  evening peak + decline
], dtype=float)


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")

    rows = []
    for day in dates:
        # Day-level stochastic components
        level_shift = rng.normal(0, 18)        # overall price level that day
        scale = rng.uniform(0.65, 1.45)        # daily volatility factor
        # Occasional high-price spike days (winter/low-wind events)
        spike = rng.choice([0.0, 40.0], p=[0.88, 0.12])

        for h in range(24):
            noise = rng.normal(0, 7)
            price = max(0.0, BASE_PATTERN[h] * scale + level_shift + noise + spike)
            rows.append({
                "date": day.date().isoformat(),
                "period": h + 1,
                "price": round(price, 2),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_FILE, index=False)
    print(f"Written {len(df)} rows ({N_DAYS} days x 24 periods) to {OUT_FILE}")
    print(f"Price stats:  mean={df['price'].mean():.1f}  "
          f"std={df['price'].std():.1f}  "
          f"min={df['price'].min():.1f}  "
          f"max={df['price'].max():.1f}  €/MWh")
