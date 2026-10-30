"""Generate the large test dataset for match_sorted_x.

Writes data/A.csv (10^7 points) and data/B.csv (10^5 points), each with
columns x,y. Points are uniform over a 1000 x 1000 square. The seed is
fixed, so rerunning this produces identical files.

Usage: python data/generate_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

N_A = 10_000_000
N_B = 100_000
SIZE = 1000.0
SEED = 12345

DATA_DIR = Path(__file__).resolve().parent


def main():
    rng = np.random.default_rng(SEED)
    for name, n in (("A", N_A), ("B", N_B)):
        pts = rng.uniform(0.0, SIZE, size=(n, 2))
        path = DATA_DIR / f"{name}.csv"
        pd.DataFrame(pts, columns=["x", "y"]).to_csv(
            path, index=False, float_format="%.6f"
        )
        print(f"wrote {path} ({n:,} rows, {path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
