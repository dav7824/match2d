"""Test match_sorted_x on the large dataset in data/ (A: 10^7, B: 10^5).

Generate the data first with: python data/generate_dataset.py

Run with: python tests/test_large_dataset.py   (or pytest, if installed)

Comparing all 10^12 pairs directly is too slow, so the test checks:
  1. Every reported pair satisfies its metric (soundness, full dataset).
  2. No pair is reported twice.
  3. For a random sample of B points, the reported matches equal a
     brute-force scan over all of A (completeness).
  4. manhattan matches are a subset of euclidean, which are a subset of
     square, as the geometry requires.
"""
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from match_sorted_x import METRICS, match_sorted_x  # noqa: E402

DATA_DIR = ROOT / "data"
R = 0.1  # matching radius / half width
N_SAMPLE = 50  # B points checked by brute force


def within(dx, dy, metric):
    if metric == "manhattan":
        return np.abs(dx) + np.abs(dy) <= R
    if metric == "euclidean":
        return dx * dx + dy * dy <= R * R
    return (np.abs(dx) <= R) & (np.abs(dy) <= R)


@lru_cache(maxsize=None)
def load():
    paths = [DATA_DIR / "A.csv", DATA_DIR / "B.csv"]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing}; run: python data/generate_dataset.py"
        )
    A, B = (pd.read_csv(p, dtype=np.float64).to_numpy() for p in paths)
    assert A.shape == (10_000_000, 2) and B.shape == (100_000, 2)
    return A, B


@lru_cache(maxsize=None)
def matches(metric):
    A, B = load()
    pairs = np.array(list(match_sorted_x(A, B, R, metric)), dtype=np.int64)
    return pairs.reshape(-1, 2)


def as_set(pairs):
    return set(map(tuple, pairs.tolist()))


def test_pairs_satisfy_metric():
    A, B = load()
    for metric in METRICS:
        pairs = matches(metric)
        assert len(pairs) > 0, metric
        i, j = pairs[:, 0], pairs[:, 1]
        ok = within(A[i, 0] - B[j, 0], A[i, 1] - B[j, 1], metric)
        assert ok.all(), f"{metric}: {(~ok).sum()} pairs out of range"


def test_no_duplicate_pairs():
    for metric in METRICS:
        pairs = matches(metric)
        assert len(as_set(pairs)) == len(pairs), metric


def test_sample_matches_brute_force():
    A, B = load()
    rng = np.random.default_rng(0)
    sample = rng.choice(len(B), size=N_SAMPLE, replace=False)
    found = {m: matches(m) for m in METRICS}
    for j in sample:
        dx = A[:, 0] - B[j, 0]
        dy = A[:, 1] - B[j, 1]
        for metric in METRICS:
            expected = set(np.flatnonzero(within(dx, dy, metric)).tolist())
            pairs = found[metric]
            got = set(pairs[pairs[:, 1] == j, 0].tolist())
            assert got == expected, f"{metric}, B[{j}]"


def test_metric_nesting():
    man, euc, sq = (as_set(matches(m)) for m in METRICS)
    assert man <= euc <= sq


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        start = time.perf_counter()
        t()
        print(f"PASS {t.__name__} ({time.perf_counter() - start:.1f}s)")
    for m in METRICS:
        print(f"  {m}: {len(matches(m)):,} pairs")
    print(f"all {len(tests)} tests passed")
