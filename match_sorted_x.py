import numpy as np

METRICS = ("manhattan", "euclidean", "square")


def match_sorted_x(A, B, r, metric="manhattan"):
    """Yield (i, j) pairs where point A[i] is within r of point B[j].

    A and B are arrays of shape (n, 2) and (m, 2) holding (x, y) points.

    metric:
        "manhattan" - |dx| + |dy| <= r
        "euclidean" - sqrt(dx**2 + dy**2) <= r
        "square"    - |dx| <= r and |dy| <= r (a square of half width r
                      centered on each B point)
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")

    order = np.argsort(A[:, 0])
    xs = A[order, 0]
    ys = A[order, 1]

    # Every metric requires |dx| <= r, so only A points in this x-strip
    # can match.
    lo = np.searchsorted(xs, B[:, 0] - r)
    hi = np.searchsorted(xs, B[:, 0] + r, side="right")

    for j, (start, stop) in enumerate(zip(lo, hi)):
        dx = xs[start:stop] - B[j, 0]
        dy = ys[start:stop] - B[j, 1]

        if metric == "manhattan":
            mask = np.abs(dx) + np.abs(dy) <= r
        elif metric == "euclidean":
            # Compare squared distances to avoid a sqrt.
            mask = dx * dx + dy * dy <= r * r
        else:  # square
            # The x-strip already guarantees |dx| <= r.
            mask = np.abs(dy) <= r

        for i in order[start:stop][mask]:
            yield int(i), j
