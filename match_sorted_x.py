import numpy as np

def match_sorted_x(A, B, r):
    order = np.argsort(A[:, 0])
    xs = A[order, 0]
    ys = A[order, 1]

    lo = np.searchsorted(xs, B[:, 0] - r)
    hi = np.searchsorted(
        xs, B[:, 0] + r, side="right"
    )

    for j, (start, stop) in enumerate(zip(lo, hi)):
        d = (
            np.abs(xs[start:stop] - B[j, 0])
            + np.abs(ys[start:stop] - B[j, 1])
        )

        for i in order[start:stop][d <= r]:
            yield int(i), j
