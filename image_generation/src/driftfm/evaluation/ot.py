from __future__ import annotations

import numpy as np


def compute_ot_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    solver: str = "emd",
    metric: str = "l2_sq",
    sinkhorn_reg: float = 0.05,
    ot_iters: int = 200000,
) -> float:
    try:
        import ot
    except ImportError as exc:
        raise ImportError("POT is required for OT evaluation. Install the `eval` extra.") from exc

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_x = x.shape[0]
    n_y = y.shape[0]
    a = np.full(n_x, 1.0 / n_x, dtype=np.float64)
    b = np.full(n_y, 1.0 / n_y, dtype=np.float64)
    ot_metric = "sqeuclidean" if metric == "l2_sq" else "euclidean"
    cost = ot.dist(x, y, metric=ot_metric)
    if solver == "emd":
        return float(ot.emd2(a, b, cost, numItermax=ot_iters))
    if solver == "sinkhorn":
        return float(ot.sinkhorn2(a, b, cost, reg=sinkhorn_reg, numItermax=ot_iters, method="sinkhorn_log"))
    raise ValueError(f"Unsupported OT solver: {solver}")
