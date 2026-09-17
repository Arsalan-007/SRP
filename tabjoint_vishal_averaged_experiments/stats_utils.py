
import numpy as np

def offdiag_stats(M: np.ndarray) -> dict:
    n = M.shape[0]
    off = M[~np.eye(n, dtype=bool)]
    return {
        "mean_offdiag": float(off.mean()),
        "std_offdiag":  float(off.std()),
        "min_offdiag":  float(off.min()),
        "max_offdiag":  float(off.max()),
    }
