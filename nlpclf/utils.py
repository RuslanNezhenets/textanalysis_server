import numpy as np


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    t = max(float(temp), 1e-6)
    x = np.nan_to_num(x, neginf=-1e9, posinf=1e9)
    z = (x - np.max(x)) / t
    e = np.exp(z - np.max(z))
    return e / (np.sum(e) + 1e-12)

def _l2n(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    return m / n