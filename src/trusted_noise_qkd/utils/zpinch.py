import numpy as np
from trusted_noise_qkd.utils._herm import _herm


def Zpinch(X: np.ndarray) -> np.ndarray:
    ZX = np.zeros_like(X, dtype=complex)
    ZX[0:4, 0:4] = X[0:4, 0:4]
    ZX[4:8, 4:8] = X[4:8, 4:8]
    return _herm(ZX)
