import numpy as np
from trusted_noise_qkd.utils._herm import _herm


def G_dagger_eps(
    X8: np.ndarray, K0: np.ndarray, K1: np.ndarray, eps_dep: float
) -> np.ndarray:
    X00 = X8[0:4, 0:4]
    X01 = X8[0:4, 4:8]
    X10 = X8[4:8, 0:4]
    X11 = X8[4:8, 4:8]

    X = (
        K0.conj().T @ X00 @ K0
        + K0.conj().T @ X01 @ K1
        + K1.conj().T @ X10 @ K0
        + K1.conj().T @ X11 @ K1
    )

    return _herm((1 - eps_dep) * X)
