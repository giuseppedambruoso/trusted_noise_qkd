import numpy as np


def _herm(A: np.ndarray) -> np.ndarray:
    return (A + A.conj().T) / 2.0
