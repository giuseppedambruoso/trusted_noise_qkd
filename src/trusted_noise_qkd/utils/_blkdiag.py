import numpy as np


def _blkdiag(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [A, np.zeros((A.shape[0], B.shape[1]), dtype=complex)],
            [np.zeros((B.shape[0], A.shape[1]), dtype=complex), B],
        ]
    )
