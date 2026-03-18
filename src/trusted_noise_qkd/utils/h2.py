import numpy as np


def h2(x: float) -> float:
    x = float(np.real(x))
    x = min(max(x, 1e-15), 1 - 1e-15)
    return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))
