import numpy as np
from trusted_noise_qkd.utils.zpinch import Zpinch


def build_Zsigma(sigma: np.ndarray) -> np.ndarray:
    return Zpinch(sigma)
