import numpy as np
from trusted_noise_qkd.utils._herm import _herm


def build_Grho_eps(
    rho: np.ndarray, K0: np.ndarray, K1: np.ndarray, eps_dep: float
) -> np.ndarray:
    G00 = K0 @ rho @ K0.conj().T
    G01 = K0 @ rho @ K1.conj().T
    G10 = K1 @ rho @ K0.conj().T
    G11 = K1 @ rho @ K1.conj().T

    Grho = np.block([[G00, G01], [G10, G11]])

    Grho = (1 - eps_dep) * Grho + (eps_dep / 8.0) * np.eye(8, dtype=complex)
    return _herm(Grho)
