from typing import Tuple

import numpy as np
from trusted_noise_qkd.utils._blkdiag import _blkdiag
from trusted_noise_qkd.utils._herm import _herm
from trusted_noise_qkd.utils.G_dagger_eps import G_dagger_eps
from trusted_noise_qkd.utils.spectral_kernel import spectral_kernel
from trusted_noise_qkd.utils.zpinch import Zpinch


def objective_and_gradient(
    Grho: np.ndarray,
    Zsigma: np.ndarray,
    beta: float,
    K0: np.ndarray,
    K1: np.ndarray,
    eps_eig: float,
    eps_dep: float,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if abs(beta - 1.0) > 1e-15:
        mu = (1.0 - beta) / (2.0 * beta)

        ds_vals, Us = np.linalg.eigh(Zsigma)
        ds = np.maximum(np.real(ds_vals), eps_eig)

        Zsigma_mu = _herm(Us @ np.diag(ds**mu) @ Us.conj().T)
        Xi = _herm(Zsigma_mu @ Grho @ Zsigma_mu)

        dx_vals, Ux = np.linalg.eigh(Xi)
        dx = np.maximum(np.real(dx_vals), eps_eig)

        Q_beta = float(np.sum(dx**beta))
        tr_Grho = float(max(np.real(np.trace(Grho)), eps_eig))

        obj = float(np.real((1.0 / (beta - 1.0)) * np.log2(Q_beta / tr_Grho)))

        Xi_bm1 = _herm(Ux @ np.diag(dx ** (beta - 1.0)) @ Ux.conj().T)

        def tmu(A: np.ndarray) -> np.ndarray:
            return spectral_kernel(A, Us, ds, mu)

        chi1 = _herm(beta * Zpinch(tmu(Grho @ Zsigma_mu @ Xi_bm1)))
        chi2 = _herm(beta * (Zsigma_mu @ Xi_bm1 @ Zsigma_mu))
        chi3 = _herm(beta * Zpinch(tmu(Xi_bm1 @ Zsigma_mu @ Grho)))

        arg = chi2 / (Q_beta * np.log(2.0)) - np.eye(8, dtype=complex) / tr_Grho

        grad_rho = _herm((1.0 / (beta - 1.0)) * G_dagger_eps(arg, K0, K1, eps_dep))
        grad_sigma = _herm(
            (1.0 / (beta - 1.0)) * (chi1 + chi3) / (Q_beta * np.log(2.0))
        )

        grad_full = _blkdiag(grad_rho, grad_sigma)
        return obj, grad_full, grad_rho, grad_sigma

    vals_r, Ur = np.linalg.eigh(Grho)
    dr = np.maximum(np.real(vals_r), eps_eig)
    log2Grho = _herm(Ur @ np.diag(np.log(dr) / np.log(2.0)) @ Ur.conj().T)

    ZGrho = Zpinch(Grho)
    vals_z, Uz = np.linalg.eigh(ZGrho)
    dz = np.maximum(np.real(vals_z), eps_eig)
    log2ZGrho = _herm(Uz @ np.diag(np.log(dz) / np.log(2.0)) @ Uz.conj().T)

    obj = float(np.real(np.trace(Grho @ (log2Grho - log2ZGrho))))

    grad_rho = _herm(G_dagger_eps(log2Grho - log2ZGrho, K0, K1, eps_dep))
    grad_sigma = np.zeros((8, 8), dtype=complex)
    grad_full = _blkdiag(grad_rho, grad_sigma)
    return obj, grad_full, grad_rho, grad_sigma
