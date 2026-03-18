import numpy as np
from trusted_noise_qkd.utils._herm import _herm
from trusted_noise_qkd.utils.build_Grho_eps import build_Grho_eps
from trusted_noise_qkd.utils.build_Zsigma import build_Zsigma
from trusted_noise_qkd.utils.zpinch import Zpinch


def objective_only(
    rho: np.ndarray,
    sigma: np.ndarray,
    beta: float,
    K0: np.ndarray,
    K1: np.ndarray,
    eps_eig: float,
    eps_dep: float,
) -> float:
    Grho = build_Grho_eps(rho, K0, K1, eps_dep)

    if abs(beta - 1.0) > 1e-15:
        Zsigma = build_Zsigma(sigma)
        mu = (1.0 - beta) / (2.0 * beta)

        ds_vals, Us = np.linalg.eigh(Zsigma)
        ds = np.maximum(np.real(ds_vals), eps_eig)

        Zsigma_mu = _herm(Us @ np.diag(ds**mu) @ Us.conj().T)
        Xi = _herm(Zsigma_mu @ Grho @ Zsigma_mu)

        dx_vals, _ = np.linalg.eigh(Xi)
        dx = np.maximum(np.real(dx_vals), eps_eig)

        Q_beta = float(np.sum(dx**beta))
        tr_Grho = float(max(np.real(np.trace(Grho)), eps_eig))

        return float(np.real((1.0 / (beta - 1.0)) * np.log2(Q_beta / tr_Grho)))

    ZGrho = Zpinch(Grho)

    vals_r, Ur = np.linalg.eigh(Grho)
    dr = np.maximum(np.real(vals_r), eps_eig)
    log2Grho = _herm(Ur @ np.diag(np.log(dr) / np.log(2.0)) @ Ur.conj().T)

    vals_z, Uz = np.linalg.eigh(ZGrho)
    dz = np.maximum(np.real(vals_z), eps_eig)
    log2ZGrho = _herm(Uz @ np.diag(np.log(dz) / np.log(2.0)) @ Uz.conj().T)

    return float(np.real(np.trace(Grho @ (log2Grho - log2ZGrho))))
