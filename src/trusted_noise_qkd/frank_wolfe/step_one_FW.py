from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from trusted_noise_qkd.cvx_optimization.sdp_descent import sdp_descent
from trusted_noise_qkd.frank_wolfe.FWOptions import FWOptions
from trusted_noise_qkd.objective_and_gradients.objective_and_gradient import (
    objective_and_gradient,
)
from trusted_noise_qkd.objective_and_gradients.objective_only import objective_only
from trusted_noise_qkd.utils._blkdiag import _blkdiag
from trusted_noise_qkd.utils._herm import _herm
from trusted_noise_qkd.utils.build_Grho_eps import build_Grho_eps
from trusted_noise_qkd.utils.build_Zsigma import build_Zsigma


def step_one_FW(
    rho: np.ndarray,
    sigma: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    beta: float,
    PiZerr: np.ndarray,
    PiXerr: np.ndarray,
    p: float,
    opts: FWOptions,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    for it in range(opts.maxIter):
        Grho = build_Grho_eps(rho, K0, K1, opts.eps_dep)
        Zsigma = (
            build_Zsigma(sigma)
            if abs(beta - 1.0) > 1e-15
            else np.zeros((8, 8), dtype=complex)
        )

        f_val, grad_full, grad_rho, grad_sigma = objective_and_gradient(
            Grho, Zsigma, beta, K0, K1, opts.eps_eig, opts.eps_dep
        )

        rho_f, sigma_f = sdp_descent(
            grad_rho, grad_sigma, PiZerr, PiXerr, p, beta, solver=opts.oracle_solver
        )

        gap = float(
            abs(
                np.real(
                    np.trace((rho - rho_f) @ grad_rho)
                    + np.trace((sigma - sigma_f) @ grad_sigma)
                )
            )
        )

        def obj_fun(lam: float) -> float:
            rho_l = _herm((1 - lam) * rho + lam * rho_f)
            sigma_l = _herm((1 - lam) * sigma + lam * sigma_f)
            return objective_only(
                rho_l, sigma_l, beta, K0, K1, opts.eps_eig, opts.eps_dep
            )

        res = minimize_scalar(obj_fun, bounds=(0.0, 1.0), method="bounded")
        lam = float(res.x)

        rho = _herm((1 - lam) * rho + lam * rho_f)
        sigma = _herm((1 - lam) * sigma + lam * sigma_f)

        if gap < opts.tol_gap:
            break

    rho_hat, sigma_hat = rho, sigma
    M_hat = _blkdiag(rho_hat, sigma_hat)

    Grho_hat = build_Grho_eps(rho_hat, K0, K1, opts.eps_dep)
    Zsigma_hat = (
        build_Zsigma(sigma_hat)
        if abs(beta - 1.0) > 1e-15
        else np.zeros((8, 8), dtype=complex)
    )

    f_hat, grad_full_hat, grad_rho_hat, grad_sigma_hat = objective_and_gradient(
        Grho_hat, Zsigma_hat, beta, K0, K1, opts.eps_eig, opts.eps_dep
    )
    return M_hat, float(f_hat), grad_full_hat, grad_rho_hat, grad_sigma_hat
