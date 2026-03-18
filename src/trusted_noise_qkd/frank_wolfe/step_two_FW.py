import numpy as np
from trusted_noise_qkd.cvx_optimization.sdp_descent import sdp_descent
from trusted_noise_qkd.frank_wolfe.FWOptions import FWOptions
from trusted_noise_qkd.utils._blkdiag import _blkdiag


def step_two_FW(
    f_hat: float,
    M_hat: np.ndarray,
    grad_rho_hat: np.ndarray,
    grad_sigma_hat: np.ndarray,
    grad_full_hat: np.ndarray,
    PiZerr: np.ndarray,
    PiXerr: np.ndarray,
    p: float,
    beta: float,
    opts: FWOptions,
) -> float:
    first_term = float(np.real(f_hat))
    second_term = -float(np.real(np.trace(grad_full_hat @ M_hat)))

    rho_opt, sigma_opt = sdp_descent(
        grad_rho_hat, grad_sigma_hat, PiZerr, PiXerr, p, beta, solver=opts.oracle_solver
    )
    M_opt = _blkdiag(rho_opt, sigma_opt)
    third_term = float(np.real(np.trace(grad_full_hat @ M_opt)))

    return float(np.real(first_term + second_term + third_term))
