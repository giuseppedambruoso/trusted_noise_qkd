from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from trusted_noise_qkd.cvx_optimization.find_initial_state import find_initial_state
from trusted_noise_qkd.frank_wolfe.FWOptions import FWOptions
from trusted_noise_qkd.frank_wolfe.step_one_FW import step_one_FW
from trusted_noise_qkd.frank_wolfe.step_two_FW import step_two_FW
from trusted_noise_qkd.utils.h2 import h2


def frank_wolfe(
    beta: float,
    p: float,
    q: float,
    opts: FWOptions,
    init_state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    Id2 = np.eye(2, dtype=complex)
    ket0 = np.array([[1], [0]], dtype=complex)
    ket1 = np.array([[0], [1]], dtype=complex)
    plus = (ket0 + ket1) / np.sqrt(2)
    minus = (ket0 - ket1) / np.sqrt(2)

    Z0 = ket0 @ ket0.conj().T
    Z1 = ket1 @ ket1.conj().T
    Pp = plus @ plus.conj().T
    Pm = minus @ minus.conj().T

    PiZerr = np.kron(Z0, Z1) + np.kron(Z1, Z0)
    PiXerr = np.kron(Pp, Pm) + np.kron(Pm, Pp)

    sqrtL0 = np.diag([np.sqrt(1 - q), np.sqrt(q)])
    sqrtL1 = np.diag([np.sqrt(q), np.sqrt(1 - q)])

    K0 = np.kron(Id2, sqrtL0)
    K1 = np.kron(Id2, sqrtL1)

    if init_state is not None:
        rho0, sigma0 = init_state
    else:
        rho0, sigma0 = find_initial_state(beta, PiZerr, PiXerr, p)

    M_hat, f_hat, grad_full_hat, grad_rho_hat, grad_sigma_hat = step_one_FW(
        rho0, sigma0, K0, K1, beta, PiZerr, PiXerr, p, opts
    )

    rho_final = M_hat[0:4, 0:4]
    sigma_final = M_hat[4:12, 4:12]
    final_state = (rho_final, sigma_final)

    if opts.use_step2:
        val = step_two_FW(
            f_hat,
            M_hat,
            grad_rho_hat,
            grad_sigma_hat,
            grad_full_hat,
            PiZerr,
            PiXerr,
            p,
            beta,
            opts,
        )
        return val, final_state

    return float(f_hat), final_state


def rate_raw_fs(
    beta: float,
    p: float,
    q: float,
    alpha: float,
    N: int,
    opts: FWOptions,
    eps_PA: float,
    init_state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    lb, final_state = frank_wolfe(beta, p, q, opts, init_state=init_state)
    leak = h2(p + q - 2 * p * q)
    g_alpha = np.log2(1.0 / eps_PA) * alpha / (alpha - 1.0) - 2.0
    return float(lb - leak - g_alpha / float(N)), final_state


def rate_raw_asym(
    beta: float,
    p: float,
    q: float,
    alpha: float,
    opts: FWOptions,
    eps_PA: float,
    init_state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    lb, final_state = frank_wolfe(beta, p, q, opts, init_state=init_state)
    leak = h2(p + q - 2 * p * q)
    return float(lb - leak), final_state


class StateCache:
    def __init__(self):
        self.last_state = None


def beta_from_alpha(alpha: float) -> float:
    return float(alpha / (2.0 * alpha - 1.0))


def optimize_q(
    objective_func,
    q_lo: float = 0.0,
    q_hi: float = 0.5,
) -> Tuple[float, float]:
    cache = StateCache()

    def neg_obj(q):
        val, new_state = objective_func(q, cache.last_state)
        cache.last_state = new_state
        return -val

    res = minimize_scalar(
        neg_obj,
        bounds=(q_lo, q_hi),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 40},
    )
    q_star = float(res.x)
    raw_star = float(-res.fun)
    return q_star, raw_star
