from typing import Tuple

import cvxpy as cp
import numpy as np
from trusted_noise_qkd.cvx_optimization._solve import _solve
from trusted_noise_qkd.utils._herm import _herm


def sdp_descent(
    grad_rho: np.ndarray,
    grad_sigma: np.ndarray,
    PiZerr: np.ndarray,
    PiXerr: np.ndarray,
    p: float,
    beta: float,
    solver: str = "MOSEK",
) -> Tuple[np.ndarray, np.ndarray]:

    if abs(beta - 1.0) < 1e-15:
        rho_var = cp.Variable((4, 4), hermitian=True)
        constraints = [rho_var >> 0, cp.trace(rho_var) == 1]

        for a in range(2):
            for ap in range(2):
                expr = 0
                for b in range(2):
                    i = a * 2 + b
                    ip = ap * 2 + b
                    expr += rho_var[i, ip]
                constraints.append(expr == (0.5 if a == ap else 0.0))

        constraints += [
            cp.real(cp.trace(PiZerr @ rho_var)) == p,
            cp.real(cp.trace(PiXerr @ rho_var)) == p,
        ]

        prob = cp.Problem(
            cp.Minimize(cp.real(cp.trace(grad_rho @ rho_var))), constraints
        )
        _solve(prob, solver=solver, scs_opts={"eps": 1e-7, "max_iters": 200000})
        return _herm(rho_var.value), np.zeros((8, 8), dtype=complex)

    rho_var = cp.Variable((4, 4), hermitian=True)
    sigma_var = cp.Variable((8, 8), hermitian=True)

    constraints = [
        rho_var >> 0,
        sigma_var >> 0,
        cp.trace(rho_var) == 1,
        cp.trace(sigma_var) == 1,
    ]

    for a in range(2):
        for ap in range(2):
            expr = 0
            for b in range(2):
                i = a * 2 + b
                ip = ap * 2 + b
                expr += rho_var[i, ip]
            constraints.append(expr == (0.5 if a == ap else 0.0))

    constraints += [
        cp.real(cp.trace(PiZerr @ rho_var)) == p,
        cp.real(cp.trace(PiXerr @ rho_var)) == p,
    ]

    prob = cp.Problem(
        cp.Minimize(
            cp.real(cp.trace(grad_rho @ rho_var) + cp.trace(grad_sigma @ sigma_var))
        ),
        constraints,
    )
    _solve(prob, solver=solver, scs_opts={"eps": 1e-7, "max_iters": 200000})
    return _herm(rho_var.value), _herm(sigma_var.value)
