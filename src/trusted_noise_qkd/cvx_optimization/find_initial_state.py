from typing import Tuple

import cvxpy as cp
import numpy as np
from trusted_noise_qkd.cvx_optimization._solve import _solve
from trusted_noise_qkd.utils._herm import _herm


def find_feasible_rho(
    PiZerr: np.ndarray, PiXerr: np.ndarray, p: float, solver: str = "SCS"
) -> np.ndarray:
    rho = cp.Variable((4, 4), hermitian=True)
    constraints = [rho >> 0, cp.trace(rho) == 1]

    for a in range(2):
        for ap in range(2):
            expr = 0
            for b in range(2):
                i = a * 2 + b
                ip = ap * 2 + b
                expr += rho[i, ip]
            constraints.append(expr == (0.5 if a == ap else 0.0))

    constraints += [
        cp.real(cp.trace(PiZerr @ rho)) == p,
        cp.real(cp.trace(PiXerr @ rho)) == p,
    ]

    prob = cp.Problem(cp.Minimize(0), constraints)
    _solve(prob, solver=solver, scs_opts={"eps": 1e-7, "max_iters": 200000})

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError(f"Feasible rho SDP failed: {prob.status}")
    return _herm(rho.value)


def find_feasible_sigma(solver: str = "SCS") -> np.ndarray:
    sigma = cp.Variable((8, 8), hermitian=True)
    constraints = [sigma >> 0, cp.trace(sigma) == 1]
    prob = cp.Problem(cp.Minimize(0), constraints)
    _solve(prob, solver=solver, scs_opts={"eps": 1e-7, "max_iters": 200000})

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError(f"Feasible sigma SDP failed: {prob.status}")
    return _herm(sigma.value)


def find_initial_state(
    beta: float, PiZerr: np.ndarray, PiXerr: np.ndarray, p: float
) -> Tuple[np.ndarray, np.ndarray]:
    rho = find_feasible_rho(PiZerr, PiXerr, p)
    sigma = (
        np.zeros((8, 8), dtype=complex)
        if abs(beta - 1.0) < 1e-15
        else find_feasible_sigma()
    )
    return rho, sigma
