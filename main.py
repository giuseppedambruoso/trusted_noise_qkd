# main.py
from __future__ import annotations

import warnings
import time
import csv
import sys
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar

from config import config

# ============================================================
# Helpers & Forward/Adjoint Maps (UNCHANGED)
# ============================================================

def _herm(A: np.ndarray) -> np.ndarray:
    return (A + A.conj().T) / 2.0

def _blkdiag(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.block([
        [A, np.zeros((A.shape[0], B.shape[1]), dtype=complex)],
        [np.zeros((B.shape[0], A.shape[1]), dtype=complex), B],
    ])

def h2(x: float) -> float:
    x = float(np.real(x))
    x = min(max(x, 1e-15), 1 - 1e-15)
    return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))

def Zpinch(X: np.ndarray) -> np.ndarray:
    ZX = np.zeros_like(X, dtype=complex)
    ZX[0:4, 0:4] = X[0:4, 0:4]
    ZX[4:8, 4:8] = X[4:8, 4:8]
    return _herm(ZX)

def build_Zsigma(sigma: np.ndarray) -> np.ndarray:
    return Zpinch(sigma)

def spectral_kernel(A: np.ndarray, U: np.ndarray, d: np.ndarray, mu: float) -> np.ndarray:
    tol = 1e-14
    Atil = U.conj().T @ A @ U
    n = len(d)
    M = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            di = max(float(np.real(d[i])), tol)
            dj = max(float(np.real(d[j])), tol)
            if i == j:
                M[i, j] = mu * di ** (mu - 1.0)
            else:
                denom = di - dj
                if abs(denom) < tol:
                    M[i, j] = mu * di ** (mu - 1.0)
                else:
                    M[i, j] = (di ** mu - dj ** mu) / denom

    return U @ (M * Atil) @ U.conj().T

def build_Grho_eps(rho: np.ndarray, K0: np.ndarray, K1: np.ndarray, eps_dep: float) -> np.ndarray:
    G00 = K0 @ rho @ K0.conj().T
    G01 = K0 @ rho @ K1.conj().T
    G10 = K1 @ rho @ K0.conj().T
    G11 = K1 @ rho @ K1.conj().T

    Grho = np.block([[G00, G01],
                     [G10, G11]])

    Grho = (1 - eps_dep) * Grho + (eps_dep / 8.0) * np.eye(8, dtype=complex)
    return _herm(Grho)

def G_dagger_eps(X8: np.ndarray, K0: np.ndarray, K1: np.ndarray, eps_dep: float) -> np.ndarray:
    X00 = X8[0:4, 0:4]
    X01 = X8[0:4, 4:8]
    X10 = X8[4:8, 0:4]
    X11 = X8[4:8, 4:8]

    X = (K0.conj().T @ X00 @ K0 +
         K0.conj().T @ X01 @ K1 +
         K1.conj().T @ X10 @ K0 +
         K1.conj().T @ X11 @ K1)

    return _herm((1 - eps_dep) * X)

# ============================================================
# Objective + gradients (UNCHANGED)
# ============================================================

def objective_and_gradient(
    Grho: np.ndarray, Zsigma: np.ndarray, beta: float, K0: np.ndarray, K1: np.ndarray,
    eps_eig: float, eps_dep: float,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    if abs(beta - 1.0) > 1e-15:
        mu = (1.0 - beta) / (2.0 * beta)

        ds_vals, Us = np.linalg.eigh(Zsigma)
        ds = np.maximum(np.real(ds_vals), eps_eig)

        Zsigma_mu = _herm(Us @ np.diag(ds ** mu) @ Us.conj().T)
        Xi = _herm(Zsigma_mu @ Grho @ Zsigma_mu)

        dx_vals, Ux = np.linalg.eigh(Xi)
        dx = np.maximum(np.real(dx_vals), eps_eig)

        Q_beta = float(np.sum(dx ** beta))
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
        grad_sigma = _herm((1.0 / (beta - 1.0)) * (chi1 + chi3) / (Q_beta * np.log(2.0)))

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

def objective_only(
    rho: np.ndarray, sigma: np.ndarray, beta: float, K0: np.ndarray, K1: np.ndarray,
    eps_eig: float, eps_dep: float,
) -> float:
    Grho = build_Grho_eps(rho, K0, K1, eps_dep)

    if abs(beta - 1.0) > 1e-15:
        Zsigma = build_Zsigma(sigma)
        mu = (1.0 - beta) / (2.0 * beta)

        ds_vals, Us = np.linalg.eigh(Zsigma)
        ds = np.maximum(np.real(ds_vals), eps_eig)

        Zsigma_mu = _herm(Us @ np.diag(ds ** mu) @ Us.conj().T)
        Xi = _herm(Zsigma_mu @ Grho @ Zsigma_mu)

        dx_vals, _ = np.linalg.eigh(Xi)
        dx = np.maximum(np.real(dx_vals), eps_eig)

        Q_beta = float(np.sum(dx ** beta))
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

# ============================================================
# CVXPY solve & Feasible Init (UNCHANGED)
# ============================================================

def _solve(prob: cp.Problem, solver: str, scs_opts: Optional[Dict[str, Any]] = None) -> None:
    scs_opts = scs_opts or {}
    if solver.upper() == "MOSEK":
        try:
            prob.solve(solver=cp.MOSEK, verbose=False)
            return
        except Exception:
            pass
    prob.solve(solver=cp.SCS, verbose=False, **scs_opts)

def find_feasible_rho(PiZerr: np.ndarray, PiXerr: np.ndarray, p: float, solver: str = "SCS") -> np.ndarray:
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

def find_initial_state(beta: float, PiZerr: np.ndarray, PiXerr: np.ndarray, p: float) -> Tuple[np.ndarray, np.ndarray]:
    rho = find_feasible_rho(PiZerr, PiXerr, p)
    sigma = np.zeros((8, 8), dtype=complex) if abs(beta - 1.0) < 1e-15 else find_feasible_sigma()
    return rho, sigma

# ============================================================
# Linear oracle (SDP descent) & Frank-Wolfe (UNCHANGED)
# ============================================================

def sdp_descent(
    grad_rho: np.ndarray, grad_sigma: np.ndarray, PiZerr: np.ndarray, PiXerr: np.ndarray,
    p: float, beta: float, solver: str = "MOSEK",
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

        prob = cp.Problem(cp.Minimize(cp.real(cp.trace(grad_rho @ rho_var))), constraints)
        _solve(prob, solver=solver, scs_opts={"eps": 1e-7, "max_iters": 200000})
        return _herm(rho_var.value), np.zeros((8, 8), dtype=complex)

    rho_var = cp.Variable((4, 4), hermitian=True)
    sigma_var = cp.Variable((8, 8), hermitian=True)

    constraints = [
        rho_var >> 0, sigma_var >> 0,
        cp.trace(rho_var) == 1, cp.trace(sigma_var) == 1,
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

    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(grad_rho @ rho_var) + cp.trace(grad_sigma @ sigma_var))), constraints)
    _solve(prob, solver=solver, scs_opts={"eps": 1e-7, "max_iters": 200000})
    return _herm(rho_var.value), _herm(sigma_var.value)


@dataclass
class FWOptions:
    maxIter: int = 500         
    tol_gap: float = 1e-6
    eps_dep: float = 1e-6      
    eps_eig: float = 1e-12
    oracle_solver: str = "MOSEK"
    use_step2: bool = True
    verbose: bool = False

def step_one_FW(
    rho: np.ndarray, sigma: np.ndarray, K0: np.ndarray, K1: np.ndarray, beta: float,
    PiZerr: np.ndarray, PiXerr: np.ndarray, p: float, opts: FWOptions,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    for it in range(opts.maxIter):
        Grho = build_Grho_eps(rho, K0, K1, opts.eps_dep)
        Zsigma = build_Zsigma(sigma) if abs(beta - 1.0) > 1e-15 else np.zeros((8, 8), dtype=complex)

        f_val, grad_full, grad_rho, grad_sigma = objective_and_gradient(
            Grho, Zsigma, beta, K0, K1, opts.eps_eig, opts.eps_dep
        )

        rho_f, sigma_f = sdp_descent(
            grad_rho, grad_sigma, PiZerr, PiXerr, p, beta, solver=opts.oracle_solver
        )

        gap = float(abs(np.real(np.trace((rho - rho_f) @ grad_rho) + np.trace((sigma - sigma_f) @ grad_sigma))))

        def obj_fun(lam: float) -> float:
            rho_l = _herm((1 - lam) * rho + lam * rho_f)
            sigma_l = _herm((1 - lam) * sigma + lam * sigma_f)
            return objective_only(rho_l, sigma_l, beta, K0, K1, opts.eps_eig, opts.eps_dep)

        res = minimize_scalar(obj_fun, bounds=(0.0, 1.0), method="bounded")
        lam = float(res.x)

        rho = _herm((1 - lam) * rho + lam * rho_f)
        sigma = _herm((1 - lam) * sigma + lam * sigma_f)

        if gap < opts.tol_gap:
            break

    rho_hat, sigma_hat = rho, sigma
    M_hat = _blkdiag(rho_hat, sigma_hat)

    Grho_hat = build_Grho_eps(rho_hat, K0, K1, opts.eps_dep)
    Zsigma_hat = build_Zsigma(sigma_hat) if abs(beta - 1.0) > 1e-15 else np.zeros((8, 8), dtype=complex)

    f_hat, grad_full_hat, grad_rho_hat, grad_sigma_hat = objective_and_gradient(
        Grho_hat, Zsigma_hat, beta, K0, K1, opts.eps_eig, opts.eps_dep
    )
    return M_hat, float(f_hat), grad_full_hat, grad_rho_hat, grad_sigma_hat

def step_two_FW(
    f_hat: float, M_hat: np.ndarray, grad_rho_hat: np.ndarray, grad_sigma_hat: np.ndarray,
    grad_full_hat: np.ndarray, PiZerr: np.ndarray, PiXerr: np.ndarray, p: float,
    beta: float, opts: FWOptions,
) -> float:
    first_term = float(np.real(f_hat))
    second_term = -float(np.real(np.trace(grad_full_hat @ M_hat)))

    rho_opt, sigma_opt = sdp_descent(
        grad_rho_hat, grad_sigma_hat, PiZerr, PiXerr, p, beta, solver=opts.oracle_solver
    )
    M_opt = _blkdiag(rho_opt, sigma_opt)
    third_term = float(np.real(np.trace(grad_full_hat @ M_opt)))

    return float(np.real(first_term + second_term + third_term))

def frank_wolf(
    beta: float, p: float, q: float, opts: FWOptions, init_state: Optional[Tuple[np.ndarray, np.ndarray]] = None
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
            f_hat, M_hat, grad_rho_hat, grad_sigma_hat, grad_full_hat,
            PiZerr, PiXerr, p, beta, opts
        )
        return val, final_state
    
    return float(f_hat), final_state


def rate_raw_fs(
    beta: float, p: float, q: float, alpha: float, N: int, opts: FWOptions, 
    eps_PA: float, init_state: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    lb, final_state = frank_wolf(beta, p, q, opts, init_state=init_state)
    leak = h2(p + q - 2 * p * q)
    g_alpha = np.log2(1.0 / eps_PA) * alpha / (alpha - 1.0) - 2.0
    return float(lb - leak - g_alpha / float(N)), final_state

def rate_raw_asym(
    beta: float, p: float, q: float, alpha: float, opts: FWOptions, 
    eps_PA: float, init_state: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    lb, final_state = frank_wolf(beta, p, q, opts, init_state=init_state)
    leak = h2(p + q - 2 * p * q)
    return float(lb - leak), final_state


class StateCache:
    def __init__(self):
        self.last_state = None

def beta_from_alpha(alpha: float) -> float:
    return float(alpha / (2.0 * alpha - 1.0))

def optimize_q(
    objective_func, q_lo: float = 0.0, q_hi: float = 0.5,
) -> Tuple[float, float]:
    cache = StateCache()

    def neg_obj(q):
        val, new_state = objective_func(q, cache.last_state)
        cache.last_state = new_state
        return -val

    res = minimize_scalar(neg_obj, bounds=(q_lo, q_hi), method="bounded", options={"xatol": 1e-4, "maxiter": 40})
    q_star = float(res.x)
    raw_star = float(-res.fun)
    return q_star, raw_star

# ============================================================
# MAIN (UPDATED FOR HTCONDOR ARGS)
# ============================================================

def main() -> None:
    # Ensure arguments are passed
    if len(sys.argv) < 3:
        print("Usage: python main.py <p_value> <alpha_value>")
        sys.exit(1)

    N = int(sys.argv[1])
    p = float(sys.argv[2])
    alpha = float(sys.argv[3])
    conf = config()
    q_lo = conf['q_lo']
    q_hi = conf['q_hi']
    opts_FW = conf['opts_FW']
    alpha_corr_at_1 = conf['alpha_corr_at_1']
    eps_PA = conf['eps_PA']
    phase1 = conf['phase1']

    # Make sure output directory exists
    os.makedirs("results", exist_ok=True)
    out_grid = f"results/results_N{N}__p{p:.3f}_a{alpha:.4f}.csv"
    rows_grid = []
    
    beta = beta_from_alpha(alpha)
    alpha_g = alpha_corr_at_1 if abs(alpha - 1.0) < 1e-15 else alpha

    if phase1:
        print(f"=== Asym BB84 job | p={p:.3f}, alpha={alpha:.4f} ===", flush=True)

        # Wrapper for R0 (q=0)
        raw0, _ = rate_raw_asym(beta, p, 0.0, alpha_g, opts_FW, eps_PA, init_state=None)
        R0 = max(0.0, raw0)

        # Wrapper for R* (optimize q)
        def raw_obj_wrapper(q, current_state):
            return rate_raw_asym(beta, p, q, alpha_g, opts_FW, eps_PA, init_state=current_state)

        q_star, raw_star = optimize_q(raw_obj_wrapper, q_lo=q_lo, q_hi=q_hi)
        R_star = max(0.0, raw_star)

        print(f"Result: q*={q_star:.6f} | R*={R_star:.6e}", flush=True)
        N = 1
        rows_grid.append((int(N), float(p), float(q_star), float(R_star), float(R0), float(alpha)))

    else:
        print(f"=== Finite-size BB84 job | p={p:.3f}, alpha={alpha:.4f} ===", flush=True)

        raw0, _ = rate_raw_fs(beta, p, 0.0, alpha_g, N, opts_FW, eps_PA, init_state=None)
        R0 = max(0.0, raw0)

        def raw_obj_wrapper(q, current_state):
            return rate_raw_fs(beta, p, q, alpha_g, N, opts_FW, eps_PA, init_state=current_state)

        q_star, raw_star = optimize_q(raw_obj_wrapper, q_lo=q_lo, q_hi=q_hi)
        R_star = max(0.0, raw_star)

        print(f"N={N:.1e}: q*={q_star:.6f} | R*={R_star:.6e}", flush=True)
        rows_grid.append((int(N), float(p), float(q_star), float(R_star), float(R0), float(alpha)))

    # Save unique CSV for this specific job
    with open(out_grid, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "p", "Q_star", "R_star", "R_0", "alpha"])
        for r in rows_grid:
            formatted_row = list(r)
            formatted_row[0] = f"{r[0]:.1e}" 
            w.writerow(formatted_row)

    print(f"Done. Saved to {out_grid}", flush=True)

if __name__ == "__main__":
    main()
