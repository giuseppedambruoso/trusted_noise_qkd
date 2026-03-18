import numpy as np


def spectral_kernel(
    A: np.ndarray, U: np.ndarray, d: np.ndarray, mu: float
) -> np.ndarray:
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
                    M[i, j] = (di**mu - dj**mu) / denom

    return U @ (M * Atil) @ U.conj().T
