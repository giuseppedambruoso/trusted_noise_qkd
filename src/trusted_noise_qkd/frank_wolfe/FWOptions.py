from dataclasses import dataclass


@dataclass
class FWOptions:
    maxIter: int = 500
    tol_gap: float = 1e-6
    eps_dep: float = 1e-6
    eps_eig: float = 1e-12
    oracle_solver: str = "MOSEK"
    use_step2: bool = True
    verbose: bool = False
