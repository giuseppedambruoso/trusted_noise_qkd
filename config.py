import numpy as np
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

def config():
    p_list = np.linspace(0.12,0.123,10).tolist()
    alpha_list = np.linspace(1.001, 1.01, 10).tolist()
    N_list = [10**6,10**7,10**8,10**9,10**10,10**11]

    alpha_corr_at_1 = 1 + 10**-10
    eps_PA = 1e-10

    return {
        'alpha_list': alpha_list,
        'p_list': p_list,
        'N_list': N_list,
        'q_lo': 0.0,
        'q_hi': 0.5,
        'opts_fmin': {'xatol': 1e-4, "maxiter": 60},
        'opts_FW': FWOptions(
            maxIter=500,
            tol_gap=1e-6,
            eps_eig=1e-12,
            eps_dep=1e-9,
            oracle_solver="MOSEK"
        ),
        'alpha_corr_at_1' : alpha_corr_at_1,
        'eps_PA' : 1e-10,
        'phase1' : False
    }
