from typing import Optional, Tuple

import numpy as np
from trusted_noise_qkd.frank_wolfe.frank_wolfe import frank_wolfe
from trusted_noise_qkd.frank_wolfe.FWOptions import FWOptions
from trusted_noise_qkd.utils.h2 import h2


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
