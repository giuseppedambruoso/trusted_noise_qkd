from typing import Tuple

from scipy.optimize import minimize_scalar


class StateCache:
    def __init__(self):
        self.last_state = None


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
