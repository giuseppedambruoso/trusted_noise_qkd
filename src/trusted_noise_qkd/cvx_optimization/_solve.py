from typing import Any, Dict, Optional

import cvxpy as cp


def _solve(
    prob: cp.Problem, solver: str, scs_opts: Optional[Dict[str, Any]] = None
) -> None:
    scs_opts = scs_opts or {}
    if solver.upper() == "MOSEK":
        try:
            prob.solve(solver=cp.MOSEK, verbose=False)
            return
        except Exception:
            pass
    prob.solve(solver=cp.SCS, verbose=False, **scs_opts)
