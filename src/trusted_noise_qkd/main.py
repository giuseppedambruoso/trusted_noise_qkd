# main.py
from __future__ import annotations

import csv
import os
import sys

from trusted_noise_qkd.config.config import config
from trusted_noise_qkd.key_rate.rate_raw_asym import rate_raw_asym
from trusted_noise_qkd.key_rate.rate_raw_fs import rate_raw_fs
from trusted_noise_qkd.utils.beta_from_alpha import beta_from_alpha
from trusted_noise_qkd.utils.optimize_q import optimize_q


def main() -> None:
    # Ensure arguments are passed
    if len(sys.argv) < 4:
        print("Usage: python main.py <p_value> <alpha_value>")
        sys.exit(1)

    N = int(sys.argv[1])
    p = float(sys.argv[2])
    alpha = float(sys.argv[3])
    conf = config()
    q_lo = conf["q_lo"]
    q_hi = conf["q_hi"]
    opts_FW = conf["opts_FW"]
    alpha_corr_at_1 = conf["alpha_corr_at_1"]
    eps_PA = conf["eps_PA"]
    phase1 = conf["phase1"]

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
            return rate_raw_asym(
                beta, p, q, alpha_g, opts_FW, eps_PA, init_state=current_state
            )

        q_star, raw_star = optimize_q(raw_obj_wrapper, q_lo=q_lo, q_hi=q_hi)
        R_star = max(0.0, raw_star)

        print(f"Result: q*={q_star:.6f} | R*={R_star:.6e}", flush=True)
        N = 1
        rows_grid.append(
            (int(N), float(p), float(q_star), float(R_star), float(R0), float(alpha))
        )

    else:
        print(
            f"=== Finite-size BB84 job | p={p:.3f}, alpha={alpha:.4f} ===", flush=True
        )

        raw0, _ = rate_raw_fs(
            beta, p, 0.0, alpha_g, N, opts_FW, eps_PA, init_state=None
        )
        R0 = max(0.0, raw0)

        def raw_obj_wrapper(q, current_state):
            return rate_raw_fs(
                beta, p, q, alpha_g, N, opts_FW, eps_PA, init_state=current_state
            )

        q_star, raw_star = optimize_q(raw_obj_wrapper, q_lo=q_lo, q_hi=q_hi)
        R_star = max(0.0, raw_star)

        print(f"N={N:.1e}: q*={q_star:.6f} | R*={R_star:.6e}", flush=True)
        rows_grid.append(
            (int(N), float(p), float(q_star), float(R_star), float(R0), float(alpha))
        )

    # Save unique CSV for this specific job
    with open(out_grid, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "p", "Q_star", "R_star", "R_0", "alpha"])
        for r in rows_grid:
            # Create a new list: [formatted_string] + [the rest of the numbers]
            formatted_row = [f"{r[0]:.1e}"] + list(r[1:])
            w.writerow(formatted_row)

    print(f"Done. Saved to {out_grid}", flush=True)


if __name__ == "__main__":
    main()
