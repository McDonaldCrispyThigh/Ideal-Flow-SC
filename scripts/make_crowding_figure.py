import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, brentq
from scipy.special import ellipk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.sc_solver import integrate_real


def k_for_aspect(L):
    def f(k):
        Kk = float(ellipk(k * k))
        Kp = float(ellipk(1.0 - k * k))
        return 2.0 * Kk / Kp - L
    return brentq(f, 1e-8, 1 - 1e-14, xtol=1e-14)


def solve_rectangle(L):
    k = k_for_aspect(L)
    s_target = 2.0 * k / (1.0 + k * k)
    K_val = float(ellipk(k * k))
    Kp_val = float(ellipk(1.0 - k * k))

    alphas = np.array([0.5, 0.5, 0.5, 0.5])
    betas = alphas - 1.0

    w = np.array([
        complex(-K_val, Kp_val),
        complex(-K_val, 0.0),
        complex(K_val, 0.0),
        complex(K_val, Kp_val),
    ])
    target_ratios = np.array([
        abs(w[1] - w[0]),
        abs(w[2] - w[1]),
        abs(w[3] - w[2]),
    ])
    target_ratios = target_ratios / target_ratios[-1]

    def _softmax(p):
        e = np.exp(p)
        return e[0] / (e.sum() + 1.0)

    def residuals(p):
        s_val = _softmax(p)
        zk = np.array([-1.0, 0.0, s_val, 1.0])
        vals = np.array([
            abs(integrate_real(zk[0], zk[1], zk, betas)),
            abs(integrate_real(zk[1], zk[2], zk, betas)),
            abs(integrate_real(zk[2], zk[3], zk, betas)),
        ])
        ratios = vals / vals[-1]
        return ratios - target_ratios

    p0_guess = np.log(s_target / (1.0 - s_target)) if s_target < 0.999 else 5.0
    sol = least_squares(residuals, np.array([p0_guess]), method="lm",
                         max_nfev=500, ftol=1e-15, xtol=1e-15, gtol=1e-15)
    s_recovered = _softmax(sol.x)
    return {
        "L": L,
        "k": k,
        "s_target": s_target,
        "s_recovered": float(s_recovered),
        "spacing_target": 1.0 - s_target,
        "spacing_recovered": 1.0 - float(s_recovered),
    }


def plot_figure_B(results, out_png, out_pdf):
    L_arr = np.array([r["L"] for r in results])
    spacing_target = np.array([r["spacing_target"] for r in results])
    spacing_recovered = np.array([r["spacing_recovered"] for r in results])

    L_dense = np.linspace(1.0, 12.0, 200)
    k_dense = np.array([k_for_aspect(L) for L in L_dense])
    s_dense = 2.0 * k_dense / (1.0 + k_dense * k_dense)
    theory_inner_spacing = 1.0 - s_dense

    classical_asym = 8.0 * np.exp(-np.pi * L_dense / 2.0)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    ax.semilogy(L_dense, theory_inner_spacing, "-",
                color="#1f77b4", linewidth=2.0,
                label=r"Exact inner spacing $1 - s$ in pipeline normalization")
    ax.semilogy(L_dense, classical_asym, "--",
                color="#888888", linewidth=1.3,
                label=r"Classical asymptotic $8\,e^{-\pi L/2}$ "
                      r"(pre-vertex $1/k - 1$)")
    ax.semilogy(L_arr, spacing_recovered, "o",
                color="#d62728", markersize=9, zorder=5,
                markeredgecolor="black", markeredgewidth=0.8,
                label="SC solver recovered $1 - s$")

    ax.axhline(2.2e-16, color="#5e5e5e", linestyle=":", linewidth=1.0)
    ax.text(11.7, 2.2e-16 * 3, "double precision limit",
            fontsize=8, color="#5e5e5e", ha="right")

    ax.axhline(1e-8, color="#c07020", linestyle=":", linewidth=1.0)
    ax.text(11.7, 1e-8 * 3, r"$10^{-8}$ reliable-recovery threshold",
            fontsize=8, color="#c07020", ha="right")

    ax.set_xlabel(r"Rectangle aspect ratio $L = 2K(k) / K'(k)$", fontsize=11)
    ax.set_ylabel(r"Inner pre-vertex spacing (log scale)", fontsize=11)
    ax.set_title(
        r"SC crowding: pre-vertex spacing decays exponentially with conformal modulus",
        fontsize=11)
    ax.set_xlim(1.0, 12.0)
    ax.set_ylim(1e-20, 2.0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    png_path = os.path.join(figures_dir, "figB_sc_crowding.png")
    pdf_path = os.path.join(figures_dir, "figB_sc_crowding.pdf")

    L_list = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]
    results = []
    print("Running SC solver for each aspect ratio:")
    print(f"{'L':>6} {'k':>10} {'s_target':>12} {'s_recov':>12} "
          f"{'spacing_target':>15} {'spacing_recov':>15}")
    for L in L_list:
        r = solve_rectangle(L)
        results.append(r)
        print(f"{r['L']:>6.2f} {r['k']:>10.6f} "
              f"{r['s_target']:>12.8f} {r['s_recovered']:>12.8f} "
              f"{r['spacing_target']:>15.3e} {r['spacing_recovered']:>15.3e}")

    plot_figure_B(results, png_path, pdf_path)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
