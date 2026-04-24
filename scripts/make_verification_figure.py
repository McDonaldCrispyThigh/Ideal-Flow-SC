import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.special import ellipk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.sc_solver import integrate_real


def rectangle_run():
    k = 1.0 / np.sqrt(2.0)
    s_target = 2.0 * k / (1.0 + k * k)
    K_val = float(ellipk(k * k))

    zk_fixed_left = -1.0
    zk_fixed_mid = 0.0
    zk_fixed_right = 1.0
    alphas = np.array([0.5, 0.5, 0.5, 0.5])
    betas = alphas - 1.0

    Kp_val = K_val
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

    cost_history = []
    s_history = []

    def _softmax(p):
        e = np.exp(p)
        return e[0] / (e.sum() + 1.0)

    def residuals(p):
        s_val = _softmax(p)
        zk = np.array([zk_fixed_left, zk_fixed_mid, s_val, zk_fixed_right])
        L = np.array([
            abs(integrate_real(zk[0], zk[1], zk, betas)),
            abs(integrate_real(zk[1], zk[2], zk, betas)),
            abs(integrate_real(zk[2], zk[3], zk, betas)),
        ])
        L_ratios = L / L[-1]
        r = L_ratios - target_ratios
        cost_history.append(0.5 * float(np.dot(r, r)))
        s_history.append(s_val)
        return r

    p0 = np.array([0.5])
    sol = least_squares(residuals, p0, method="lm", max_nfev=2000,
                         ftol=1e-15, xtol=1e-15, gtol=1e-15)
    s_recovered = _softmax(sol.x)
    return {
        "k": k,
        "s_target": s_target,
        "s_recovered": float(s_recovered),
        "K_val": K_val,
        "cost_history": np.array(cost_history),
        "s_history": np.array(s_history),
        "final_residual_norm": float(np.linalg.norm(sol.fun)),
    }


def plot_figure_A(result, out_path_png, out_path_pdf):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax_a, ax_b = axes

    s_target = result["s_target"]
    K_val = result["K_val"]

    pre_vertices = np.array([-1.0, 0.0, s_target, 1.0])
    labels = [r"$\zeta_0$", r"$\zeta_1$", r"$\zeta_2 = s$", r"$\zeta_3$"]
    fixed_mask = [True, True, False, True]

    for xv, lab, fx in zip(pre_vertices, labels, fixed_mask):
        color = "#d62728" if not fx else "#1f77b4"
        marker = "o" if fx else "s"
        ax_a.plot(xv, 0, marker=marker, color=color, markersize=10,
                  markeredgecolor="black", markeredgewidth=0.8, zorder=3)
        ax_a.annotate(lab, xy=(xv, 0), xytext=(xv, -0.22),
                      ha="center", fontsize=10)

    ax_a.axhspan(0.0, 1.0, color="#f0f4ff", alpha=0.6, zorder=0)
    ax_a.axhline(0, color="black", linewidth=1.0, zorder=1)

    for eta0 in np.linspace(0.05, 0.9, 7):
        xs = np.linspace(-1.3, 1.3, 60)
        ax_a.plot(xs, np.full_like(xs, eta0), color="#4a90e2",
                  linewidth=0.7, alpha=0.55, zorder=2)
    for xi0 in np.linspace(-1.1, 1.1, 9):
        ys = np.linspace(0.0, 0.98, 40)
        ax_a.plot(np.full_like(ys, xi0), ys, color="#e07a5f",
                  linewidth=0.7, alpha=0.55, zorder=2)

    ax_a.set_xlim(-1.35, 1.35)
    ax_a.set_ylim(-0.35, 1.05)
    ax_a.set_xlabel(r"$\xi = \mathrm{Re}\,\zeta$", fontsize=11)
    ax_a.set_ylabel(r"$\eta = \mathrm{Im}\,\zeta$", fontsize=11)
    ax_a.set_title(
        r"(a) Upper half-plane $\mathbb{H}$: pre-vertices on $\mathbb{R}$, "
        r"rectangular grid overlay",
        fontsize=11)
    ax_a.text(-1.3, 0.92, "blue squares: fixed anchors $\{-1, 0, 1\}$",
              fontsize=8, color="#1f77b4")
    ax_a.text(-1.3, 0.82, "red circle: free pre-vertex $s = 2\sqrt{2}/3$",
              fontsize=8, color="#d62728")

    ax_a.set_aspect("auto")

    iters = np.arange(1, len(result["cost_history"]) + 1)
    cost = result["cost_history"]
    s_err = np.abs(result["s_history"] - result["s_target"])

    ax_b.semilogy(iters, np.maximum(cost, 1e-16), "-o",
                  color="#1f77b4", markersize=3.5,
                  label=r"Side-ratio residual cost $\frac{1}{2}\,\|r\|_2^2$")
    ax_b.semilogy(iters, np.maximum(s_err, 1e-16), "-s",
                  color="#d62728", markersize=3.5,
                  label=r"$|s_{\mathrm{computed}} - s_{\mathrm{exact}}|$")
    ax_b.axhline(1e-8, color="gray", linestyle="--", linewidth=0.8,
                 label=r"LM tolerance $10^{-8}$")
    ax_b.set_ylim(1e-16, 1e-1)
    ax_b.set_xlabel("LM function evaluation", fontsize=11)
    ax_b.set_ylabel("Error magnitude (log scale)", fontsize=11)
    ax_b.set_title("(b) LM convergence on the rectangle verification",
                   fontsize=11)
    ax_b.grid(True, which="both", alpha=0.25)
    ax_b.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    png_path = os.path.join(figures_dir, "figA_rectangle_verification.png")
    pdf_path = os.path.join(figures_dir, "figA_rectangle_verification.pdf")

    print("Running rectangle verification...")
    result = rectangle_run()
    print(f"  target k = {result['k']:.6f}")
    print(f"  target s = {result['s_target']:.8f}")
    print(f"  recovered s = {result['s_recovered']:.8f}")
    print(f"  |s_recovered - s_target| = {abs(result['s_recovered'] - result['s_target']):.2e}")
    print(f"  K(k) = {result['K_val']:.8f}")
    print(f"  LM evaluations logged: {len(result['cost_history'])}")
    print(f"  final residual norm: {result['final_residual_norm']:.2e}")

    plot_figure_A(result, png_path, pdf_path)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
