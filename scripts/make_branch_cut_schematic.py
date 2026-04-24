"""Figure G: Branch-cut integration path schematic in the upper half-plane."""
import os
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_schematic(out_png, out_pdf):
    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    ax.axhspan(0.0, 1.6, color="#eef4ff", alpha=0.5, zorder=0)
    ax.axhline(0, color="black", linewidth=1.2, zorder=3)

    pre_vertices = np.array([-1.0, -0.55, -0.2, 0.15, 0.4, 0.72, 1.0])
    for xv in pre_vertices:
        ax.plot(xv, 0.0, "o", color="#d62728", markersize=8,
                markeredgecolor="black", markeredgewidth=0.7, zorder=5)

    for xv in pre_vertices:
        ax.plot([xv, xv], [0.0, -0.22], color="#d62728",
                linewidth=1.6, zorder=4)
        ax.plot([xv - 0.04, xv + 0.04], [-0.22, -0.22],
                color="#d62728", linewidth=1.6, zorder=4)

    zeta_ref = 0.0 + 0.5j
    target = 0.85 + 0.12j
    ax.plot(zeta_ref.real, zeta_ref.imag, "s",
            color="#1f77b4", markersize=9,
            markeredgecolor="black", markeredgewidth=0.8, zorder=6)
    ax.annotate(r"$\zeta_{\mathrm{ref}} = 0.5\,i$",
                xy=(zeta_ref.real, zeta_ref.imag),
                xytext=(zeta_ref.real - 0.5, zeta_ref.imag + 0.25),
                fontsize=10, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4",
                                shrinkA=4, shrinkB=4))

    ax.plot(target.real, target.imag, "*",
            color="#2ca02c", markersize=14,
            markeredgecolor="black", markeredgewidth=0.6, zorder=6)
    ax.annotate(r"target $\zeta$",
                xy=(target.real, target.imag),
                xytext=(target.real + 0.05, target.imag + 0.3),
                fontsize=10, color="#2ca02c",
                arrowprops=dict(arrowstyle="->", color="#2ca02c",
                                shrinkA=4, shrinkB=4))

    p1 = zeta_ref
    p2 = target.real + 1j * 0.5
    p3 = target
    ax.annotate("", xy=(p2.real, p2.imag), xytext=(p1.real, p1.imag),
                arrowprops=dict(arrowstyle="->", color="#4a90e2",
                                linewidth=2.2, mutation_scale=17))
    ax.annotate("", xy=(p3.real, p3.imag), xytext=(p2.real, p2.imag),
                arrowprops=dict(arrowstyle="->", color="#4a90e2",
                                linewidth=2.2, mutation_scale=17))
    ax.text(0.42, 0.58, "horizontal leg",
            fontsize=9, color="#2e5aa8")
    ax.text(0.87, 0.32, "vertical\nleg",
            fontsize=9, color="#2e5aa8")

    ax.text(-1.5, 0.15, r"$\mathbb{H}$ (upper half-plane)",
            fontsize=10, color="#334")
    ax.text(-1.5, -0.4, r"branch cuts below $\mathbb{R}$ at each $\zeta_k$",
            fontsize=9, color="#d62728")
    ax.text(-1.5, -0.58, r"integrand $\prod_k (t-\zeta_k)^{\alpha_k-1}$ "
                          r"is multi-valued on $\mathbb{R}$",
            fontsize=8, color="#888")

    for i, xv in enumerate(pre_vertices):
        ax.text(xv, 0.06, f"$\\zeta_{{{i}}}$",
                fontsize=8, color="#d62728", ha="center")

    ax.set_xlim(-1.6, 1.5)
    ax.set_ylim(-0.85, 1.35)
    ax.set_aspect("auto")
    ax.set_xlabel(r"$\xi = \mathrm{Re}\,\zeta$", fontsize=11)
    ax.set_ylabel(r"$\eta = \mathrm{Im}\,\zeta$", fontsize=11)
    ax.set_title("", fontsize=11)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    png = os.path.join(figures_dir, "figG_branch_cut_path.png")
    pdf = os.path.join(figures_dir, "figG_branch_cut_path.pdf")
    plot_schematic(png, pdf)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
