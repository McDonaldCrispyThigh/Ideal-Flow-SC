"""Figure E: Milne-Thomson image mechanism schematic (no numerical data)."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_schematic(out_png, out_pdf):
    fig, ax = plt.subplots(figsize=(8.4, 5.6))

    zeta0 = (0.6, 1.4)
    zeta0_bar = (0.6, -1.4)
    a = 0.45

    ax.axhline(0, color="black", linewidth=1.2, zorder=3)
    ax.axhspan(0.0, 2.5, color="#eef4ff", alpha=0.5, zorder=0)
    ax.axhspan(-2.5, 0.0, color="#f5f0ff", alpha=0.5, zorder=0)
    ax.text(-2.3, 2.25, r"Upper half-plane $\mathbb{H}$",
            fontsize=10, color="#334")
    ax.text(-2.3, -2.35, r"Image plane (below $\mathbb{R}$)",
            fontsize=10, color="#443355")

    circle_true = Circle(zeta0, a, fill=False, edgecolor="#d62728",
                         linewidth=1.8, zorder=4)
    circle_img = Circle(zeta0_bar, a, fill=False, edgecolor="#1f77b4",
                        linewidth=1.3, linestyle="--", zorder=4)
    ax.add_patch(circle_true)
    ax.add_patch(circle_img)

    ax.plot(*zeta0, "o", color="#d62728", markersize=9,
            markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    ax.plot(*zeta0_bar, "s", color="#1f77b4", markersize=9,
            markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    ax.annotate(r"$\zeta_0$, obstacle centre",
                xy=zeta0, xytext=(zeta0[0] + 0.65, zeta0[1] + 0.2),
                fontsize=10, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728",
                                shrinkA=3, shrinkB=3))
    ax.annotate(r"$\bar{\zeta}_0$, mirror image",
                xy=zeta0_bar, xytext=(zeta0_bar[0] + 0.65, zeta0_bar[1] - 0.2),
                fontsize=10, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4",
                                shrinkA=3, shrinkB=3))

    ax.annotate("", xy=(zeta0[0] + 0.28, zeta0[1]),
                xytext=(zeta0[0] - 0.28, zeta0[1]),
                arrowprops=dict(arrowstyle="->", color="#d62728",
                                linewidth=1.8, mutation_scale=15))
    ax.text(zeta0[0], zeta0[1] - 0.15, "dipole",
            fontsize=8, color="#d62728", ha="center")

    ax.annotate("", xy=(zeta0_bar[0] + 0.28, zeta0_bar[1]),
                xytext=(zeta0_bar[0] - 0.28, zeta0_bar[1]),
                arrowprops=dict(arrowstyle="->", color="#1f77b4",
                                linewidth=1.4, mutation_scale=15))
    ax.text(zeta0_bar[0], zeta0_bar[1] - 0.15, "image dipole",
            fontsize=8, color="#1f77b4", ha="center")

    t = np.linspace(-2.2, 2.2, 400)
    for y0 in [0.35, 0.7, 1.0, 1.8, 2.15]:
        x = t
        y = y0 + 0.22 * np.exp(-((x - zeta0[0]) ** 2) / 0.25) * np.sign(y0 - zeta0[1])
        inside = np.sqrt((x - zeta0[0]) ** 2 + (y - zeta0[1]) ** 2) > a + 0.03
        x_plot = np.where(inside, x, np.nan)
        y_plot = np.where(inside, y, np.nan)
        ax.plot(x_plot, y_plot, color="#6b8e23", linewidth=1.1,
                alpha=0.7, zorder=2)

    ax.text(0.6, 0.05, r"$\psi = 0$ on $\partial\mathbb{H}$ by image sum",
            fontsize=9, color="#444", ha="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

    ax.text(zeta0[0] + a + 0.05, zeta0[1] + a - 0.1,
            r"$|\zeta - \zeta_0| = a$",
            fontsize=9, color="#d62728")

    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}\,\zeta$", fontsize=11)
    ax.set_ylabel(r"$\mathrm{Im}\,\zeta$", fontsize=11)
    ax.set_title("Milne-Thomson image construction: obstacle + real-axis mirror",
                 fontsize=11)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    png = os.path.join(figures_dir, "figE_milne_thomson_schematic.png")
    pdf = os.path.join(figures_dir, "figE_milne_thomson_schematic.pdf")
    plot_schematic(png, pdf)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
