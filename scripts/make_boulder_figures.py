"""Generates Boulder-specific figures using the pickled cache:
  Figure C: pre-vertex layout on R paired with Boulder polygon vertices
  Figure D: conformal grid (rectangular mesh in H pushed through SC map)
  Figure F: velocity magnitude heatmap for uniform + urban potentials"""
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.sc_solver import (
    SCParameters, sc_map, integrate_complex,
)


def load_cache():
    path = os.path.join(PROJECT_ROOT, "scripts", "boulder_cache.pkl")
    with open(path, "rb") as f:
        c = pickle.load(f)
    params = SCParameters(
        zk=c["params_zk"],
        alphas=c["params_alphas"],
        betas=c["params_betas"],
        A=c["params_A"],
        C=c["params_C"],
        z_poly=c["params_z_poly"],
    )
    return params


def figure_C(params, out_png, out_pdf):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2),
                              gridspec_kw={"width_ratios": [1.0, 1.1]})
    ax_left, ax_right = axes

    zk = params.zk
    alphas = params.alphas
    z_poly = params.z_poly
    n = len(zk)

    cmap = plt.colormaps["tab20"]
    colors = [cmap(i / max(n - 1, 1) * 0.95) for i in range(n)]

    ax_left.axhline(0, color="black", linewidth=1.0, zorder=2)
    ax_left.axhspan(-0.02, 0.02, color="#eef4ff", alpha=0.6, zorder=0)
    for i, (xv, col) in enumerate(zip(zk, colors)):
        ax_left.plot(xv, 0.0, "o", color=col, markersize=11,
                     markeredgecolor="black", markeredgewidth=0.9, zorder=5)
        fixed = i in (0, 1, n - 1)
        label = rf"$\zeta_{{{i}}}$"
        y_offset = -0.045 if i % 2 == 0 else 0.05
        va = "top" if y_offset < 0 else "bottom"
        ax_left.annotate(label, xy=(xv, 0), xytext=(xv, y_offset),
                         ha="center", va=va, fontsize=9, color=col)
        if fixed:
            ax_left.plot(xv, 0.0, marker="s", color="none",
                         markeredgecolor="black", markersize=17,
                         markeredgewidth=1.0, zorder=4)

    ax_left.set_xlim(-1.15, 1.15)
    ax_left.set_ylim(-0.25, 0.25)
    ax_left.set_yticks([])
    ax_left.set_xlabel(r"Real axis $\zeta \in \mathbb{R}$", fontsize=11)
    ax_left.set_title("(a) Pre-vertices on the real axis of $\\mathbb{H}$",
                       fontsize=11)
    ax_left.text(-1.10, 0.18,
                 r"black squares: fixed anchors $\zeta_0, \zeta_1, \zeta_{n-1}$",
                 fontsize=8)
    ax_left.text(-1.10, 0.14,
                 r"filled circles: solved pre-vertices",
                 fontsize=8)
    ax_left.grid(True, alpha=0.2)

    closed_poly = np.append(z_poly, z_poly[0])
    ax_right.plot(closed_poly.real, closed_poly.imag,
                  color="#333", linewidth=1.5, zorder=2)
    ax_right.fill(closed_poly.real, closed_poly.imag,
                  color="#f0f4ff", alpha=0.6, zorder=1)

    for i, (w, col) in enumerate(zip(z_poly, colors)):
        ax_right.plot(w.real, w.imag, "o", color=col, markersize=11,
                      markeredgecolor="black", markeredgewidth=0.9, zorder=5)
        offset = 0.06
        dx = offset if w.real > 0 else -offset
        dy = offset if w.imag > 0 else -offset
        ax_right.annotate(
            rf"$w_{{{i}}}$ ($\alpha = {alphas[i]:.2f}$)",
            xy=(w.real, w.imag),
            xytext=(w.real + dx, w.imag + dy),
            fontsize=8, color=col,
            ha=("left" if dx > 0 else "right"),
            va=("bottom" if dy > 0 else "top"),
        )

    pad = 0.18
    xmin, xmax = z_poly.real.min() - pad, z_poly.real.max() + pad
    ymin, ymax = z_poly.imag.min() - pad, z_poly.imag.max() + pad
    ax_right.set_xlim(xmin, xmax)
    ax_right.set_ylim(ymin, ymax)
    ax_right.set_aspect("equal")
    ax_right.set_xlabel(r"$\mathrm{Re}\,z$ (normalised)", fontsize=11)
    ax_right.set_ylabel(r"$\mathrm{Im}\,z$ (normalised)", fontsize=11)
    ax_right.set_title(
        rf"(b) Boulder polygon vertices $w_k$ with interior angles $\alpha_k \pi$",
        fontsize=11)
    ax_right.grid(True, alpha=0.2)

    fig.suptitle(
        "Pre-vertex $\\zeta_k$ on $\\mathbb{R}$ and polygon vertex $w_k$ correspondence",
        fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def figure_D(params, out_png, out_pdf):
    fig, ax = plt.subplots(figsize=(8.0, 7.0))

    n_grid = 260
    n_horiz = 14
    n_vert = 14

    eta_vals = np.linspace(0.03, 1.8, n_horiz)
    xi_vals = np.linspace(-2.4, 2.4, n_vert)
    xi_samp = np.linspace(-2.4, 2.4, n_grid)
    eta_samp = np.linspace(0.02, 1.8, n_grid)

    cmap_horiz = plt.colormaps["Blues"]
    cmap_vert = plt.colormaps["Oranges"]

    for k, eta0 in enumerate(eta_vals):
        zeta_line = xi_samp + 1j * eta0
        z_line = sc_map(zeta_line, params, n_pts=300)
        col = cmap_horiz(0.25 + 0.65 * k / max(n_horiz - 1, 1))
        ax.plot(z_line.real, z_line.imag, color=col,
                linewidth=0.9, zorder=3)

    for k, xi0 in enumerate(xi_vals):
        zeta_line = xi0 + 1j * eta_samp
        z_line = sc_map(zeta_line, params, n_pts=300)
        col = cmap_vert(0.25 + 0.65 * k / max(n_vert - 1, 1))
        ax.plot(z_line.real, z_line.imag, color=col,
                linewidth=0.9, zorder=3)

    z_poly = params.z_poly
    closed = np.append(z_poly, z_poly[0])
    ax.plot(closed.real, closed.imag, color="#333",
            linewidth=1.8, zorder=5)

    ax.set_aspect("equal")
    pad = 0.1
    xmin, xmax = z_poly.real.min() - pad, z_poly.real.max() + pad
    ymin, ymax = z_poly.imag.min() - pad, z_poly.imag.max() + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\mathrm{Re}\,z$", fontsize=11)
    ax.set_ylabel(r"$\mathrm{Im}\,z$", fontsize=11)
    ax.set_title(
        r"Forward image of a rectangular grid in $\mathbb{H}$ under $f : \mathbb{H} \to \Omega$",
        fontsize=11)
    ax.text(xmin + 0.05, ymax - 0.08,
            r"blue: images of horizontal lines $\mathrm{Im}\,\zeta = \eta_0$",
            fontsize=9, color="#1f558a")
    ax.text(xmin + 0.05, ymax - 0.13,
            r"orange: images of vertical lines $\mathrm{Re}\,\zeta = \xi_0$",
            fontsize=9, color="#c26300")
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _sc_integrand_complex(zeta, params):
    """f'(zeta) as a complex number (avoids branch issues for zeta off R)."""
    zk = params.zk
    betas = params.betas
    diffs = zeta - zk
    return params.C * np.prod(diffs ** betas)


def figure_F(params, out_png, out_pdf):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6),
                              gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax_u, ax_urban = axes

    n_zeta = 180
    xi_arr = np.linspace(-2.5, 2.5, n_zeta)
    eta_arr = np.linspace(0.02, 1.6, n_zeta)
    XI, ETA = np.meshgrid(xi_arr, eta_arr)
    zeta_flat = (XI + 1j * ETA).ravel()

    z_flat = sc_map(zeta_flat, params, n_pts=300)
    fprime = np.array(
        [_sc_integrand_complex(zv, params) for zv in zeta_flat]
    )

    U = 1.0
    Wprime_uniform = U / fprime

    obstacle_zeta0 = 0.6 + 0.35j
    a = obstacle_zeta0.imag / 5.36
    Wprime_urban_zeta = (
        U
        - U * a * a / (zeta_flat - obstacle_zeta0) ** 2
        - U * a * a / (zeta_flat - np.conj(obstacle_zeta0)) ** 2
    )
    Wprime_urban = Wprime_urban_zeta / fprime

    def _interp_and_plot(ax, speed_flat, title):
        from scipy.interpolate import griddata
        grid_res = 220
        z_poly = params.z_poly
        pad = 0.03
        xr = (z_poly.real.min() - pad, z_poly.real.max() + pad)
        yr = (z_poly.imag.min() - pad, z_poly.imag.max() + pad)
        gx = np.linspace(*xr, grid_res)
        gy = np.linspace(*yr, grid_res)
        GX, GY = np.meshgrid(gx, gy)

        pts = np.column_stack([z_flat.real, z_flat.imag])
        speed_grid = griddata(pts, speed_flat, (GX, GY),
                              method="linear")

        from matplotlib.path import Path as MplPath
        poly_pts = np.column_stack([z_poly.real, z_poly.imag])
        poly_path = MplPath(np.vstack([poly_pts, poly_pts[:1]]))
        grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
        inside = poly_path.contains_points(grid_pts).reshape(GX.shape)
        masked = np.where(inside, speed_grid, np.nan)

        vmin = np.nanpercentile(masked, 5)
        vmax = np.nanpercentile(masked, 80)

        im = ax.imshow(
            masked,
            origin="lower",
            extent=(xr[0], xr[1], yr[0], yr[1]),
            cmap="viridis",
            vmin=vmin, vmax=vmax,
            aspect="equal",
        )

        closed = np.append(z_poly, z_poly[0])
        ax.plot(closed.real, closed.imag, color="black",
                linewidth=1.5, zorder=4)
        ax.set_xlim(xr)
        ax.set_ylim(yr)
        ax.set_xlabel(r"$\mathrm{Re}\,z$", fontsize=10)
        ax.set_ylabel(r"$\mathrm{Im}\,z$", fontsize=10)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _interp_and_plot(ax_u, np.abs(Wprime_uniform),
                      r"(a) $|W'(z)|$: uniform flow $W = U\zeta$")
    _interp_and_plot(ax_urban, np.abs(Wprime_urban),
                      r"(b) $|W'(z)|$: urban-obstacle potential $\eqref*{eq:urban}$"
                      .replace(r"$\eqref*{eq:urban}$", "(urban)"))

    fig.suptitle(
        r"Normalised velocity magnitude $|W'(z)|/U$ (5th--80th percentile clipped)",
        fontsize=11, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    figures_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    params = load_cache()

    for name, fn in [
        ("figC_prevertex_polygon", figure_C),
        ("figD_conformal_grid", figure_D),
        ("figF_velocity_heatmap", figure_F),
    ]:
        png = os.path.join(figures_dir, f"{name}.png")
        pdf = os.path.join(figures_dir, f"{name}.pdf")
        print(f"Generating {name} ...")
        fn(params, png, pdf)
        print(f"  wrote {png}")


if __name__ == "__main__":
    main()
