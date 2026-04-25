"""Regenerate fig1–fig8 using the cached county SC params (no re-solve)."""
from __future__ import annotations

import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.polygon import (
    load_boulder_polygon,
    simplify_polygon,
    polygon_to_complex,
    complex_to_polygon,
    ensure_ccw,
    smooth_extreme_angles,
    find_best_rotation,
)
from src.angles import interior_angles_pi
from src.sc_solver import SCParameters, sc_map
from src.flow import compute_flow_grid, compute_flow_grid_urban, compute_curves_forward
from src.visualization import (
    plot_polygon_comparison,
    plot_streamlines,
    plot_equipotentials,
    plot_combined,
    plot_urban_flow,
    plot_flow_comparison,
    URBAN_STREAM,
    URBAN_EQUIP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("regen_figures")


def main():
    shapefile = PROJECT_ROOT / "data" / "raw" / "tl_2025_08_county"
    cache_path = PROJECT_ROOT / "scripts" / "boulder_cache.pkl"
    n_grid = 80

    # ── Load cached SC params ──────────────────────────────────────────────
    logger.info("Loading cached SC params from %s", cache_path)
    with open(cache_path, "rb") as f:
        c = pickle.load(f)
    params = SCParameters(
        zk=c["params_zk"],
        alphas=c["params_alphas"],
        betas=c["params_betas"],
        A=c["params_A"],
        C=c["params_C"],
        z_poly=c["params_z_poly"],
    )
    z_poly = c["z_poly_norm"]
    logger.info("Cached polygon: %d vertices, W/H=%.3f",
                len(z_poly),
                (z_poly.real.max() - z_poly.real.min()) /
                (z_poly.imag.max() - z_poly.imag.min()))

    # ── Load shapefile to get original + simplified UTM polygons ───────────
    logger.info("Loading Boulder County shapefile ...")
    original_utm = load_boulder_polygon(
        shapefile, name_field="NAME", name_value="Boulder"
    )
    simplified_utm = simplify_polygon(
        original_utm, tolerance=1500.0, min_vertices=12, max_vertices=16
    )

    # Derive center/scale from the cached z_poly so axes match
    z_raw, center, scale = polygon_to_complex(simplified_utm, normalise=True)
    logger.info("center=%.1f%+.1fj  scale=%.1f", center.real, center.imag, scale)

    norm_polygon = complex_to_polygon(z_poly)

    # ── Compute uniform flow grid ──────────────────────────────────────────
    logger.info("Computing uniform flow grid (%d×%d) ...", n_grid, n_grid)
    XX, YY, Psi, Phi, Zeta = compute_flow_grid(norm_polygon, params, n_grid=n_grid)
    logger.info("Valid ψ points: %d", int(np.isfinite(Psi).sum()))

    # ── Generate figs 1–4 ─────────────────────────────────────────────────
    logger.info("Generating fig1 (polygon comparison) ...")
    plot_polygon_comparison(original_utm, simplified_utm)

    logger.info("Computing forward-map curves ...")
    stream_curves, equip_curves = compute_curves_forward(
        params, norm_polygon, n_stream=28, n_equip=28,
    )

    logger.info("Generating fig2 (streamlines) ...")
    plot_streamlines(XX, YY, Psi, norm_polygon, stream_curves=stream_curves)

    logger.info("Generating fig3 (equipotentials) ...")
    plot_equipotentials(XX, YY, Phi, norm_polygon, equip_curves=equip_curves)

    logger.info("Generating fig4 (combined grid) ...")
    plot_combined(XX, YY, Psi, Phi, norm_polygon,
                  stream_curves=stream_curves, equip_curves=equip_curves)

    # ── Urban obstacle (figs 7, 8) ─────────────────────────────────────────
    logger.info("Computing urban obstacle ...")
    try:
        from src.urban import get_urban_polygon, polygon_to_complex_inner
        from src.sc_solver_dc import compute_urban_obstacle

        urban_poly_utm = get_urban_polygon(simplified_utm, method="osmnx", n_vertices=8)
        if urban_poly_utm is None:
            raise RuntimeError("get_urban_polygon returned None")

        z_inner = polygon_to_complex_inner(urban_poly_utm, center, scale)
        norm_poly_inner = ShapelyPolygon([(z.real, z.imag) for z in z_inner])

        urban_obstacle = compute_urban_obstacle(
            norm_poly_inner, params, n_boundary_pts=24
        )
        if urban_obstacle is None:
            raise RuntimeError("Urban obstacle fitting failed")

        logger.info("Computing urban flow grid ...")
        XX_u, YY_u, Psi_u, Phi_u, _ = compute_flow_grid_urban(
            norm_polygon, norm_poly_inner, params, urban_obstacle, n_grid=n_grid
        )
        logger.info("Valid urban ψ: %d", int(np.isfinite(Psi_u).sum()))

        logger.info("Generating fig7 (urban flow) ...")
        psi_ref = (float(np.nanpercentile(Psi, 5)),
                   float(np.nanpercentile(Psi, 95)))
        plot_urban_flow(
            XX_u, YY_u, Psi_u, Phi_u, norm_polygon,
            norm_polygon_inner=norm_poly_inner,
            obstacle=urban_obstacle,
            psi_ref_range=psi_ref,
        )

        logger.info("Generating fig8 (urban vs uniform) ...")
        plot_flow_comparison(
            XX, YY, Psi, Phi, Psi_u, Phi_u, norm_polygon,
            filename="fig8_urban_vs_uniform.png",
            title_right=r"Urban-Obstacle Flow  (doubly-connected)",
            suptitle="Flow Comparison - Uniform vs. Urban Obstacle",
            stream_color_right=URBAN_STREAM,
            equip_color_right=URBAN_EQUIP,
        )

    except Exception as exc:
        logger.warning("Urban obstacle skipped: %s", exc)

    logger.info("Done — figures written to figures/")


if __name__ == "__main__":
    main()
