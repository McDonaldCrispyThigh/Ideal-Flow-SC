"""Run the Boulder SC pipeline once and pickle the SCParameters + polygon
for reuse in figure-generation scripts (no re-solving needed)."""
from __future__ import annotations

import logging
import os
import pickle
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.polygon import (
    load_boulder_polygon,
    simplify_polygon,
    polygon_to_complex,
    ensure_ccw,
    smooth_extreme_angles,
)
from src.angles import interior_angles_pi, sc_exponents
from src.sc_solver import solve_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_cache")


def main():
    shp = os.path.join(PROJECT_ROOT, "data", "raw", "tl_2025_08_county")
    out_path = os.path.join(PROJECT_ROOT, "scripts", "boulder_cache.pkl")

    logger.info("Loading Boulder polygon from %s", shp)
    original_utm = load_boulder_polygon(shp)
    simplified_utm = simplify_polygon(
        original_utm, tolerance=1500.0,
        min_vertices=12, max_vertices=16,
    )

    z_poly_raw, _, _ = polygon_to_complex(simplified_utm, normalise=True)
    z_poly_raw = ensure_ccw(z_poly_raw)
    z_poly_norm = smooth_extreme_angles(
        z_poly_raw, alpha_min=0.35, alpha_max=1.75, min_vertices=10
    )
    logger.info("Simplified polygon: %d vertices", len(z_poly_norm))

    alphas = interior_angles_pi(z_poly_norm)
    betas = sc_exponents(alphas)
    logger.info("Solving SC parameter problem (this takes a few minutes)...")
    params = solve_parameters(z_poly_norm, alphas)

    cache = {
        "z_poly_norm": z_poly_norm,
        "alphas": alphas,
        "betas": betas,
        "params_zk": params.zk,
        "params_alphas": params.alphas,
        "params_betas": params.betas,
        "params_A": params.A,
        "params_C": params.C,
        "params_z_poly": params.z_poly,
    }
    with open(out_path, "wb") as f:
        pickle.dump(cache, f)
    logger.info("Cache saved to %s", out_path)
    logger.info("zk = %s", params.zk)
    logger.info("A = %s, C = %s", params.A, params.C)


if __name__ == "__main__":
    main()
