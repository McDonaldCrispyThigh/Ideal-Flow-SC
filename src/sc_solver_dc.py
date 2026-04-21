
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from shapely.geometry import Polygon

from .sc_solver import SCParameters
from .flow import sc_inverse_single

logger = logging.getLogger(__name__)


@dataclass
class UrbanObstacle:
    zeta0: complex
    radius: float
    mapped_boundary: np.ndarray
    inner_polygon_norm: Polygon


def compute_urban_obstacle(
    inner_polygon_norm: Polygon,
    outer_params: SCParameters,
    *,
    n_boundary_pts: int = 24,
    radius_margin: float = 1.15,
) -> Optional[UrbanObstacle]:
    boundary_pts = _sample_boundary(inner_polygon_norm, n_boundary_pts)
    logger.info("Mapping %d inner boundary points to ℍ …", len(boundary_pts))

    mapped = []
    n_fail = 0
    zeta_prev = 0.0 + 0.5j
    for z in boundary_pts:
        zeta = sc_inverse_single(z, outer_params, zeta0=zeta_prev, maxfev=600)
        if zeta is not None and zeta.imag > 1e-6:
            mapped.append(zeta)
            zeta_prev = zeta
        else:
            n_fail += 1

    if n_fail > len(boundary_pts) // 2:
        logger.error(
            "Too many inverse-map failures (%d / %d) - cannot fit obstacle",
            n_fail, len(boundary_pts),
        )
        return None

    logger.info(
        "Inner boundary in ℍ: %d / %d points OK  (%d failed)",
        len(mapped), len(boundary_pts), n_fail,
    )

    mapped_arr = np.array(mapped)
    zeta0, radius = _minimum_enclosing_circle(mapped_arr)
    radius *= radius_margin

    if zeta0.imag <= radius:
        logger.warning(
            "Obstacle circle (ζ₀=%.3f+%.3fj, a=%.3f) touches or crosses ℝ - "
            "lifting centre upward",
            zeta0.real, zeta0.imag, radius,
        )
        zeta0 = zeta0.real + (radius + 0.05) * 1j

    logger.info(
        "Urban obstacle circle: ζ₀ = %.4f + %.4fj,  a = %.4f",
        zeta0.real, zeta0.imag, radius,
    )
    logger.info(
        "Separation Im(ζ₀)/a = %.2f  (error ~ (a/Im)² ≈ %.1e)",
        zeta0.imag / radius,
        (radius / zeta0.imag) ** 2,
    )

    return UrbanObstacle(
        zeta0=zeta0,
        radius=radius,
        mapped_boundary=mapped_arr,
        inner_polygon_norm=inner_polygon_norm,
    )


def urban_potential(
    zeta: complex,
    U: float,
    obstacle: UrbanObstacle,
) -> complex:
    z0 = obstacle.zeta0
    a  = obstacle.radius
    a2 = a * a

    d1 = zeta - z0
    d2 = zeta - np.conj(z0)

    if abs(d1) < 1e-12:
        d1 = 1e-12 + 0j
    if abs(d2) < 1e-12:
        d2 = 1e-12 + 0j

    return U * zeta + U * a2 / d1 + U * a2 / d2


def urban_terrain_potential(
    zeta: complex,
    U: float,
    obstacle: UrbanObstacle,
    terrain_sources,
) -> complex:
    from .terrain import terrain_potential
    W_terrain_full = terrain_potential(zeta, U, terrain_sources)
    W_uniform = U * zeta
    terrain_correction = W_terrain_full - W_uniform

    return urban_potential(zeta, U, obstacle) + terrain_correction


def road_terrain_potential(
    zeta: complex,
    U: float,
    terrain_sources,
    road_vortices,
) -> complex:
    from .terrain import terrain_potential
    from .roads import road_potential
    W_terrain = terrain_potential(zeta, U, terrain_sources)
    W_road_correction = road_potential(zeta, 0.0, road_vortices)
    return W_terrain + W_road_correction


def full_potential(
    zeta: complex,
    U: float,
    obstacle: UrbanObstacle,
    terrain_sources,
    road_vortices,
) -> complex:
    from .terrain import terrain_potential
    from .roads import road_potential
    W_urban = urban_potential(zeta, U, obstacle)
    W_terrain_full = terrain_potential(zeta, U, terrain_sources)
    terrain_correction = W_terrain_full - U * zeta
    W_road_full = road_potential(zeta, 0.0, road_vortices)
    return W_urban + terrain_correction + W_road_full


def _sample_boundary(poly: Polygon, n_pts: int) -> list[complex]:
    coords = np.array(poly.exterior.coords)
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumlen[-1]

    sample_lens = np.linspace(0, total, n_pts, endpoint=False)
    pts = []
    for s in sample_lens:
        idx = np.searchsorted(cumlen, s, side="right") - 1
        idx = min(idx, len(coords) - 2)
        t = (s - cumlen[idx]) / max(seg_lengths[idx], 1e-15)
        p = coords[idx] * (1 - t) + coords[idx + 1] * t
        pts.append(p[0] + 1j * p[1])
    return pts


def _minimum_enclosing_circle(
    points: np.ndarray,
) -> tuple[complex, float]:
    import random as _random

    pts = list(points)
    _random.shuffle(pts)


    def _c1(p: complex) -> tuple[complex, float]:
        return p, 0.0

    def _c2(p: complex, q: complex) -> tuple[complex, float]:
        c = (p + q) / 2
        return c, abs(p - c)

    def _c3(p: complex, q: complex, r: complex) -> tuple[complex, float]:
        ax, ay = p.real, p.imag
        bx, by = q.real, q.imag
        cx, cy = r.real, r.imag
        D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-12:
            d_pq, d_pr, d_qr = abs(p - q), abs(p - r), abs(q - r)
            if d_pq >= d_pr and d_pq >= d_qr:
                return _c2(p, q)
            elif d_pr >= d_pq and d_pr >= d_qr:
                return _c2(p, r)
            else:
                return _c2(q, r)
        ux = ((ax**2 + ay**2) * (by - cy) +
              (bx**2 + by**2) * (cy - ay) +
              (cx**2 + cy**2) * (ay - by)) / D
        uy = ((ax**2 + ay**2) * (cx - bx) +
              (bx**2 + by**2) * (ax - cx) +
              (cx**2 + cy**2) * (bx - ax)) / D
        centre = ux + 1j * uy
        return centre, abs(p - centre)

    def _in_circle(c: complex, r: float, p: complex) -> bool:
        return abs(p - c) <= r + 1e-10


    def _welzl(P: list, R: list, n: int) -> tuple[complex, float]:
        if n == 0 or len(R) == 3:
            if len(R) == 0:
                return 0j, 0.0
            elif len(R) == 1:
                return _c1(R[0])
            elif len(R) == 2:
                return _c2(R[0], R[1])
            else:
                return _c3(R[0], R[1], R[2])
        p = P[n - 1]
        c, r = _welzl(P, R, n - 1)
        if _in_circle(c, r, p):
            return c, r
        return _welzl(P, R + [p], n - 1)

    return _welzl(pts, [], len(pts))
