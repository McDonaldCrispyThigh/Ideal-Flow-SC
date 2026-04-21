
from __future__ import annotations

import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from .sc_solver import SCParameters

logger = logging.getLogger(__name__)


@dataclass
class TerrainInfo:
    elevations: np.ndarray
    grad_xy: Tuple[float, float]
    theta_downhill: float
    slope_magnitude: float
    sources: List[Tuple[complex, float]] = field(default_factory=list)


_EPQS_URL = "https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&units=Meters&wkid=4326"
_UTM_TO_LONLAT = None


def _get_transformer(epsg_source: int = 26913) -> Transformer:
    global _UTM_TO_LONLAT
    if _UTM_TO_LONLAT is None:
        _UTM_TO_LONLAT = Transformer.from_crs(
            f"EPSG:{epsg_source}", "EPSG:4326", always_xy=True
        )
    return _UTM_TO_LONLAT


def _query_epqs(lon: float, lat: float, timeout: float = 12.0) -> Optional[float]:
    url = _EPQS_URL.format(lon=lon, lat=lat)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "APPM4360-SC-Flow/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            val = float(data["value"])
            if val < -1000:
                return None
            return val
    except Exception as exc:
        logger.debug("EPQS query failed (%.4f, %.4f): %s", lon, lat, exc)
        return None


def _sample_boundary_points(
    polygon_utm: Polygon,
    n_per_edge: int = 3,
) -> np.ndarray:
    coords = np.array(polygon_utm.exterior.coords)
    pts = []
    for i in range(len(coords) - 1):
        for t in np.linspace(0, 1, n_per_edge + 2)[1:-1]:
            pts.append(coords[i] * (1 - t) + coords[i + 1] * t)
    return np.array(pts) if pts else np.empty((0, 2))


def _sample_interior_points(
    polygon_utm: Polygon,
    n_pts: int = 25,
) -> np.ndarray:
    minx, miny, maxx, maxy = polygon_utm.bounds
    side = int(np.ceil(np.sqrt(n_pts * 1.5)))
    xs = np.linspace(minx, maxx, side + 2)[1:-1]
    ys = np.linspace(miny, maxy, side + 2)[1:-1]
    pts = []
    for x in xs:
        for y in ys:
            if polygon_utm.contains(Point(x, y)):
                pts.append([x, y])
            if len(pts) >= n_pts:
                break
        if len(pts) >= n_pts:
            break
    return np.array(pts) if pts else np.empty((0, 2))


def _batch_query_elevations(
    coords_utm: np.ndarray,
    epsg_source: int = 26913,
    max_workers: int = 6,
    timeout: float = 15.0,
) -> np.ndarray:
    n = len(coords_utm)
    elevations = np.full(n, np.nan)
    if n == 0:
        return elevations

    transformer = _get_transformer(epsg_source)
    lonlats = np.array([transformer.transform(x, y) for x, y in coords_utm])

    def _query_one(idx: int) -> Tuple[int, Optional[float]]:
        lon, lat = lonlats[idx]
        return idx, _query_epqs(lon, lat, timeout=timeout)

    logger.info("Querying USGS 3DEP for %d points (%d concurrent) …",
                n, max_workers)
    n_ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_query_one, i): i for i in range(n)}
        for fut in tqdm(as_completed(futures), total=n, desc="USGS 3DEP"):
            idx, val = fut.result()
            if val is not None:
                elevations[idx] = val
                n_ok += 1

    logger.info("Elevation API: %d / %d points OK (%.0f%%)",
                n_ok, n, 100 * n_ok / max(n, 1))
    return elevations


def get_vertex_elevations(
    polygon_utm: Polygon,
    epsg_source: int = 26913,
) -> np.ndarray:
    coords = np.array(polygon_utm.exterior.coords)[:-1]
    elevations = _batch_query_elevations(coords, epsg_source)

    if np.sum(np.isfinite(elevations)) < 3:
        logger.warning("Too few API results - using fallback elevation model")
        elevations = _boulder_fallback(coords)

    return elevations


def get_dense_elevations(
    polygon_utm: Polygon,
    n_per_edge: int = 3,
    n_interior: int = 25,
    epsg_source: int = 26913,
    max_workers: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    vertex_coords = np.array(polygon_utm.exterior.coords)[:-1]
    boundary_coords = _sample_boundary_points(polygon_utm, n_per_edge)
    interior_coords = _sample_interior_points(polygon_utm, n_interior)

    all_coords = np.vstack([
        vertex_coords,
        boundary_coords,
        interior_coords,
    ])
    n_v = len(vertex_coords)
    n_b = len(boundary_coords)
    n_i = len(interior_coords)
    logger.info("Elevation sample plan: %d vertices + %d boundary + %d interior = %d total",
                n_v, n_b, n_i, len(all_coords))

    all_elevs = _batch_query_elevations(all_coords, epsg_source, max_workers)

    nan_mask = np.isnan(all_elevs)
    if nan_mask.any():
        fb = _boulder_fallback(all_coords[nan_mask])
        all_elevs[nan_mask] = fb
        logger.info("Filled %d NaN elevations with fallback model", nan_mask.sum())

    return all_coords, all_elevs


def _boulder_fallback(coords_utm: np.ndarray) -> np.ndarray:
    x = coords_utm[:, 0]
    x_min, x_max = x.min(), x.max()
    span = x_max - x_min + 1e-6
    elev = 1770.0 - 200.0 * (x - x_min) / span
    logger.info("Fallback elevations: %.0f … %.0f m", elev.min(), elev.max())
    return elev


def _fit_rbf(all_coords: np.ndarray, all_elevs: np.ndarray) -> RBFInterpolator:
    scale = all_coords.std(axis=0).mean() + 1e-9
    centre = all_coords.mean(axis=0)
    coords_n = (all_coords - centre) / scale
    rbf = RBFInterpolator(coords_n, all_elevs, kernel="thin_plate_spline")

    n_pts = len(all_elevs)
    k = min(5, n_pts)
    fold_size = n_pts // k
    oof_pred = np.empty(n_pts)
    for fold in range(k):
        val_mask = np.zeros(n_pts, dtype=bool)
        val_mask[fold * fold_size: (fold + 1) * fold_size] = True
        train_mask = ~val_mask
        rbf_fold = RBFInterpolator(
            coords_n[train_mask], all_elevs[train_mask],
            kernel="thin_plate_spline",
        )
        oof_pred[val_mask] = rbf_fold(coords_n[val_mask])
    remainder = k * fold_size
    if remainder < n_pts:
        val_mask = np.zeros(n_pts, dtype=bool)
        val_mask[remainder:] = True
        train_mask = ~val_mask
        rbf_fold = RBFInterpolator(
            coords_n[train_mask], all_elevs[train_mask],
            kernel="thin_plate_spline",
        )
        oof_pred[val_mask] = rbf_fold(coords_n[val_mask])
    ss_res = np.sum((all_elevs - oof_pred) ** 2)
    ss_tot = np.sum((all_elevs - all_elevs.mean()) ** 2)
    cv_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
    logger.info(
        "RBF thin-plate-spline 5-fold CV R² = %.4f  (%d sample points)",
        cv_r2, n_pts,
    )

    def _eval(coords_utm: np.ndarray) -> np.ndarray:
        return rbf((coords_utm - centre) / scale)

    return _eval


def _vertex_gradients(
    vertex_coords: np.ndarray,
    rbf_eval,
    h: float = 100.0,
) -> np.ndarray:
    n_v = len(vertex_coords)
    grads = np.zeros((n_v, 2))

    xp = vertex_coords.copy(); xp[:, 0] += h
    xm = vertex_coords.copy(); xm[:, 0] -= h
    yp = vertex_coords.copy(); yp[:, 1] += h
    ym = vertex_coords.copy(); ym[:, 1] -= h

    grads[:, 0] = (rbf_eval(xp) - rbf_eval(xm)) / (2.0 * h)
    grads[:, 1] = (rbf_eval(yp) - rbf_eval(ym)) / (2.0 * h)
    return grads


def compute_terrain_info(
    polygon_utm: Polygon,
    sc_params: SCParameters,
    *,
    delta: float = 0.25,
    Q_scale: float = 0.35,
    flow_direction: float = 0.0,
    n_per_edge: int = 3,
    n_interior: int = 25,
    max_workers: int = 6,
    epsg_source: int = 26913,
) -> TerrainInfo:
    all_coords, all_elevs = get_dense_elevations(
        polygon_utm, n_per_edge=n_per_edge, n_interior=n_interior,
        epsg_source=epsg_source, max_workers=max_workers,
    )

    vertex_coords = np.array(polygon_utm.exterior.coords)[:-1]
    n_v = len(vertex_coords)
    elevations = all_elevs[:n_v]

    rbf_eval = _fit_rbf(all_coords, all_elevs)

    x, y = all_coords[:, 0], all_coords[:, 1]
    A_mat = np.column_stack([np.ones_like(x), x, y])
    coeffs, *_ = np.linalg.lstsq(A_mat, all_elevs, rcond=None)
    grad_x, grad_y = float(coeffs[1]), float(coeffs[2])
    slope_mag = float(np.hypot(grad_x, grad_y))
    theta_down = float(np.arctan2(-grad_y, -grad_x))

    logger.info("Mean terrain gradient (linear fit): "
                "(%.5f, %.5f) m/m  |∇e| = %.5f  downhill = %.1f°",
                grad_x, grad_y, slope_mag, np.degrees(theta_down))

    vertex_grads = _vertex_gradients(vertex_coords, rbf_eval)

    cos_f, sin_f = np.cos(flow_direction), np.sin(flow_direction)
    projected = vertex_grads[:, 0] * cos_f + vertex_grads[:, 1] * sin_f

    max_proj = np.max(np.abs(projected))
    if max_proj < 1e-8:
        logger.warning("Terrain gradient too small - no sources added")
        q_strengths = np.zeros(n_v)
    else:
        q_strengths = Q_scale * projected / max_proj

    zk = sc_params.zk
    sources: List[Tuple[complex, float]] = []
    n_sources = n_sinks = 0

    for k in range(n_v):
        q = float(q_strengths[k])
        if abs(q) < 1e-4 * Q_scale:
            continue
        s_k = complex(zk[k]) + delta * 1j
        sources.append((s_k, q))
        if q > 0:
            n_sources += 1
        else:
            n_sinks += 1

    logger.info(
        "Per-vertex sources: %d sources (+) and %d sinks (−)  "
        "(|q| threshold = %.4f)",
        n_sources, n_sinks, 1e-4 * Q_scale,
    )
    if sources:
        qs = [abs(q) for _, q in sources]
        logger.info("  Q range: %.4f … %.4f  (max = %.4f)",
                    min(qs), max(qs), Q_scale)

    return TerrainInfo(
        elevations=elevations,
        grad_xy=(grad_x, grad_y),
        theta_downhill=theta_down,
        slope_magnitude=slope_mag,
        sources=sources,
    )


def terrain_potential(
    zeta: complex,
    U: float,
    sources: List[Tuple[complex, float]],
) -> complex:
    W = U * zeta
    for s, Q in sources:
        d1 = zeta - s
        d2 = zeta - np.conj(s)
        if abs(d1) < 1e-12:
            d1 = 1e-12 + 0j
        if abs(d2) < 1e-12:
            d2 = 1e-12 + 0j
        W += (Q / (2.0 * np.pi)) * (np.log(d1) + np.log(d2))
    return W


def uniform_potential(zeta: complex, U: float = 1.0) -> complex:
    return U * zeta
