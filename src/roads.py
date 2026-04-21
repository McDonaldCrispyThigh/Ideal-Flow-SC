
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .sc_solver import SCParameters

logger = logging.getLogger(__name__)


@dataclass
class RoadInfo:
    n_intersections: int
    intersection_positions_utm: np.ndarray
    intersection_degrees: np.ndarray
    vortices: List[Tuple[complex, float]] = field(default_factory=list)


def get_road_intersections(
    polygon_utm,
    method: str = "osmnx",
    n_max: int = 12,
    road_types: tuple = ("primary", "secondary", "tertiary"),
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if method == "osmnx":
        try:
            return _osmnx_intersections(polygon_utm, n_max, road_types)
        except Exception as exc:
            logger.warning("OSMnx road fetch failed: %s  — using fallback", exc)

    return _fallback_intersections(polygon_utm, n_max)


def _osmnx_intersections(
    polygon_utm,
    n_max: int,
    road_types: tuple,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    import osmnx as ox
    from pyproj import Transformer
    from shapely.geometry import Point, Polygon as ShapelyPolygon

    logger.info("Downloading OSM road network (types: %s) …", ", ".join(road_types))

    tf_fwd = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)
    coords_utm = np.array(polygon_utm.exterior.coords)
    coords_wgs = np.array([tf_fwd.transform(x, y) for x, y in coords_utm])
    polygon_wgs = ShapelyPolygon(coords_wgs)

    cf = '["highway"~"{}"]'.format("|".join(road_types))
    G = ox.graph_from_polygon(polygon_wgs, custom_filter=cf, network_type="drive")
    logger.info("Road graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    G_ud = ox.convert.to_undirected(G)
    degrees = dict(G_ud.degree())

    crossings = [(nid, deg) for nid, deg in degrees.items() if deg >= 3]
    if not crossings:
        logger.warning("No intersections with degree ≥ 3 found")
        return None

    crossings.sort(key=lambda x: x[1], reverse=True)
    crossings = crossings[:n_max]

    tf_back = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)
    coords_out, degrees_out = [], []
    for nid, deg in crossings:
        node = G_ud.nodes[nid]
        x_utm, y_utm = tf_back.transform(node["x"], node["y"])
        if polygon_utm.contains(Point(x_utm, y_utm)):
            coords_out.append([x_utm, y_utm])
            degrees_out.append(deg)

    if not coords_out:
        logger.warning("No OSM intersections landed inside polygon")
        return None

    logger.info("OSM road intersections inside polygon: %d", len(coords_out))
    return np.array(coords_out), np.array(degrees_out, dtype=float)


def _fallback_intersections(
    polygon_utm,
    n_max: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    _LONLAT = [
        (-105.2835, 40.0186),
        (-105.2835, 40.0135),
        (-105.2519, 40.0186),
        (-105.2519, 40.0135),
        (-105.2366, 40.0135),
        (-105.2835, 40.0296),
        (-105.2519, 40.0296),
        (-105.2835, 40.0050),
        (-105.2519, 40.0050),
        (-105.2366, 40.0050),
        (-105.2683, 40.0186),
        (-105.2683, 40.0050),
    ]
    _DEG = [5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0]

    from pyproj import Transformer
    from shapely.geometry import Point

    tf = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)
    coords_out, degrees_out = [], []
    for (lon, lat), deg in zip(_LONLAT[:n_max], _DEG[:n_max]):
        x, y = tf.transform(lon, lat)
        if polygon_utm.contains(Point(x, y)):
            coords_out.append([x, y])
            degrees_out.append(deg)

    if not coords_out:
        logger.warning("No fallback intersections inside polygon")
        return None

    logger.info("Fallback road intersections inside polygon: %d", len(coords_out))
    return np.array(coords_out), np.array(degrees_out, dtype=float)


def compute_road_info(
    polygon_utm,
    center: complex,
    scale: float,
    sc_params: SCParameters,
    *,
    method: str = "osmnx",
    n_max: int = 12,
    Gamma_scale: float = 0.20,
    delta_min: float = 0.15,
    road_types: tuple = ("primary", "secondary", "tertiary"),
) -> Optional[RoadInfo]:
    result = get_road_intersections(polygon_utm, method=method,
                                    n_max=n_max, road_types=road_types)
    if result is None:
        return None

    coords_utm, degrees = result

    z_norm = (coords_utm[:, 0] + 1j * coords_utm[:, 1] - center) / scale
    logger.info("Road intersections (normalised): %s",
                np.array2string(z_norm, precision=3))

    from .flow import sc_inverse_single

    vortices: List[Tuple[complex, float]] = []
    n_ok = 0
    logger.info("SC inverse map for %d road intersections …", len(z_norm))

    zeta_prev = 0.0 + 0.5j
    for k, z_t in enumerate(z_norm):
        zeta = sc_inverse_single(z_t, sc_params, zeta0=zeta_prev, maxfev=800)
        if zeta is None:
            logger.warning("  Intersection %d: SC inverse failed — skipped", k)
            continue
        if zeta.imag < delta_min:
            zeta = zeta.real + delta_min * 1j

        z_centroid = np.mean(z_norm)
        sign = +1.0 if z_t.imag >= z_centroid.imag else -1.0

        Gamma = sign * Gamma_scale * float(degrees[k]) / float(max(degrees))
        vortices.append((zeta, Gamma))
        logger.info(
            "  [%d] z=%.3f%+.3fj → ζ=%.4f%+.4fj  Γ=%+.4f  (deg=%d)",
            k, z_t.real, z_t.imag, zeta.real, zeta.imag, Gamma, int(degrees[k]),
        )
        zeta_prev = zeta
        n_ok += 1

    if not vortices:
        logger.warning("No road vortices placed — road model disabled")
        return None

    logger.info("Road vortex model: %d vortices placed (Γ_max = %.4f)",
                n_ok, Gamma_scale)
    return RoadInfo(
        n_intersections=n_ok,
        intersection_positions_utm=coords_utm[:n_ok],
        intersection_degrees=degrees[:n_ok],
        vortices=vortices,
    )


def road_potential(
    zeta: complex,
    U: float,
    vortices: List[Tuple[complex, float]],
) -> complex:
    W = U * zeta
    for s, Gamma in vortices:
        d1 = zeta - s
        d2 = zeta - np.conj(s)
        if abs(d1) < 1e-12:
            d1 = 1e-12 + 0j
        if abs(d2) < 1e-12:
            d2 = 1e-12 + 0j
        W += (-1j * Gamma / (2.0 * np.pi)) * (np.log(d1) + np.log(d2))
    return W
