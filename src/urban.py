
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

URBAN_BOUNDARY_COLOR = "#5C2D91"
URBAN_FILL_COLOR     = "#EDE7F6"

_DOWNTOWN_UTM = np.array([
    [474800, 4428500],
    [476600, 4428100],
    [477900, 4428600],
    [478200, 4430000],
    [477800, 4431100],
    [476000, 4431400],
    [474500, 4430800],
    [474400, 4429500],
], dtype=float)


def _fallback_urban_polygon() -> Polygon:
    return Polygon(_DOWNTOWN_UTM)


def get_urban_polygon(
    boulder_polygon_utm: Polygon,
    method: str = "osmnx",
    n_vertices: int = 8,
    margin_fraction: float = 0.04,
    epsg_utm: int = 26913,
) -> Optional[Polygon]:
    poly: Optional[Polygon] = None

    if method == "osmnx":
        poly = _get_from_osmnx(boulder_polygon_utm, epsg_utm)

    if poly is None:
        logger.warning("Using fallback downtown Boulder polygon")
        poly = _fallback_urban_polygon()

    if poly is None or poly.is_empty:
        logger.error("Failed to obtain any urban polygon")
        return None

    poly = _clip_inside(poly, boulder_polygon_utm, margin_fraction)
    if poly is None or poly.is_empty:
        logger.error("Urban polygon is empty after clipping to Boulder boundary")
        return None

    poly = _simplify_to_n(poly, n_vertices)

    n_v = len(poly.exterior.coords) - 1
    logger.info(
        "Urban polygon ready: %d vertices, area %.2f km²",
        n_v, poly.area / 1e6,
    )
    return poly


def _get_from_osmnx(
    boulder_polygon_utm: Polygon,
    epsg_utm: int = 26913,
) -> Optional[Polygon]:
    try:
        import osmnx as ox
        from pyproj import Transformer

        to_wgs84 = Transformer.from_crs(
            f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True
        )
        coords_wgs84 = np.array([
            to_wgs84.transform(x, y)
            for x, y in boulder_polygon_utm.exterior.coords
        ])
        poly_wgs84 = Polygon(coords_wgs84)

        logger.info("Querying OSM for commercial/retail core polygons …")
        tags = {"landuse": ["commercial", "retail", "institutional"]}
        gdf = ox.features_from_polygon(poly_wgs84, tags=tags)

        polys = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        if len(polys) == 0:
            logger.warning("OSM returned no commercial/retail polygons - trying broader tags")
            tags2 = {"landuse": ["commercial", "retail", "industrial", "institutional"]}
            gdf2 = ox.features_from_polygon(poly_wgs84, tags=tags2)
            polys = gdf2[gdf2.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            if len(polys) == 0:
                return None

        logger.info("OSM commercial/retail: %d polygons found", len(polys))
        polys = polys.set_crs("EPSG:4326").to_crs(f"EPSG:{epsg_utm}")

        union = unary_union(polys.geometry.values)
        if isinstance(union, MultiPolygon):
            union = max(union.geoms, key=lambda g: g.area)

        return union if isinstance(union, Polygon) else None

    except ImportError:
        logger.warning("osmnx not installed - falling back to hardcoded polygon")
        return None
    except Exception as exc:
        logger.warning("OSM download failed (%s) - falling back", exc)
        return None


def _clip_inside(
    inner: Polygon,
    outer: Polygon,
    margin_fraction: float = 0.04,
) -> Optional[Polygon]:
    margin = outer.length * margin_fraction * 0.5
    outer_shrunk = outer.buffer(-margin)

    clipped = inner.intersection(outer_shrunk)
    if clipped.is_empty:
        clipped = inner.intersection(outer)

    if clipped.is_empty:
        return None
    if isinstance(clipped, MultiPolygon):
        clipped = max(clipped.geoms, key=lambda g: g.area)
    return clipped if isinstance(clipped, Polygon) else None


def _simplify_to_n(poly: Polygon, n_target: int) -> Polygon:
    hull = poly.convex_hull
    if not isinstance(hull, Polygon):
        hull = poly

    lo, hi = 0.0, hull.length
    best = hull
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = hull.simplify(mid, preserve_topology=True)
        n = len(s.exterior.coords) - 1
        if n > n_target:
            lo = mid
        elif n < max(4, n_target - 2):
            hi = mid
            best = s
        else:
            best = s
            break

    n_got = len(best.exterior.coords) - 1
    logger.info("Urban polygon simplified: %d vertices (target %d)", n_got, n_target)
    return best


def polygon_to_complex_inner(
    inner_utm: Polygon,
    center: complex,
    scale: float,
) -> np.ndarray:
    coords = np.array(inner_utm.exterior.coords)[:-1]
    z = coords[:, 0] + 1j * coords[:, 1]
    return (z - center) / scale
