
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def interior_angles_pi(z_poly: np.ndarray) -> np.ndarray:
    n = len(z_poly)
    alphas = np.empty(n)

    for k in range(n):
        v_in  = z_poly[k] - z_poly[k - 1]
        v_out = z_poly[(k + 1) % n] - z_poly[k]
        turn = np.angle(v_out / v_in)
        alphas[k] = 1.0 - turn / np.pi

    return alphas


def verify_angle_sum(alphas: np.ndarray, tol: float = 1e-6) -> bool:
    n = len(alphas)
    expected = n - 2
    actual = alphas.sum()
    ok = abs(actual - expected) < tol
    if ok:
        logger.info("Angle sum PASSED: Σαₖ = %.6f ≈ %d (n=%d)", actual, expected, n)
    else:
        logger.warning("Angle sum FAILED: Σαₖ = %.6f ≠ %d (err=%.2e)",
                        actual, expected, abs(actual - expected))
    return ok


def sc_exponents(alphas: np.ndarray) -> np.ndarray:
    return alphas - 1.0
