
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)


@dataclass
class SCParameters:
    zk: np.ndarray
    alphas: np.ndarray
    betas: np.ndarray
    A: complex
    C: complex
    z_poly: np.ndarray


def _sc_prod_real(t: np.ndarray, zk: np.ndarray, betas: np.ndarray) -> np.ndarray:
    diffs = t[:, None] - zk[None, :]
    signs = np.where(diffs < 0, np.pi, 0.0)
    log_abs = betas[None, :] * np.log(np.abs(diffs) + 1e-300)
    phases  = 1j * betas[None, :] * signs
    return np.prod(np.exp(log_abs + phases), axis=1)


def _sc_prod_complex(t: np.ndarray, zk: np.ndarray, betas: np.ndarray) -> np.ndarray:
    diffs = t[:, None] - zk[None, :]
    return np.prod(np.exp(betas[None, :] * np.log(diffs)), axis=1)


_GL_CACHE: dict[int, tuple] = {}

def _gl_nodes(n_pts: int):
    if n_pts not in _GL_CACHE:
        _GL_CACHE[n_pts] = np.polynomial.legendre.leggauss(n_pts)
    return _GL_CACHE[n_pts]


def integrate_real(a: float, b: float,
                   zk: np.ndarray, betas: np.ndarray,
                   n_pts: int = 500) -> complex:
    nodes, weights = _gl_nodes(n_pts)
    mid  = 0.5 * (a + b)
    half = 0.5 * (b - a)
    t = mid + half * nodes
    vals = _sc_prod_real(t, zk, betas)
    return complex(half * np.dot(weights, vals))


def integrate_complex(za: complex, zb: complex,
                      zk: np.ndarray, betas: np.ndarray,
                      n_pts: int = 400) -> complex:
    nodes, weights = _gl_nodes(n_pts)
    mid  = 0.5 * (za + zb)
    half = 0.5 * (zb - za)
    t = mid + half * nodes
    vals = _sc_prod_complex(t, zk, betas)
    return complex(half * np.dot(weights, vals))


def _side_length(zk_a: float, zk_b: float,
                 zk: np.ndarray, betas: np.ndarray) -> float:
    return abs(integrate_real(zk_a, zk_b, zk, betas))


def _all_side_lengths(zk: np.ndarray, betas: np.ndarray,
                      R: float = 5000.0) -> np.ndarray:
    n = len(zk)
    lengths = np.empty(n)
    for i in range(n - 1):
        lengths[i] = _side_length(zk[i], zk[i + 1], zk, betas)
    seg1 = abs(integrate_real(zk[-1], zk[-1] + R, zk, betas))
    seg2 = abs(integrate_real(zk[0] - R, zk[0], zk, betas))
    lengths[n - 1] = seg1 + seg2
    return lengths


def solve_parameters(
    z_poly: np.ndarray,
    alphas: np.ndarray,
    *,
    maxiter: int = 2000,
    tol: float = 1e-10,
) -> SCParameters:
    n = len(z_poly)
    betas = alphas - 1.0

    target_sides = np.abs(np.diff(np.append(z_poly, z_poly[0])))
    target_ratios = target_sides[:-1] / target_sides[-1]

    fixed_vals = {0: -1.0, 1: 0.0, n - 1: 1.0}
    free_idx = [i for i in range(n) if i not in fixed_vals]
    n_free = len(free_idx)

    if n_free == 0:
        zk = np.array([-1.0, 0.0, 1.0])
    else:

        def _softmax_to_zk_free(p: np.ndarray) -> np.ndarray:
            w = np.append(np.exp(p - p.max()), 1.0)
            w /= w.sum()
            return np.cumsum(w[:n_free])

        init_sides = target_sides[1:n - 1]
        ratios = np.maximum(init_sides[:-1] / init_sides[-1], 1e-6)
        x0 = np.log(ratios)

        def _residuals(p):
            zk_arr = np.empty(n)
            for idx, val in fixed_vals.items():
                zk_arr[idx] = val
            zk_free = _softmax_to_zk_free(p)
            for k, idx in enumerate(free_idx):
                zk_arr[idx] = zk_free[k]
            sl = _all_side_lengths(zk_arr, betas)
            ratios = sl[:-1] / sl[-1]
            return ratios - target_ratios

        logger.info("Solving SC parameters (n=%d, free=%d) …", n, n_free)
        result = optimize.least_squares(
            _residuals, x0,
            method="lm",
            max_nfev=maxiter * 100,
            ftol=tol, xtol=tol, gtol=tol,
        )
        logger.info("SC solver initial run: cost=%.2e", result.cost)

        best_result = result
        for restart in range(5):
            if best_result.cost < 1e-4:
                break
            noise = np.random.randn(len(x0)) * 0.3
            r2 = optimize.least_squares(
                _residuals, best_result.x + noise,
                method="lm",
                max_nfev=maxiter * 50,
                ftol=tol, xtol=tol, gtol=tol,
            )
            logger.info("SC solver restart %d: cost=%.2e", restart + 1, r2.cost)
            if r2.cost < best_result.cost:
                best_result = r2
        result = best_result

        if result.cost < 1e-4:
            logger.info("SC solver converged: cost=%.2e", result.cost)
        else:
            logger.warning("SC solver: cost=%.2e after restarts (may not have converged)",
                           result.cost)

        zk = np.empty(n)
        for idx, val in fixed_vals.items():
            zk[idx] = val
        zk_free = _softmax_to_zk_free(result.x)
        for k, idx in enumerate(free_idx):
            zk[idx] = zk_free[k]

    zk = np.sort(zk)

    A, C = _solve_AC(zk, betas, z_poly)

    params = SCParameters(zk=zk, alphas=alphas, betas=betas,
                          A=A, C=C, z_poly=z_poly)

    logger.info("=== SC solver diagnostics ===")
    logger.info("pre-vertices zk = %s", np.round(zk, 6))
    logger.info("A = %.6g%+.6gj   |C| = %.6g   arg(C) = %.4f rad",
                A.real, A.imag, abs(C), np.angle(C))
    for k in range(n - 1):
        I_side = integrate_real(zk[k], zk[k + 1], zk, betas)
        computed = C * I_side
        target   = z_poly[k + 1] - z_poly[k]
        err      = abs(computed - target)
        logger.info("  side %2d: |C*I - Δz| = %.4e  (|Δz|=%.4f, arg_ratio=%.3f°)",
                    k, err, abs(target), np.degrees(np.angle(computed/target)))

    return params


def _solve_AC(zk, betas, z_poly):
    n = len(z_poly)

    sides_z = np.diff(np.append(z_poly, z_poly[0]))
    sides_I = np.array(
        [integrate_real(zk[k], zk[k + 1], zk, betas) for k in range(n - 1)],
        dtype=complex,
    )

    M_C = np.zeros((2 * (n - 1), 2))
    rhs_C = np.zeros(2 * (n - 1))
    for k in range(n - 1):
        Ik, sk = sides_I[k], sides_z[k]
        M_C[2 * k,     :] = [ Ik.real, -Ik.imag]
        M_C[2 * k + 1, :] = [ Ik.imag,  Ik.real]
        rhs_C[2 * k]     = sk.real
        rhs_C[2 * k + 1] = sk.imag
    xC, _, _, _ = np.linalg.lstsq(M_C, rhs_C, rcond=None)
    C = xC[0] + 1j * xC[1]

    gaps = np.diff(zk)
    k_safe = int(np.argmax(gaps[:-1]))
    mid_real = 0.5 * (zk[k_safe] + zk[k_safe + 1])

    f_mid_real = z_poly[k_safe] + C * integrate_real(zk[k_safe], mid_real, zk, betas)

    h = 0.3
    mid_above = mid_real + 1j * h
    I_ascent = integrate_complex(mid_real + 1e-9j, mid_above, zk, betas)
    f_mid_above = f_mid_real + C * I_ascent

    p1 = _ZETA_REF.real + 1j * h
    p2 = mid_real        + 1j * h
    I_ref_to_mid = 0.0 + 0.0j
    if abs(_ZETA_REF - p1) > 1e-14:
        I_ref_to_mid += integrate_complex(_ZETA_REF, p1, zk, betas)
    if abs(p1 - p2) > 1e-14:
        I_ref_to_mid += integrate_complex(p1, p2, zk, betas)

    A = f_mid_above - C * I_ref_to_mid
    return A, C


_ZETA_REF = 0.0 + 0.5j


def sc_map_single(zeta: complex, params: SCParameters, n_pts: int = 400) -> complex:
    zk, betas, A, C = params.zk, params.betas, params.A, params.C

    if abs(zeta.imag) < 1e-12:
        zeta = zeta.real + 1e-10j

    delta = max(abs(zeta.imag), 0.3)
    p_mid1 = _ZETA_REF.real + 1j * delta
    p_mid2 = zeta.real       + 1j * delta

    I_path = 0.0 + 0.0j
    if abs(_ZETA_REF - p_mid1) > 1e-14:
        I_path += integrate_complex(_ZETA_REF, p_mid1, zk, betas, n_pts)
    if abs(p_mid1 - p_mid2) > 1e-14:
        I_path += integrate_complex(p_mid1, p_mid2, zk, betas, n_pts)
    if abs(p_mid2 - zeta) > 1e-14:
        I_path += integrate_complex(p_mid2, zeta, zk, betas, n_pts)

    return A + C * I_path


def sc_map(zeta_arr: np.ndarray, params: SCParameters, n_pts: int = 400) -> np.ndarray:
    return np.array([sc_map_single(z, params, n_pts) for z in zeta_arr])
