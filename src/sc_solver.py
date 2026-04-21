"""
sc_solver.py
============
Schwarz-Christoffel parameter problem solver and forward-map evaluator.

    f(ζ) = A + C ∫₀^ζ  ∏ₖ (t − ζₖ)^(αₖ − 1)  dt

Vectorised with NumPy for performance.  The Möbius normalisation fixes
three pre-vertices; the remaining n − 3 are found by nonlinear least
squares on the side-length ratios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)


# ── Data container ────────────────────────────────────────────────────────

@dataclass
class SCParameters:
    """Solved SC mapping parameters."""
    zk: np.ndarray        # pre-vertices on ℝ, shape (n,)
    alphas: np.ndarray    # interior angles / π
    betas: np.ndarray     # exponents αₖ − 1
    A: complex
    C: complex
    z_poly: np.ndarray    # target polygon vertices (complex, normalised)


# ── Vectorised SC integrand ──────────────────────────────────────────────

def _sc_prod_real(t: np.ndarray, zk: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Evaluate ∏ₖ (t − ζₖ)^βₖ for real t (vectorised over t).

    t : (N,)   zk : (n,)   betas : (n,)   → result : (N,) complex
    """
    diffs = t[:, None] - zk[None, :]            # (N, n)
    signs = np.where(diffs < 0, np.pi, 0.0)     # phase correction
    log_abs = betas[None, :] * np.log(np.abs(diffs) + 1e-300)
    phases  = 1j * betas[None, :] * signs
    return np.prod(np.exp(log_abs + phases), axis=1)


def _sc_prod_complex(t: np.ndarray, zk: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Evaluate ∏ₖ (t − ζₖ)^βₖ for complex t (vectorised over t).

    Uses the principal branch of log.
    """
    diffs = t[:, None] - zk[None, :]            # (N, n) complex
    return np.prod(np.exp(betas[None, :] * np.log(diffs)), axis=1)


# ── Gauss-Legendre integration (vectorised) ──────────────────────────────

_GL_CACHE: dict[int, tuple] = {}

def _gl_nodes(n_pts: int):
    """Cached Gauss-Legendre nodes and weights."""
    if n_pts not in _GL_CACHE:
        _GL_CACHE[n_pts] = np.polynomial.legendre.leggauss(n_pts)
    return _GL_CACHE[n_pts]


def integrate_real(a: float, b: float,
                   zk: np.ndarray, betas: np.ndarray,
                   n_pts: int = 500) -> complex:
    """∫_a^b ∏ₖ (t−ζₖ)^βₖ dt   along the real axis."""
    nodes, weights = _gl_nodes(n_pts)
    mid  = 0.5 * (a + b)
    half = 0.5 * (b - a)
    t = mid + half * nodes                       # (n_pts,)
    vals = _sc_prod_real(t, zk, betas)           # (n_pts,) complex
    return complex(half * np.dot(weights, vals))


def integrate_complex(za: complex, zb: complex,
                      zk: np.ndarray, betas: np.ndarray,
                      n_pts: int = 400) -> complex:
    """∫_{za}^{zb} ∏ₖ (t−ζₖ)^βₖ dt   along a straight line in ℂ."""
    nodes, weights = _gl_nodes(n_pts)
    mid  = 0.5 * (za + zb)
    half = 0.5 * (zb - za)
    t = mid + half * nodes                       # (n_pts,) complex
    vals = _sc_prod_complex(t, zk, betas)        # (n_pts,) complex
    return complex(half * np.dot(weights, vals))


# ── Side lengths (for the parameter problem) ─────────────────────────────

def _side_length(zk_a: float, zk_b: float,
                 zk: np.ndarray, betas: np.ndarray) -> float:
    """| ∫_{zk_a}^{zk_b} integrand dt |"""
    return abs(integrate_real(zk_a, zk_b, zk, betas))


def _all_side_lengths(zk: np.ndarray, betas: np.ndarray,
                      R: float = 5000.0) -> np.ndarray:
    """Compute side lengths for all n sides including the infinite one."""
    n = len(zk)
    lengths = np.empty(n)
    for i in range(n - 1):
        lengths[i] = _side_length(zk[i], zk[i + 1], zk, betas)
    # Last side: zk[n-1] → +∞ → −∞ → zk[0]
    seg1 = abs(integrate_real(zk[-1], zk[-1] + R, zk, betas))
    seg2 = abs(integrate_real(zk[0] - R, zk[0], zk, betas))
    lengths[n - 1] = seg1 + seg2
    return lengths


# ── SC parameter problem ─────────────────────────────────────────────────

def solve_parameters(
    z_poly: np.ndarray,
    alphas: np.ndarray,
    *,
    maxiter: int = 2000,
    tol: float = 1e-10,
) -> SCParameters:
    """Solve the SC parameter problem.

    Möbius normalisation: ζ₀ = −1, ζ₁ = 0, ζ_{n−1} = 1.
    Free parameters: ζ₂, …, ζ_{n−2}  (must satisfy −1 < ζ₀ < ζ₁ < … < ζ_{n−1} = 1).
    """
    n = len(z_poly)
    betas = alphas - 1.0

    # Target side-length ratios
    target_sides = np.abs(np.diff(np.append(z_poly, z_poly[0])))
    target_ratios = target_sides[:-1] / target_sides[-1]

    # Fixed pre-vertices
    fixed_vals = {0: -1.0, 1: 0.0, n - 1: 1.0}
    free_idx = [i for i in range(n) if i not in fixed_vals]
    n_free = len(free_idx)

    if n_free == 0:
        zk = np.array([-1.0, 0.0, 1.0])
    else:
        # Softmax parameterisation: p ∈ ℝⁿ_free (unconstrained).
        # Maps to n_free strictly-ordered values in (0, 1) via:
        #   w = [exp(p₀), …, exp(p_{m-1}), 1]   (m+1 weights, last fixed)
        #   gaps = w / sum(w)                     (sum to 1, all positive)
        #   ζ_free[i] = cumsum(gaps)[i]           (strictly increasing in (0,1))
        # Initial p=0 → equal spacing: ζ_free = [1/(m+1), …, m/(m+1)].
        # This gives a smooth, everywhere-differentiable residual (no sort).

        def _softmax_to_zk_free(p: np.ndarray) -> np.ndarray:
            w = np.append(np.exp(p - p.max()), 1.0)   # numerically stable
            w /= w.sum()
            return np.cumsum(w[:n_free])               # n_free values in (0, 1)

        # Side-length-proportional init to reduce crowding.
        # The n_free+1 softmax weights correspond to polygon sides 1..n-2.
        init_sides = target_sides[1:n - 1]          # shape (n_free + 1,)
        ratios = np.maximum(init_sides[:-1] / init_sides[-1], 1e-6)
        x0 = np.log(ratios)     # last weight fixed at 1 (log=0)

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
        # p is unconstrained; use Levenberg-Marquardt for fastest convergence.
        result = optimize.least_squares(
            _residuals, x0,
            method="lm",
            max_nfev=maxiter * 100,
            ftol=tol, xtol=tol, gtol=tol,
        )
        logger.info("SC solver initial run: cost=%.2e", result.cost)

        # Multi-start restart: if cost is poor, perturb and retry up to 5 times.
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

    # Determine A and C
    A, C = _solve_AC(zk, betas, z_poly)

    params = SCParameters(zk=zk, alphas=alphas, betas=betas,
                          A=A, C=C, z_poly=z_poly)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    logger.info("=== SC solver diagnostics ===")
    logger.info("pre-vertices zk = %s", np.round(zk, 6))
    logger.info("A = %.6g%+.6gj   |C| = %.6g   arg(C) = %.4f rad",
                A.real, A.imag, abs(C), np.angle(C))
    # Side-vector check: C * real_integral(side k) should equal z_poly[k+1] - z_poly[k]
    for k in range(n - 1):
        I_side = integrate_real(zk[k], zk[k + 1], zk, betas)
        computed = C * I_side
        target   = z_poly[k + 1] - z_poly[k]
        err      = abs(computed - target)
        logger.info("  side %2d: |C*I - Δz| = %.4e  (|Δz|=%.4f, arg_ratio=%.3f°)",
                    k, err, abs(target), np.degrees(np.angle(computed/target)))

    return params


def _solve_AC(zk, betas, z_poly):
    """Determine C from real-axis side integrals, then A from a safe reference.

    Two-stage approach:

    Stage 1 — C from real-axis side integrals (no branch-point issues):
        C * ∫_{zk[k]}^{zk[k+1]} integrand dt  =  z_poly[k+1] − z_poly[k]
    Solved by least-squares over all n−1 finite sides.  The integrand on the
    real axis is evaluated with `_sc_prod_real` (accurate GL quadrature).

    Stage 2 — A from a safe real-axis midpoint:
        Pick mid = midpoint of the widest finite pre-vertex interval (well
        separated from all branch points).  Compute f(mid) by accumulating
        from the nearest polygon vertex along the real axis, then compute
        A = f(mid+i·h) − C · ∫_{_ZETA_REF}^{mid+i·h} integrand dt
    where h = 0.3 keeps the complex path far from the real-axis singularities.
    The vertical ascent from mid to mid+i·h is safe because mid is not a
    branch point.
    """
    n = len(z_poly)

    # ── Stage 1: C from real-axis side integrals ─────────────────────────
    sides_z = np.diff(np.append(z_poly, z_poly[0]))      # z_poly[k+1]−z_poly[k], shape (n,)
    sides_I = np.array(
        [integrate_real(zk[k], zk[k + 1], zk, betas) for k in range(n - 1)],
        dtype=complex,
    )

    # C * I_k = s_k  →  2-real-equation system per side, solve with lstsq
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

    # ── Stage 2: A from safe real-axis midpoint ───────────────────────────
    # Index of the widest finite interval (best numerical conditioning).
    gaps = np.diff(zk)                       # zk is sorted, shape (n,)
    k_safe = int(np.argmax(gaps[:-1]))       # exclude the infinite last gap
    mid_real = 0.5 * (zk[k_safe] + zk[k_safe + 1])

    # f(mid_real) via real-axis accumulation from vertex k_safe.
    f_mid_real = z_poly[k_safe] + C * integrate_real(zk[k_safe], mid_real, zk, betas)

    # Connect mid_real to mid_real+i·h via vertical ascent (safe, not a branch point).
    h = 0.3
    mid_above = mid_real + 1j * h
    I_ascent = integrate_complex(mid_real + 1e-9j, mid_above, zk, betas)
    f_mid_above = f_mid_real + C * I_ascent

    # A = f(mid_above) − C · ∫_{_ZETA_REF}^{mid_above} integrand dt
    # L-shaped complex path from _ZETA_REF = 0.5j to mid_above = mid_real + 0.3j.
    p1 = _ZETA_REF.real + 1j * h          # 0 + 0.3j  (same height as mid_above)
    p2 = mid_real        + 1j * h          # mid_real + 0.3j  (= mid_above)
    I_ref_to_mid = 0.0 + 0.0j
    if abs(_ZETA_REF - p1) > 1e-14:
        I_ref_to_mid += integrate_complex(_ZETA_REF, p1, zk, betas)
    if abs(p1 - p2) > 1e-14:
        I_ref_to_mid += integrate_complex(p1, p2, zk, betas)

    A = f_mid_above - C * I_ref_to_mid
    return A, C


# ── Forward map  f(ζ) ────────────────────────────────────────────────────

# Reference point in upper half-plane (avoids real-axis singularities).
# A = f(_ZETA_REF) is stored in SCParameters.A; all integrals are relative
# to this base so no branch-point arithmetic is ever needed.
_ZETA_REF = 0.0 + 0.5j


def sc_map_single(zeta: complex, params: SCParameters, n_pts: int = 400) -> complex:
    """Evaluate f(ζ) = A + C ∫_{_ZETA_REF}^{ζ} integrand dt.

    A = f(_ZETA_REF) is stored in params.A.
    Integration path stays in ℍ via an L-shaped route.
    """
    zk, betas, A, C = params.zk, params.betas, params.A, params.C

    # For points on or very near the real axis, lift slightly
    if abs(zeta.imag) < 1e-12:
        zeta = zeta.real + 1e-10j

    # L-shaped path from _ZETA_REF to ζ, staying at height ≥ min(Im(ζ), 0.3)
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
    """Evaluate the SC forward map on an array of complex points."""
    return np.array([sc_map_single(z, params, n_pts) for z in zeta_arr])
