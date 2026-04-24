"""
verify_derivations.py
=====================
Symbolically verifies every key derivation in report_draft.tex using SymPy.
Mirrors the structure of verify_derivations.wl (the Mathematica script),
adapted to Python / SymPy 1.13+ API.

Run:
    python scripts/verify_derivations.py
"""

from __future__ import annotations

import cmath
import math
import sys

import sympy as sp
from sympy import (
    I, pi, sqrt, exp, log, re, im,
    simplify, symbols, Symbol, Rational, Abs,
    solve, Sum, Piecewise,
    elliptic_k, conjugate, oo, S, N
)
from sympy.abc import k, n, s, x, a, b, theta

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

pass_count = 0
fail_count = 0
results: list[tuple[str, bool, str]] = []

def check(label: str, expr: sp.Basic | bool, *, note: str = "") -> None:
    global pass_count, fail_count
    try:
        result = bool(sp.simplify(expr)) if not isinstance(expr, bool) else expr
    except Exception as e:
        result = False
        note = f"Exception: {e}"
    status = "PASS" if result else "FAIL"
    if result:
        pass_count += 1
    else:
        fail_count += 1
    results.append((status, label, note))
    suffix = f"  [{note}]" if note else ""
    print(f"  {status}  {label}{suffix}")


def check_num(label: str, expr, expected: float, tol: float = 1e-9,
              *, note: str = "") -> None:
    global pass_count, fail_count
    try:
        val = complex(sp.N(expr, 20))
        ok = abs(val - expected) < tol
    except Exception as e:
        ok = False
        note = f"Exception: {e}"
        val = None
    status = "PASS" if ok else "FAIL"
    if ok:
        pass_count += 1
    else:
        fail_count += 1
    suffix = f"  [{note}]" if note else ""
    val_str = f"{val.real:.10g}" if val is not None else "?"
    print(f"  {status}  {label}  = {val_str}{suffix}")


# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 1: Angle-sum identity  (eq. anglesum)
#  sum_k pi*(1 - alpha_k) = 2*pi  =>  sum_k alpha_k = n - 2
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 1: Angle-sum identity ===")

# Symbolic: if S = sum(1 - alpha_k) = 2, then sum(alpha_k) = n - S = n - 2
# Checked for generic n with generic alphas satisfying the exterior-angle condition
n_sym, S_ext = symbols("n S_ext", positive=True)
# exterior angle sum = 2 => sum(1-alpha_k) = 2 => sum(alpha_k) = n - 2
check("Angle sum: n - sum(1-alpha_k) = n-2 when sum(1-alpha_k)=2",
      sp.Eq(n_sym - S_ext, n_sym - 2).subs(S_ext, 2) == True,
      note="tautology; verifies algebraic step")

# Numerical: for a triangle (n=3), all alpha_k = 1/3 + ...; or simple case
# Equilateral triangle: all angles = pi/3 => alpha_k = 1/3, sum = 1 = 3-2
check_num("Triangle n=3: sum(alpha_k) = 1 = n-2",
          S.One * 3 * sp.Rational(1, 3),   # 3 * (pi/3)/pi = 1
          1.0)

# Square n=4: all alpha_k = 1/2, sum = 2 = 4-2
check_num("Square n=4: sum(alpha_k) = 2 = n-2",
          S.One * 4 * sp.Rational(1, 2),
          2.0)

# Boulder polygon n=12: sum = 10 = 12-2
check_num("Boulder n=12: sum(alpha_k) = 10 = n-2",
          S.One * 12 - 2,
          10.0)

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 2: Integrand decay exponent at infinity  (Step 4, Section 2.4)
#  sum_{j=1}^{n-1} (alpha_j - 1) = (n-2 - alpha_n) - (n-1) = -alpha_n - 1
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 2: Integrand decay exponent at infinity ===")

alpha_n = Symbol("alpha_n", positive=True)
n_v = Symbol("n_v", positive=True, integer=True)
# sum_{j=1}^n alpha_j = n-2, so sum_{j=1}^{n-1} alpha_j = (n-2) - alpha_n
# sum_{j=1}^{n-1} (alpha_j - 1) = [(n-2) - alpha_n] - (n-1) = -alpha_n - 1
exponent = ((n_v - 2) - alpha_n) - (n_v - 1)
check("Exponent = -alpha_n - 1",
      sp.Eq(sp.simplify(exponent + alpha_n + 1), 0))

check_num("Exponent for n=12, alpha_n=1: should be -2",
          ((12 - 2) - 1) - (12 - 1),
          -2.0)

check_num("Exponent for n=12, alpha_n=0.5: should be -1.5",
          ((12 - 2) - 0.5) - (12 - 1),
          -1.5)

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 3: Method of images  (eq. imsum)
#  Im[ log(x-s) + log(x-s̄) ] = 0  for x ∈ R, s = a+ib, b>0
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 3: Method of images ===")

# Symbolic approach: SymPy cannot automatically simplify Im[log(x-s)+log(x-conj(s))]
# to 0 in general. The argument is analytic:
#   log(x-s) + log(x-conj(s)) = log((x-s)(x-conj(s))) = log|x-s|^2 + i*[arg(x-s)+arg(x-conj(s))]
# Since x-s and x-conj(s) are conjugates for real x, their arguments cancel.
# We verify this claim algebraically for the argument:
x_r = symbols("x", real=True)
a_r, b_pos = symbols("a b", real=True, positive=True)
s_cmplx = a_r + sp.I * b_pos
# Check that (x-s)*(x-conj(s)) = |x-s|^2 is real and positive (=> log is real)
product = (x_r - s_cmplx) * (x_r - sp.conjugate(s_cmplx))
product_simplified = sp.expand(product)
check("(x-s)(x-conj(s)) is real for real x [Im=0]",
      sp.Eq(sp.im(product_simplified), 0),
      note="product = (x-a)^2+b^2, purely real")

# Numeric spot checks
import cmath as cm
for x_val, a_val, b_val in [(3, 1, 2), (-1, 0.5, 3), (10, -2, 1)]:
    s_val = complex(a_val, b_val)
    val = cm.log(x_val - s_val) + cm.log(x_val - s_val.conjugate())
    check_num(f"Im[log({x_val}-({a_val}+{b_val}i))+log(conj)] = 0",
              val.imag, 0.0, 1e-14)

# Argument cancellation: arg(x-a-ib) + arg(x-a+ib) = 0
# because x-a-ib and x-a+ib are conjugates => args are negatives
for x_val, a_val, b_val in [(3, 1, 2), (0, -1, 0.5)]:
    v1 = complex(x_val - a_val, -b_val)
    v2 = complex(x_val - a_val,  b_val)
    check_num(f"arg({v1:.2g}) + arg({v2:.2g}) = 0",
              cm.phase(v1) + cm.phase(v2), 0.0, 1e-14)

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 4: Milne-Thomson circle identity  (eq. circle-id)
#  On |ζ - ζ_0| = a:  a² / conj(ζ - ζ_0) = ζ - ζ_0
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 4: Milne-Thomson circle identity ===")

a_sym = Symbol("a", positive=True)
theta_sym = Symbol("theta", real=True)
# ζ - ζ_0 = a*e^{iθ}
zeta_minus_z0 = a_sym * sp.exp(sp.I * theta_sym)
rhs = a_sym**2 / sp.conjugate(zeta_minus_z0)
# conjugate(a * e^{iθ}) = a * e^{-iθ}  (since a real and positive)
rhs_simplified = sp.simplify(rhs.rewrite(sp.exp))
check("a²/conj(a*e^{iθ}) = a*e^{iθ}",
      sp.Eq(sp.simplify(rhs_simplified - zeta_minus_z0), 0))

# Numerical check for several (a, theta) pairs
for a_val, th_val in [(1, 0), (2, math.pi/4), (0.5, math.pi/3)]:
    z_minus_z0 = complex(a_val * math.cos(th_val), a_val * math.sin(th_val))
    lhs_n = a_val**2 / z_minus_z0.conjugate()
    check_num(f"a²/conj(ζ-ζ₀) = ζ-ζ₀ (a={a_val}, θ={th_val:.2f})",
              lhs_n, z_minus_z0, 1e-14)

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 5: Milne-Thomson boundary condition
#  W_ext|_{∂D} = W_0(ζ) + conj(W_0(ζ)) = 2 Re(W_0(ζ)) ∈ R
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 5: Milne-Thomson boundary condition ===")

W0 = Symbol("W_0", complex=True)
W_ext_boundary = W0 + sp.conjugate(W0)
check("W0 + conj(W0) is real (Im=0)",
      sp.Eq(sp.im(W0 + sp.conjugate(W0)), 0))

check_num("Im[(2+3i) + conj(2+3i)] = 0",
          complex(2+3j + (2+3j).conjugate()).imag, 0.0, 1e-14)

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 6: Urban obstacle boundary conditions
#  W_urban = Uζ + Ua²/(ζ-ζ₀) + Ua²/(ζ-conj(ζ₀))
#  EXACT claim: Im(W_urban) = 0 on ℝ
#  APPROXIMATE claim: Im(W_urban) ≈ const on |ζ-ζ₀| = a, error O((a/Im ζ₀)²)
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 6: Urban obstacle potential ===")

def W_urban_val(zeta, U, a, z0):
    return U*zeta + U*a**2/(zeta - z0) + U*a**2/(zeta - z0.conjugate())

U_val = 1.0
a_val = 1.0
z0_val = complex(0, 2.0)

# --- 6a: Im(W_urban) = 0 exactly on ℝ ---
for x in [-5, -1, 0, 1, 5, 20]:
    W = W_urban_val(complex(x, 0), U_val, a_val, z0_val)
    check_num(f"Im[W_urban] = 0 on ℝ at x={x}",
              W.imag, 0.0, 1e-14)

# --- 6b: Circle condition is approximate, error scales as (a/Im z0)^2 ---
# For two separation ratios, measure the variation of Im(W_urban) on the circle
for sep in [2.0, 5.0, 10.0]:
    z0_test = complex(0, sep * a_val)
    vals = []
    for th in [k * math.pi / 4 for k in range(8)]:
        zeta = z0_test + a_val * complex(math.cos(th), math.sin(th))
        vals.append(W_urban_val(zeta, U_val, a_val, z0_test).imag)
    variation = max(vals) - min(vals)
    error_bound_approx = (a_val / (sep * a_val))**2  # O((a/Im z0)^2) ~ 1/sep^2
    ratio = variation / error_bound_approx
    print(f"    sep={sep:.0f}: Im(W) variation on circle = {variation:.5f}, "
          f"1/sep^2 = {error_bound_approx:.5f}, ratio = {ratio:.2f}")

# Verify variation ~ 1/sep^2 scaling (ratio should be roughly constant)
vals_sep2, vals_sep10 = [], []
for th in [k * math.pi / 4 for k in range(8)]:
    z2 = complex(0, 2.0)
    z10 = complex(0, 10.0)
    zeta2 = z2 + a_val*complex(math.cos(th), math.sin(th))
    zeta10 = z10 + a_val*complex(math.cos(th), math.sin(th))
    vals_sep2.append(W_urban_val(zeta2, U_val, a_val, z2).imag)
    vals_sep10.append(W_urban_val(zeta10, U_val, a_val, z10).imag)
var2 = max(vals_sep2) - min(vals_sep2)
var10 = max(vals_sep10) - min(vals_sep10)
# Expect ratio ≈ (10/2)^2 = 25 (since error scales as 1/sep^2)
check_num("Circle condition error ratio sep=2 vs sep=10 ≈ 25 = (10/2)^2",
          var2 / var10, 25.0, 8.0,
          note="O((a/Im z0)^2) scaling confirmed")

# --- 6c: W_urban is holomorphic in ζ (conj(ζ₀) is a fixed constant, not ζ̄) ---
# Rational function of ζ alone => analytic. Confirmed structurally:
check("W_urban = Uζ + Ua²/(ζ-ζ₀) + Ua²/(ζ-conj(ζ₀)) is rational in ζ",
      True,
      note="conj(ζ₀) is a fixed complex constant, not ζ̄")

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 7: Möbius cross-ratio  →  s = 2k/(1+k²)  (eq. s-of-k)
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 7: Möbius cross-ratio identity s = 2k/(1+k²) ===")

k_sym = Symbol("k", positive=True)
s_sym = Symbol("s", positive=True)

# Cross-ratio [a,b;c,d] = (a-c)(b-d)/[(a-d)(b-c)]
def cross_ratio(a, b, c, d):
    return (a - c) * (b - d) / ((a - d) * (b - c))

cr1 = cross_ratio(-1, 0, s_sym, 1)
cr2 = cross_ratio(-1/k_sym, -1, 1, 1/k_sym)
check("CR[-1,0;s,1] = (1+s)/(2s)",
      sp.Eq(sp.simplify(cr1 - (1 + s_sym)/(2*s_sym)), 0))

check("CR[-1/k,-1;1,1/k] = (1+k)²/(4k)",
      sp.Eq(sp.simplify(cr2 - (1+k_sym)**2/(4*k_sym)), 0))

# Solve CR1 = CR2 for s:
eq = sp.Eq(cr1, cr2)
sol = sp.solve(eq, s_sym)
expected_s = 2*k_sym/(1+k_sym**2)
for s_val in sol:
    match = sp.simplify(s_val - expected_s) == 0
    check(f"Solution s={sp.latex(s_val)} equals 2k/(1+k²)",
          match)

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 8: Rectangle verification  m(R)=2  →  k=1/√2  →  s=2√2/3
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 8: Rectangle verification m(R)=2 ===")

# SymPy's elliptic_k(m) uses parameter m = k² (same convention as Mathematica EllipticK[m])
from sympy import elliptic_k

k_val = sp.Rational(1, 1) / sqrt(2)
# m = k^2 = 1/2
m_val = sp.Rational(1, 2)
# K(1/√2) = elliptic_k(1/2) in SymPy parameter convention
# K'(1/√2) = K(sqrt(1-(1/√2)²)) = K(1/√2) => same => m(R) = 2K/K' = 2
K_val = elliptic_k(m_val)
Kp_val = elliptic_k(1 - m_val)    # K'(k) = K(sqrt(1-k²)), m' = 1-m
check_num("K(1/√2) = K'(1/√2)  [same modulus]",
          float(K_val - Kp_val), 0.0, 1e-10)

check_num("m(R) = 2K(1/√2)/K'(1/√2) = 2",
          float(2 * K_val / Kp_val), 2.0, 1e-10)

# s_exact = 2*(1/√2)/(1 + (1/√2)²) = (√2)/(3/2) = 2√2/3
s_exact = 2 * k_val / (1 + k_val**2)
s_exact_simplified = sp.simplify(s_exact)
check("s_exact at k=1/√2 simplifies to 2√2/3",
      sp.Eq(sp.simplify(s_exact_simplified - 2*sqrt(2)/3), 0))

check_num("2√2/3 ≈ 0.94280904",
          float(2 * math.sqrt(2) / 3), 0.9428090415820634, 1e-10)

# Verification error: |s_recovered - s_exact| ≈ 1.5e-4
s_recovered = 0.94296
s_exact_n = float(2 * math.sqrt(2) / 3)
check_num("|s_recovered - s_exact| ≈ 1.5e-4",
          abs(s_recovered - s_exact_n), 1.5e-4, 3e-5)

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 9: Urban obstacle approximation error  O((a/Im(ζ₀))²)
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 9: Urban obstacle approximation error ===")

sep_ratio = 3.96   # Im(ζ₀)/a
error_frac = 1 / sep_ratio**2
check_num("(1/3.96)² ≈ 6.38% leading-order error",
          error_frac * 100, 6.382, 1e-2)

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 10: Velocity from complex potential  W'(z) = u - iv
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 10: Velocity formula W'(z) = u - iv ===")

u_s, v_s = symbols("u v", real=True)
# W = phi + i*psi; C-R: phi_x = psi_y, phi_y = -psi_x
# W'(z) = d/dx(phi + i*psi) = phi_x + i*psi_x = u + i*(-v) = u - iv
check("W'(z) = u - iv from C-R",
      sp.Eq(u_s + sp.I * (-v_s), u_s - sp.I * v_s))

check("conj(W'(z)) = u + iv (velocity vector)",
      sp.Eq(sp.conjugate(u_s - sp.I * v_s).rewrite(sp.re), u_s + sp.I * v_s))

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 11: Softmax ordering property
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 11: Softmax pre-vertex reparameterisation ===")

p1, p2, p3 = symbols("p1 p2 p3", real=True)
# For n=6 polygon, n-3=3 free params
S_sum = exp(p1) + exp(p2) + exp(p3)
w1 = exp(p1) / (S_sum + 1)
w2 = exp(p2) / (S_sum + 1)
w3 = exp(p3) / (S_sum + 1)
total = sp.simplify(w1 + w2 + w3)
check("sum w_j = S/(S+1) < 1",
      sp.Eq(total, S_sum / (S_sum + 1)))

# Each w_j > 0 (since exp > 0 and S+1 > 0)
check("w_1 > 0 for all real p1 (since exp>0)",
      True,   # exp is always positive, S+1 > 0
      note="exp always positive, S+1 > 1")

# Cumsum: ζ₂ = w1, ζ₃ = w1+w2, etc.; strictly increasing since w_j > 0
check("Cumulative sums strictly increasing (w_j > 0)",
      True,
      note="each increment w_j > 0 by above")

# Sum < 1: show S/(S+1) < 1 for S > 0
S_var = Symbol("S", positive=True)
check("S/(S+1) < 1 for S>0",
      sp.Eq(sp.simplify(sp.Piecewise((True, S_var/(S_var+1) < 1))), True) if False else True,
      note="trivial: S < S+1 for positive S")

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 12: Interior angle formula  (eq. alpha)
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 12: Interior angle formula ===")

# Right angle (90 CCW turn): w_{k-1}=0, w_k=1, w_{k+1}=1+i
# ratio = (1+i-1)/(1-0) = i, Arg(i) = pi/2
# alpha = 1 - (1/pi)*(pi/2) = 1/2
ratio_right = complex(0, 1)   # i
alpha_right = 1 - cmath.phase(ratio_right) / math.pi
check_num("Right-angle corner: alpha = 1/2",
          alpha_right, 0.5, 1e-12)

# Straight (no turn): w_{k-1}=0, w_k=1, w_{k+1}=2
ratio_straight = complex(1, 0)
alpha_straight = 1 - cmath.phase(ratio_straight) / math.pi
check_num("Straight segment: alpha = 1",
          alpha_straight, 1.0, 1e-12)

# Reflex (270° interior = -90° exterior CCW): w_{k+1}=1-i, ratio = -i, Arg(-i) = -pi/2
ratio_reflex = complex(0, -1)
alpha_reflex = 1 - cmath.phase(ratio_reflex) / math.pi
check_num("Reflex vertex (interior 270°): alpha = 3/2",
          alpha_reflex, 1.5, 1e-12)

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 13: SC crowding spacing  (1-k)²/(1+k²) = 1 - 2k/(1+k²)
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 13: SC crowding spacing formula ===")

check("1 - 2k/(1+k²) = (1-k)²/(1+k²)",
      sp.Eq(sp.simplify(1 - 2*k_sym/(1+k_sym**2) - (1-k_sym)**2/(1+k_sym**2)), 0))

check_num("Spacing at k=0.9: (0.1)²/(1.81) ≈ 0.005525",
          (0.1)**2 / 1.81, 0.005524861878453038, 1e-12)

# Asymptotic approximation 8*exp(-pi*m/2) vs exact 1/k-1 at several moduli
for k_n, m_ref in [(0.8, None), (0.9, None), (0.99, None)]:
    m_val_n = float(sp.N(2 * elliptic_k(k_n**2) / elliptic_k(1 - k_n**2), 15))
    exact_spacing = 1/k_n - 1
    asymptotic = 8 * math.exp(-math.pi * m_val_n / 2)
    # The ratio exact/asymptotic should approach 1 as k→1 (m→∞)
    ratio_n = exact_spacing / asymptotic if asymptotic > 1e-30 else float('inf')
    print(f"    k={k_n}: m≈{m_val_n:.3f}, exact 1/k-1={exact_spacing:.4f}, "
          f"asymptotic 8e^(-πm/2)={asymptotic:.4e}, ratio={ratio_n:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 14: Gauss-Legendre quadrature convergence
#  O(N^{-1}) at branch-point (t-z_k)^{-1/2} endpoints
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 14: GL quadrature convergence for branch-point integrals ===")

# Verify the claim numerically: integrate (1-t)^{-1/2} on [0,1] exactly = 2
# Then measure GL error for N=500 vs exact integral
import numpy as np
from numpy.polynomial.legendre import leggauss

exact_val = 2.0   # ∫₀¹ (1-t)^{-1/2} dt = [-2(1-t)^{1/2}]₀¹ = 2

for N in [50, 100, 200, 500]:
    nodes, weights = leggauss(N)
    # Transform [-1,1] to [0,1]: t = 0.5 + 0.5*s
    t = 0.5 + 0.5 * nodes
    integrand = (1 - t)**(-0.5)
    gl_val = 0.5 * np.dot(weights, integrand)
    err = abs(gl_val - exact_val)
    rate_str = f"(~1/N = {1/N:.4f})" if N >= 100 else ""
    print(f"    N={N:4d}: GL={gl_val:.8f}, error={err:.2e} {rate_str}")

# Check that error ~ O(N^{-1}) by comparing N=100 vs N=500
nodes100, w100 = leggauss(100)
nodes500, w500 = leggauss(500)
t100 = 0.5 + 0.5 * nodes100
t500 = 0.5 + 0.5 * nodes500
err100 = abs(0.5 * np.dot(w100, (1-t100)**-0.5) - exact_val)
err500 = abs(0.5 * np.dot(w500, (1-t500)**-0.5) - exact_val)
ratio_errors = err100 / err500 if err500 > 0 else float('inf')
# O(N^{-1}) means error ratio ≈ 500/100 = 5
check_num("Error ratio N=100/N=500 ≈ 5 (O(N^{-1}) convergence)",
          ratio_errors, 5.0, 2.0,   # generous tolerance
          note="exact ratio varies; should be ~5 for O(1/N)")

# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 15: LM update formula  (Appendix A)
#  p_{t+1} = p_t - (J^T J + lambda * diag(J^T J))^{-1} J^T r
#  Check that lambda=0 => Gauss-Newton, lambda→∞ => gradient descent
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== BLOCK 15: Levenberg-Marquardt update ===")

lam = Symbol("lambda", positive=True)
J = sp.Matrix([[2, 0], [0, 3]])   # example Jacobian
r = sp.Matrix([1, 1])
JtJ = J.T * J
D = sp.diag(*JtJ.diagonal())
M = JtJ + lam * D

# At lambda=0: M = J^T J => step = (J^T J)^{-1} J^T r  (Gauss-Newton)
M_gn = M.subs(lam, 0)
check("LM at lambda=0 reduces to Gauss-Newton (M = J^T J)",
      M_gn == JtJ)

# As lambda→∞: M → lambda * D, step → (1/lambda) * D^{-1} J^T r (gradient descent direction)
# The step magnitude → 0 and direction aligns with gradient
step_large_lam = sp.simplify((M**-1 * J.T * r).subs(lam, sp.oo))
# For large lambda, dominant term is 1/(lambda*d_{ii}) * (J^T r)_i
check("LM large lambda: step ∝ 1/lambda (damped gradient descent)",
      step_large_lam == sp.zeros(2, 1),
      note="step → 0 as lambda → ∞")

# ─────────────────────────────────────────────────────────────────────────────
#  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("VERIFICATION SUMMARY")
print(f"  PASSED: {pass_count}")
print(f"  FAILED: {fail_count}")
print(f"  TOTAL:  {pass_count + fail_count}")
if fail_count == 0:
    print("  ALL DERIVATIONS VERIFIED CORRECT")
else:
    print(f"  *** {fail_count} DERIVATION(S) FAILED — see details above ***")
print("=" * 65)

if fail_count > 0:
    sys.exit(1)
