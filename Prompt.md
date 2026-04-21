# APPM 4360 Project: Key Technical Notes

---

## Polygon Simplification

**The problem:** Boulder's city boundary, loaded from a GIS shapefile, has thousands of GPS vertices. Feeding all of them into the Schwarz-Christoffel (SC) integral causes two problems:

1. **Computation explodes.** The SC integral has one term per vertex. Thousands of terms make every function evaluation extremely slow.
2. **SC Crowding.** When many vertices are squeezed close together on the boundary, their pre-images on the real axis are forced into an impossibly tight cluster. The integrand becomes nearly singular and numerical integration breaks down.

**The fix:** We simplify the boundary to just **11 vertices** using two steps:

- **Douglas-Peucker algorithm**, a standard GIS tool that removes vertices whose removal changes the polygon shape by less than a tolerance. Think of it as "keep only the corners that matter."
- **Interior-angle smoothing**: after Douglas-Peucker, some angles are still too sharp or too flat for the SC map to handle. We merge or adjust vertices until every interior angle stays within a safe range.

The result is a polygon that still faithfully traces Boulder's west mountains, downtown core, and eastern plains, but with only 11 corners, small enough for the SC integrals to be fast and stable.

---

## Pre-vertex Computation

**What is a pre-vertex?**

The SC map $f$ sends the upper half-plane $\mathbb{H}$ to the interior of a polygon. Under this map, the $n$ corners of the polygon correspond to $n$ special points on the real axis called **pre-vertices** $z_1 < z_2 < \cdots < z_n$.

By convention, three of them are fixed to pin down the overall scaling and rotation (this is the 3-parameter freedom guaranteed by the Riemann Mapping Theorem). That leaves $n - 3 = 8$ **free pre-vertices** that we must find.

**Why does their placement matter?**

The shape of the mapped polygon is completely determined by the spacing of the pre-vertices on the real axis. A wrong spacing produces a distorted outline instead of Boulder's true boundary.

**Goal:** Find the 8 free pre-vertex positions so that the side-length ratios of the mapped polygon exactly match Boulder's true boundary ratios.

---

### Softmax Parameterization

The pre-vertices must satisfy a strict ordering constraint: $0 < z_1 < z_2 < \cdots < z_8 < 1$ (after fixing the three anchor points).

Feeding a constrained parameter directly into an optimizer is awkward, since the optimizer might step outside the feasible region and crash. Instead, we use a **softmax reparameterization**:

Given an unconstrained vector $p = (p_1, \ldots, p_8) \in \mathbb{R}^8$, we compute

$$z_k = \sum_{j=1}^{k} \frac{e^{p_j}}{\displaystyle\sum_{i} e^{p_i} + 1}$$

**Variables:**

| Symbol | Meaning | Units |
|--------|---------|-------|
| $p_j$ | $j$-th raw optimization parameter, unconstrained | dimensionless |
| $z_k$ | $k$-th free pre-vertex position on the real axis | dimensionless (normalized to $(0,1)$) |
| $\sum_i e^{p_i} + 1$ | normalization denominator (the $+1$ ensures all $z_k$ sum to less than 1) | dimensionless |

**Why does this work?**

Each softmax weight is positive, so the cumsum is strictly increasing, making ordering automatic. All weights sum to less than 1, so every $z_k$ stays inside $(0, 1)$, satisfying the box constraint automatically. The optimizer can roam freely over all of $\mathbb{R}^8$ with no projection or penalty needed.

---

### Levenberg-Marquardt Least Squares

We need to find the 8 unconstrained parameters $p$ such that the 10 side lengths of the mapped polygon match Boulder's 10 target ratios. This is a nonlinear least-squares problem.

**Algorithm:** Levenberg-Marquardt (LM) iterates a "guess-compare-correct" loop:

1. Start with an initial guess for $p$.
2. Compute the current pre-vertices via softmax, then evaluate the 10 SC side-length integrals.
3. Compare computed side ratios to Boulder's target ratios, forming a residual vector.
4. Compute a correction step using the Jacobian of the residuals, and update $p$.
5. Repeat until the residual is small (our solver runs ~18,000 iterations, reaching cost $\approx 6 \times 10^{-4}$).

**Why LM and not plain gradient descent?**

LM automatically blends two strategies:
- **Gauss-Newton** (fast, second-order) when the Jacobian is well-conditioned.
- **Gradient descent** (slow but stable) when the Jacobian is nearly singular, a situation that arises near crowded pre-vertex configurations.

This makes LM much more robust than either method alone for our problem.

---

## Numerical Integration

Every LM iteration needs the SC side-length integrals, one per polygon side. Evaluating these from scratch at each iteration would dominate the runtime.

We approximate each integral with **500-node Gauss-Legendre (GL) quadrature**: place 500 specially chosen quadrature points on each real-axis interval $[z_k, z_{k+1}]$ and take a weighted sum of the integrand values. For smooth integrands, 500 GL nodes give essentially machine-precision accuracy.

Because the quadrature nodes are fixed at the start, this turns each integral evaluation into a single vectorized dot product, fast and reusable across iterations.

---

## Uniform Flow Produces Closed Streamlines

**Setup:** In the $\zeta$-plane (upper half-plane), we place **uniform flow**:

$$W(\zeta) = U \zeta$$

**Variables:**

| Symbol | Meaning | Units |
|--------|---------|-------|
| $W = \phi + i\psi$ | complex potential; real part $\phi$ is the velocity potential, imaginary part $\psi$ is the stream function | m²/s |
| $\zeta = \xi + i\eta$ | complex coordinate in the mathematical upper half-plane $\mathbb{H}$ ($\eta > 0$); the "before-mapping" domain | dimensionless |
| $U$ | free-stream wind speed (the uniform flow amplitude) | m/s |

The streamlines of this flow are the curves where $\psi = \mathrm{Im}(W) = U\eta = \text{const}$, i.e., horizontal lines $\eta = y_0$.

**The topological catch:** Each horizontal line in $\mathbb{H}$ extends from $\zeta \to -\infty$ to $\zeta \to +\infty$. On the Riemann sphere, both ends of this line are the **same point** (the "north pole," or point at infinity). The SC map sends that single point to one fixed vertex on the Boulder polygon.

Therefore every horizontal line, topologically a circle, maps to a **closed curve** inside Boulder. The flow circulates; it does not cross from one side of the city to the other.

**This is a mathematical necessity.** Any conformal map from $\mathbb{H}$ to a bounded polygon must collapse $\pm\infty$ to a single boundary point, so all streamlines of uniform flow become loops.

**To get crossing (wind-like) streamlines instead, two options:**

1. **Relabel the polygon** so that the "infinite side" (the one corresponding to $\zeta \in (-\infty, z_1) \cup (z_n, +\infty)$) spans the west and east boundaries of Boulder. The infinite ends of each streamline then land on opposite sides and the flow crosses the domain.

2. **Switch to a source-sink potential:**

$$W(\zeta) = U \log\frac{\zeta - \zeta_{\mathrm{src}}}{\zeta - \zeta_{\mathrm{sink}}}$$

**Variables:**

| Symbol | Meaning | Units |
|--------|---------|-------|
| $\zeta_{\mathrm{src}}$ | location of the **source** in $\mathbb{H}$, the point where flow originates (enters the domain) | dimensionless |
| $\zeta_{\mathrm{sink}}$ | location of the **sink** in $\mathbb{H}$, the point where flow terminates (leaves the domain) | dimensionless |
| $U$ | source/sink strength, controls total volumetric flow rate | m²/s |
| $\log(\cdot)$ | complex natural logarithm; its imaginary part gives the stream function $\psi$, which counts the angle swept from source to sink | dimensionless inside log |

Streamlines run from $\zeta_{\mathrm{src}}$ to $\zeta_{\mathrm{sink}}$ and cross the entire domain, exactly the wind-passing-through-the-city behavior we want.

---

## Forward Mapping + Scattered Interpolation

**The naive approach** would be: for each pixel in the physical (Boulder) domain, invert the SC map to find the corresponding $\zeta$-plane point, then evaluate the complex potential there. SC map inversion requires solving a nonlinear equation per pixel, making it very expensive.

**Our approach (forward map):**

1. Lay down a dense regular grid in the $\zeta$-plane (upper half-plane).
2. Push each $\zeta$-grid point **forward** through the SC map: compute $z = f(\zeta)$. This is cheap, requiring just one SC integral per point.
3. We now have a cloud of scattered $(z, W)$ pairs in the physical domain.
4. Use **scattered-point interpolation** (`scipy.interpolate.griddata`) to recover $W$ on a regular pixel grid over Boulder.

**Variables:**

| Symbol | Meaning | Units |
|--------|---------|-------|
| $\zeta \in \mathbb{H}$ | grid point in the mathematical upper half-plane (before mapping) | dimensionless |
| $z = f(\zeta) \in \mathbb{C}$ | image of $\zeta$ under the SC map; a point inside the Boulder polygon | km (physical coordinates) |
| $W(\zeta)$ | complex potential evaluated at $\zeta$ (upstream, in the simple domain) | m²/s |

The forward map naturally respects the conformal boundary: grid points from $\mathbb{H}$ land inside the polygon by construction, so the interpolated field is automatically zero outside the domain, with smooth boundaries and no staircasing artifacts.

---

## Riemann Mapping Theorem

**What it says:** Any simply connected open domain in the complex plane (that is not the whole plane) can be conformally mapped to the upper half-plane $\mathbb{H}$.

**What it guarantees for us:** The interior of Boulder's polygon is simply connected, so a conformal map $f: \mathbb{H} \to \Omega_\text{Boulder}$ is guaranteed to exist. The theorem handles the existence question. We only have to find the map numerically, which is what the SC parameter-solving step does.

"Conformal" means the map preserves angles locally: if two curves meet at 90° in $\mathbb{H}$, their images meet at 90° inside Boulder. This is exactly what makes the physical flow field correct: perpendicularity of streamlines and equipotential lines is preserved by the map.

---

## Boundary Conditions: Method of Images

**The goal:** The stream function $\psi = \mathrm{Im}(W)$ must equal zero on the Boulder boundary, enforcing the no-penetration condition (wind cannot pass through walls).

**In $\mathbb{H}$**, the boundary is the real axis $\mathrm{Im}(\zeta) = 0$. We need $\mathrm{Im}(W) = 0$ there.

**Method of Images:** For every singularity (vortex, source, sink) placed at a point $\zeta_0$ in the upper half-plane, we place a mirror-image singularity of opposite sign at $\bar{\zeta}_0$ in the lower half-plane. The imaginary parts of the two contributions cancel exactly on the real axis by symmetry, so $\psi = 0$ on $\mathrm{Im}(\zeta) = 0$ is automatic.

**Variables:**

| Symbol | Meaning | Units |
|--------|---------|-------|
| $\psi = \mathrm{Im}(W)$ | stream function; its level curves are the streamlines; $\psi = 0$ on a wall means no flow crosses that wall | m²/s |
| $\zeta_0 \in \mathbb{H}$ | location of the physical singularity (above the real axis) | dimensionless |
| $\bar{\zeta}_0$ | complex conjugate of $\zeta_0$; the mirror image below the real axis | dimensionless |

Because the SC map is conformal, $\psi = 0$ on the real axis in $\mathbb{H}$ pulls back to $\psi = 0$ on the Boulder boundary in the physical domain.

---

## Urban Obstacles: Milne-Thomson Circle Theorem

**Motivation:** Downtown Boulder has dense built-up blocks that wind cannot penetrate. We model each such block as a circular obstacle of radius $a$ centered at $\zeta_0$.

**The theorem:** Given any complex potential $W_0(\zeta)$ for flow in an unbounded plane, the modified potential

$$W(\zeta) = W_0(\zeta) + \overline{W_0\!\left(\zeta_0 + \frac{a^2}{\bar{\zeta} - \bar{\zeta}_0}\right)}$$

produces flow that is identical to $W_0$ far from the circle, but has the circle $|\zeta - \zeta_0| = a$ as an impenetrable streamline ($\psi = \mathrm{const}$ on the boundary).

**Variables:**

| Symbol | Meaning | Units |
|--------|---------|-------|
| $W_0(\zeta)$ | original complex potential without the obstacle (e.g., uniform flow or source-sink) | m²/s |
| $W(\zeta)$ | modified complex potential with the circular obstacle inserted | m²/s |
| $\zeta_0$ | center of the circular obstacle in the $\zeta$-plane | dimensionless |
| $a$ | radius of the circular obstacle in the $\zeta$-plane | dimensionless (same scale as $\zeta$) |
| $\bar{\zeta}$ | complex conjugate of $\zeta$; flips the imaginary part sign | dimensionless |
| $\zeta_0 + \dfrac{a^2}{\bar{\zeta} - \bar{\zeta}_0}$ | **inversion** of $\zeta$ through the circle: maps every exterior point to a corresponding interior point; this is the "reflection" that enforces the no-penetration condition | dimensionless |
| $\overline{(\cdots)}$ | complex conjugate of the entire expression inside; together with the inversion, this ensures $\psi$ is constant on $|\zeta - \zeta_0| = a$ | — |

**Intuition:** The second term is constructed from the "image" of $W_0$ reflected through the circle via the inversion $\zeta \mapsto \zeta_0 + a^2/(\bar{\zeta} - \bar{\zeta}_0)$. This reflection is the circle-geometry analogue of the Method of Images for a flat wall.

**Why this is powerful:** The obstacle is inserted entirely at the formula level. The resulting flow field remains analytic everywhere outside the disk, and the no-penetration condition is exact.
