# APPM 4360 Project — Key Technical Notes

---

## Polygon Simplification

**The problem:** Boulder's city boundary, loaded from a GIS shapefile, has thousands of GPS vertices. If we feed all of them into the Schwarz-Christoffel (SC) integral, two things go wrong:

1. **Computation explodes.** The SC integral has one term per vertex. Thousands of terms make every function evaluation extremely slow.
2. **SC Crowding.** When many vertices are squeezed close together on the boundary, their pre-images on the real axis are forced into an impossibly tight cluster. The integrand becomes nearly singular and numerical integration breaks down.

**The fix:** We simplify the boundary to just **11 vertices** using two steps:

- **Douglas-Peucker algorithm** — a standard GIS tool that removes vertices whose removal changes the polygon shape by less than a tolerance. Think of it as "keep only the corners that matter."
- **Interior-angle smoothing** — after Douglas-Peucker, some angles are still too sharp or too flat for the SC map to handle. We merge or adjust vertices until every interior angle stays within a safe range.

The result is a polygon that still faithfully traces Boulder's west mountains, downtown core, and eastern plains, but with only 11 corners — small enough for the SC integrals to be fast and stable.

---

## Pre-vertex Computation

**What is a pre-vertex?**

The SC map $f$ sends the upper half-plane $\mathbb{H}$ to the interior of a polygon. Under this map, the $n$ corners of the polygon correspond to $n$ special points on the real axis called **pre-vertices** $z_1 < z_2 < \cdots < z_n$.

By convention, three of them are fixed to pin down the overall scaling and rotation (this is the 3-parameter freedom guaranteed by the Riemann Mapping Theorem). That leaves $n - 3 = 8$ **free pre-vertices** that we must find.

**Why does their placement matter?**

The shape of the mapped polygon is completely determined by the spacing of the pre-vertices on the real axis. Get the spacing wrong and the map produces the wrong shape — the "Boulder" outline will be distorted.

**Goal:** Find the 8 free pre-vertex positions so that the side-length ratios of the mapped polygon exactly match Boulder's true boundary ratios.

---

### Softmax Parameterization

The pre-vertices must satisfy a strict ordering constraint: $0 < z_1 < z_2 < \cdots < z_8 < 1$ (after fixing the three anchor points).

Feeding a constrained parameter directly into an optimizer is awkward — the optimizer might step outside the feasible region and crash. Instead, we use a **softmax reparameterization**:

Given an unconstrained vector $p = (p_1, \ldots, p_8) \in \mathbb{R}^8$, we compute

$$z_k = \sum_{j=1}^{k} \frac{e^{p_j}}{\displaystyle\sum_{i} e^{p_i} + 1}$$

(softmax weights followed by a cumulative sum).

**Why does this work?**

- Each softmax weight is positive, so the cumsum is strictly increasing — ordering is automatic.
- All weights sum to less than 1, so every $z_k$ stays inside $(0, 1)$ — the box constraint is automatic.
- The optimizer can now roam freely over all of $\mathbb{R}^8$ with no projection or penalty needed.

---

### Levenberg-Marquardt Least Squares

We need to find the 8 unconstrained parameters $p$ such that the 10 side lengths of the mapped polygon match Boulder's 10 target ratios. This is a nonlinear least-squares problem.

**Algorithm:** Levenberg-Marquardt (LM) iterates a "guess-compare-correct" loop:

1. Start with an initial guess for $p$.
2. Compute the current pre-vertices via softmax, then evaluate the 10 SC side-length integrals.
3. Compare computed side ratios to Boulder's target ratios — form a residual vector.
4. Compute a correction step using the Jacobian of the residuals, and update $p$.
5. Repeat until the residual is small (our solver runs ~18,000 iterations, reaching cost $\approx 6 \times 10^{-4}$).

**Why LM and not plain gradient descent?**

LM automatically blends two strategies:
- **Gauss-Newton** (fast, second-order) when the Jacobian is well-conditioned.
- **Gradient descent** (slow but stable) when the Jacobian is nearly singular — which happens near crowded pre-vertex configurations.

This makes LM much more robust than either method alone for our problem.

---

## Numerical Integration

Every LM iteration needs the SC side-length integrals — one integral per polygon side. Evaluating these "from scratch" at each iteration would dominate the runtime.

We approximate each integral with **500-node Gauss-Legendre (GL) quadrature**: place 500 specially chosen quadrature points on each real-axis interval $[z_k, z_{k+1}]$ and take a weighted sum of the integrand values. For smooth integrands, 500 GL nodes give essentially machine-precision accuracy.

Because the quadrature nodes are fixed at the start, this turns each integral evaluation into a single vectorized dot product — fast and reusable across iterations.

---

## Uniform Flow Produces Closed Streamlines

**Setup:** In the $\zeta$-plane (upper half-plane), we place **uniform flow**:

$$W(\zeta) = U \zeta$$

The streamlines of this flow are horizontal lines $\mathrm{Im}(\zeta) = y_0 = \text{const}$.

**The topological catch:** Each horizontal line in $\mathbb{H}$ extends from $\zeta \to -\infty$ to $\zeta \to +\infty$. On the Riemann sphere, both ends of this line are the **same point** (the "north pole," or point at infinity). The SC map sends that single point to one fixed vertex on the Boulder polygon.

Therefore every horizontal line — which is topologically a circle — maps to a **closed curve** inside Boulder. The flow circulates; it does not cross from one side of the city to the other.

**This is not a bug.** It is a mathematical necessity: any conformal map from $\mathbb{H}$ to a bounded polygon must collapse $\pm\infty$ to a single boundary point, so all streamlines of uniform flow become loops.

**To get crossing (wind-like) streamlines instead, two options:**

1. **Relabel the polygon** so that the "infinite side" (the one corresponding to $\zeta \in (-\infty, z_1) \cup (z_n, +\infty)$) spans the west and east boundaries of Boulder. Then the infinite ends of each streamline land on opposite sides and the flow crosses the domain.

2. **Switch to a source-sink potential:**

$$W(\zeta) = U \log\frac{\zeta - \zeta_{\mathrm{src}}}{\zeta - \zeta_{\mathrm{sink}}}$$

Place the source $\zeta_\mathrm{src}$ upstream and the sink $\zeta_\mathrm{sink}$ downstream. Streamlines now run from source to sink and cross the entire domain.

---

## Forward Mapping + Scattered Interpolation

**The naive approach** would be: for each pixel in the physical (Boulder) domain, invert the SC map to find the corresponding $\zeta$-plane point, then evaluate the complex potential there. But SC map inversion requires solving a nonlinear equation per pixel — very expensive.

**Our approach (forward map):**

1. Lay down a dense regular grid in the $\zeta$-plane (upper half-plane).
2. Push each $\zeta$-grid point **forward** through the SC map: compute $z = f(\zeta)$. This is cheap — just evaluate one SC integral per point.
3. We now have a cloud of scattered $(z, W)$ pairs in the physical domain.
4. Use **scattered-point interpolation** (`scipy.interpolate.griddata`) to recover $W$ on a regular pixel grid over Boulder.

**Why does this matter?**

The forward map naturally respects the conformal boundary: grid points from $\mathbb{H}$ land inside the polygon by construction, so the interpolated field is automatically zero outside the domain — no manual clipping or masking needed, and no staircasing artifacts at the boundary.

---

## Riemann Mapping Theorem

**What it says:** Any simply connected open domain in the complex plane (that is not the whole plane) can be conformally mapped to the upper half-plane $\mathbb{H}$.

**What it guarantees for us:** The interior of Boulder's polygon is simply connected, so a conformal map $f: \mathbb{H} \to \Omega_\text{Boulder}$ is guaranteed to exist. We do not have to prove the map exists — the theorem does it. We only have to find it numerically, which is what the SC parameter-solving step does.

"Conformal" means the map preserves angles locally: if two curves meet at 90° in $\mathbb{H}$, their images meet at 90° inside Boulder. This is exactly what makes the physical flow field correct: perpendicularity of streamlines and equipotential lines is preserved by the map.

---

## Boundary Conditions: Method of Images

**The goal:** The stream function $\psi = \mathrm{Im}(W)$ must equal zero on the Boulder boundary — this enforces the no-penetration condition (wind cannot pass through walls).

**In $\mathbb{H}$**, the boundary is the real axis $\mathrm{Im}(\zeta) = 0$. We need $\mathrm{Im}(W) = 0$ there.

**Method of Images:** For every singularity (vortex, source, sink) placed at a point $\zeta_0$ in the upper half-plane, we place a mirror-image singularity of opposite sign at $\bar{\zeta}_0$ in the lower half-plane. The imaginary parts of the two contributions cancel exactly on the real axis by symmetry — so $\psi = 0$ on $\mathrm{Im}(\zeta) = 0$ is automatic.

Because the SC map is conformal, $\psi = 0$ on the real axis in $\mathbb{H}$ pulls back to $\psi = 0$ on the Boulder boundary in the physical domain. No numerical boundary enforcement is needed at all.

---

## Urban Obstacles: Milne-Thomson Circle Theorem

**Motivation:** Downtown Boulder has dense built-up blocks that wind cannot penetrate. We model each such block as a circular obstacle of radius $a$ centered at $\zeta_0$.

**The theorem:** Given any complex potential $W_0(\zeta)$ for flow in an unbounded plane, the modified potential

$$W(\zeta) = W_0(\zeta) + \overline{W_0\!\left(\zeta_0 + \frac{a^2}{\bar{\zeta} - \bar{\zeta}_0}\right)}$$

produces flow that is identical to $W_0$ far from the circle, but has the circle $|\zeta - \zeta_0| = a$ as an impenetrable streamline ($\psi = \mathrm{const}$ on the boundary).

**Intuition:** The second term is constructed from the "image" of $W_0$ reflected through the circle (via the inversion $\zeta \mapsto \zeta_0 + a^2/(\bar{\zeta} - \bar{\zeta}_0)$). This reflection is the circle-geometry analogue of the Method of Images for a flat wall.

**Why this is powerful:** The obstacle is inserted entirely at the formula level — no grid refinement, no finite-element mesh, no discrete boundary conditions. The resulting flow field is still analytic everywhere outside the disk, and the no-penetration condition is exact (not approximate).
