# APPM 4360 Project — Key Technical Notes

## Polygon Simplification

Boulder's original boundary contains thousands of vertices. Running SC integrals on the raw boundary would be computationally explosive and introduce severe **SC Crowding** errors. We apply the **Douglas-Peucker algorithm** combined with interior-angle smoothing to reduce the boundary to an **11-vertex polygon**.

---

## Pre-vertex Computation

**Goal:** Locate $n - 3$ free pre-vertices on the real axis so that the side-length ratios of the mapped polygon exactly match the true Boulder boundary.

### Softmax Parameterization

We reparameterize the free pre-vertices using the formula

$$\frac{e^{p_j}}{\sum_j e^{p_j} + 1}$$

followed by a cumulative sum (cumsum). This maps an unconstrained real vector into a sequence that is automatically confined to $(0, 1)$ and strictly increasing — satisfying the ordering constraint without any explicit box projection, letting the optimizer explore freely.

### Levenberg-Marquardt Least Squares

The algorithm computes polygon side lengths from the current pre-vertices and matches them against the target Boulder ratios through a **guess-compare-correct** iteration loop (~18,000 iterations total). Key advantages:

- Automatically switches from Gauss-Newton to gradient-descent direction when the Jacobian approaches singularity.
- More robust convergence near ill-conditioned (crowded) configurations.

---

## Numerical Integration

Rather than evaluating a fresh SC integral at every iteration step, we approximate each side-length integral using **500-node Gauss-Legendre quadrature**, giving high accuracy at fixed cost.

---

## Uniform Flow Produces Closed Streamlines

Uniform flow $W = U\zeta$ has streamlines $\operatorname{Im}(\zeta) = y_0$ (horizontal lines) in the upper half-plane. Both ends of each horizontal line, $\zeta \to \pm\infty$, are the **same point on the Riemann sphere**. The SC map sends that point to a single fixed vertex on the polygon boundary, so every streamline becomes a **closed curve** inside the Boulder domain — corresponding to circulatory flow.

This is not a numerical error; it is a **topological consequence of mapping to a bounded polygon**.

To obtain crossing streamlines, one must either:

- Relabel the polygon so the infinite side spans the west and east boundaries, or
- Switch to a source-sink potential:

$$W = U \log\frac{\zeta - \zeta_{\mathrm{src}}}{\zeta - \zeta_{\mathrm{sink}}}$$

---

## Forward Mapping + Scattered Interpolation

To reduce computation, we use the **forward map**: lay down a grid in the $\zeta$-plane, push each grid point through the SC map into the physical domain, then use **scattered-point interpolation** (griddata) to recover values on a regular pixel grid. This guarantees a perfectly smooth boundary without any discrete clipping artifacts.

---

## Riemann Mapping Theorem

The **Riemann Mapping Theorem** guarantees the existence of a conformal map that analytically deforms the flat upper half-plane into the shape of Boulder's boundary.

---

## Boundary Conditions: Method of Images

To enforce $\psi = 0$ on the Boulder boundary, we use the **Method of Images**: introducing mirror-image singularities so that the imaginary parts cancel algebraically on the real axis.

---

## Urban Obstacles: Milne-Thomson Circle Theorem

For urban built-up areas modeled as circular obstacles, we apply the **Milne-Thomson Circle Theorem**. Using the Laurent-series properties of the complex potential, we insert an analytically impenetrable circle directly at the formula level:

$$W(\zeta) = W_0(\zeta) + \frac{a^2}{\zeta - \zeta_0} \cdot \overline{W_0'\!\left(\zeta_0 + \frac{a^2}{\bar{\zeta} - \bar{\zeta}_0}\right)}$$

No discrete grid correction is needed — the obstacle is "conjured" entirely within the analytic expression.
