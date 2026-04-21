# Ideal Fluid Flow via the Schwarz-Christoffel Transformation

**Complex Variables and Applications - Spring 2026**  
Congyuan Zheng · Sophia Arany · Alexander Ingalls  
University of Colorado Boulder, Department of Applied Mathematics

---

## Overview

This project applies the **Schwarz-Christoffel (SC) conformal mapping** to simulate
steady, irrotational, incompressible fluid flow inside the Boulder, Colorado city
boundary polygon. Four progressively richer physical models are implemented:

| Model | Potential $W(\zeta)$ | Physical meaning |
|-------|----------------------|-----------------|
| Uniform flow | $U\zeta$ | Ideal parallel flow |
| Terrain-corrected | $U\zeta + \sum_k q_k \log(\zeta - s_k)$ | Slope-driven sources/sinks from real DEM data |
| Urban obstacle | $U\zeta + \frac{Ua^2}{\zeta - \zeta_0} + \frac{Ua^2}{\bar\zeta - \bar\zeta_0}$ | Circle-theorem no-penetration obstacle |
| Road-vortex | $U\zeta + \sum_k \frac{-i\Gamma_k}{2\pi}[\log(\zeta-s_k)+\log(\zeta-\bar s_k)]$ | Point vortices at OSM road intersections |

The SC map $f : \mathbb{H} \to \Omega$ sends the upper half-plane to the Boulder polygon,
converting analytically tractable potentials in $\mathbb{H}$ into streamlines and
equipotentials in the physical domain.

---

## Mathematical Framework

### Schwarz-Christoffel Mapping

For a polygon with $n$ vertices and interior angles $\alpha_k\pi$, the SC map from $\mathbb{H}$ to $\Omega$ is

$$f(\zeta) = A + C \int_{\zeta_0}^{\zeta} \prod_{k} (t - \zeta_k)^{\alpha_k - 1} \, dt$$

| Symbol | Meaning | Units |
|--------|---------|-------|
| $f(\zeta)$ | SC map; outputs a point $z \in \Omega$ in the physical Boulder domain | km (physical coordinates) |
| $\zeta = \xi + i\eta$ | complex coordinate in the upper half-plane $\mathbb{H}$ ($\eta > 0$); the mathematical input domain | dimensionless |
| $A \in \mathbb{C}$ | translation constant; shifts the entire mapped polygon to the correct physical position | km |
| $C \in \mathbb{C}$ | scaling-and-rotation constant; sets the size and orientation of the mapped polygon | km |
| $t$ | integration variable along the path from $\zeta_0$ to $\zeta$ in $\mathbb{H}$ | dimensionless |
| $\zeta_k \in \mathbb{R}$ | $k$-th pre-vertex; the real-axis point that maps to the $k$-th polygon corner | dimensionless |
| $\alpha_k$ | interior angle of the $k$-th polygon corner, expressed as a fraction of $\pi$ (so the actual angle is $\alpha_k \pi$ radians); must satisfy $\sum_k (1 - \alpha_k) = 2$ | dimensionless |
| $\alpha_k - 1$ | exponent in the integrand; controls how strongly the map bends near vertex $k$ | dimensionless |
| $n$ | total number of polygon vertices (11 for the simplified Boulder boundary) | — |

The **pre-vertices** $\zeta_k \in \mathbb{R}$ are unknowns. Möbius normalisation fixes three
($\zeta_0 = -1$, $\zeta_1 = 0$, $\zeta_{n-1} = 1$); the rest are found by Levenberg-Marquardt
nonlinear least-squares matching edge-length ratios of the mapped polygon to the target.

Streamlines are computed via the **forward map**: horizontal lines $\text{Im}(\zeta) = y_0$
in $\mathbb{H}$ are mapped forward as parametric curves $z(t) = f(t + iy_0)$, giving
exact artifact-free streamlines with no inverse-solver required.

---

### Terrain Correction

Elevation data (USGS 3DEP, 81 points) is fitted with a thin-plate-spline RBF. Each polygon
vertex $k$ contributes a source/sink pair in $\mathbb{H}$:

$$W_\text{terrain}(\zeta) = U\zeta + \sum_k \frac{q_k}{2\pi}
  \Bigl[\log(\zeta - s_k) + \log(\zeta - \bar{s}_k)\Bigr]$$

| Symbol | Meaning | Units |
|--------|---------|-------|
| $W = \phi + i\psi$ | complex potential; real part $\phi$ is the velocity potential, imaginary part $\psi$ is the stream function (its level curves are the streamlines) | m²/s |
| $U$ | free-stream wind speed; amplitude of the background uniform flow | m/s |
| $q_k$ | source/sink strength at vertex $k$; positive = source (air rises, outflow), negative = sink (air descends, inflow); magnitude proportional to elevation gradient at that vertex | m²/s |
| $s_k \in \mathbb{H}$ | pre-image of polygon vertex $k$ in the upper half-plane; obtained by numerically inverting the SC map | dimensionless |
| $\bar{s}_k$ | complex conjugate of $s_k$; the mirror image below the real axis, required by the **Method of Images** to enforce $\psi = 0$ (no flow through the boundary) on the real axis | dimensionless |
| $\log(\zeta - s_k)$ | complex logarithm; its imaginary part gives the angle swept from $s_k$ to $\zeta$, which is the stream function contribution of the source/sink | dimensionless |
| $\log(\zeta - s_k) + \log(\zeta - \bar{s}_k)$ | the image pair; their imaginary parts cancel exactly on the real axis ($\text{Im}(\zeta) = 0$), enforcing the no-penetration boundary condition | dimensionless |

The image term $\log(\zeta - \bar{s}_k)$ enforces $\psi = 0$ on $\mathbb{R}$ by the
method of images, since $\text{Im}[\log(\zeta - s_k) + \log(\zeta - \bar{s}_k)] = 0$
for real $\zeta$.

---

### Urban Obstacle (Circle Theorem)

The downtown core is treated as an impenetrable interior obstacle, making the domain doubly
connected. Its boundary in $\mathbb{H}$ is approximated by a circle (centre $\zeta_0$,
radius $a$). By the Milne-Thomson circle theorem:

$$W_\text{urban}(\zeta) = U\zeta + \frac{Ua^2}{\zeta - \zeta_0} + \frac{Ua^2}{\bar\zeta - \bar\zeta_0}$$

| Symbol | Meaning | Units |
|--------|---------|-------|
| $U\zeta$ | background uniform flow (same as the baseline model) | m²/s |
| $\zeta_0 \in \mathbb{H}$ | centre of the circular obstacle in the upper half-plane; its pre-image corresponds to the centroid of downtown Boulder | dimensionless |
| $a$ | radius of the circular obstacle in the $\zeta$-plane; controls how large the no-flow zone is | dimensionless |
| $\frac{Ua^2}{\zeta - \zeta_0}$ | dipole term introduced by the circle theorem; represents the flow that is "pushed around" the obstacle | m²/s |
| $\frac{Ua^2}{\bar\zeta - \bar\zeta_0}$ | image of the dipole in the lower half-plane; enforces $\psi = 0$ on the real axis (Boulder boundary) | m²/s |
| $\text{Im}(\zeta_0)/a$ | separation ratio; measures how far the obstacle centre is above the boundary relative to its own radius; our run achieves 5.36, giving ~3.5% error | dimensionless |

Accuracy is $O\!\left((a/\text{Im}\,\zeta_0)^2\right)$; the run below achieves
$\text{Im}(\zeta_0)/a = 5.36$, giving approximately 3.5% error.

---

### Road-Vortex Model (OSM Intersections)

Major road intersections generate local circulation in urban atmospheric flow (traffic
turbulence, building-induced channelling, heat-island convection). Each intersection is
treated as a point vortex in $\mathbb{H}$:

$$W_\text{road}(\zeta) = U\zeta + \sum_k \frac{-i\Gamma_k}{2\pi}
  \Bigl[\log(\zeta - s_k) + \log(\zeta - \bar{s}_k)\Bigr]$$

| Symbol | Meaning | Units |
|--------|---------|-------|
| $\Gamma_k$ | circulation strength of the $k$-th vortex; proportional to the degree (number of roads) of intersection $k$; positive = counter-clockwise (CCW), negative = clockwise (CW) | m²/s |
| $-i\Gamma_k / (2\pi)$ | complex coefficient of a point vortex; the factor $-i$ rotates the logarithm's contribution by 90°, turning a source/sink pattern into a swirling vortex pattern | m²/s |
| $s_k \in \mathbb{H}$ | pre-image of intersection $k$ in the upper half-plane; obtained by applying the SC inverse map to each OSM road node | dimensionless |
| $\bar{s}_k$ | mirror image of $s_k$ below the real axis (Method of Images); ensures $\psi = 0$ on the Boulder boundary | dimensionless |
| $\log(\zeta - s_k) + \log(\zeta - \bar{s}_k)$ | vortex-image pair; imaginary parts cancel on the real axis, enforcing no-penetration | dimensionless |

The image term $\log(\zeta - \bar{s}_k)$ restores $\psi = 0$ on $\mathbb{R}$ (no-penetration
on $\partial\Omega$). Intersection positions $s_k \in \mathbb{H}$ are obtained by applying the
SC inverse map to each OSM node. Circulation $\Gamma_k$ is proportional to node degree;
sign is positive (CCW) north of the polygon centroid and negative (CW) south, producing a
vortex-pair structure consistent with Boulder's prevailing westerly-flow shear.

---

## Pipeline

```bash
# Uniform flow only
python main.py --shapefile data/raw/tl_2025_08_place --grid 80

# Full pipeline: terrain + urban obstacle + road vortices
python main.py --shapefile data/raw/tl_2025_08_place --terrain --urban --roads --grid 80

# Road-vortex model only (fast, no USGS API calls)
python main.py --shapefile data/raw/tl_2025_08_place --roads --grid 80

# Quick demo (no shapefile needed)
python main.py --demo
```

| Stage | Detail | Value |
|-------|--------|-------|
| Raw polygon | TIGER/Line vertices | 1935 |
| Douglas-Peucker simplification | Tolerance | adaptive → ~14 vertices |
| Extreme-angle removal | $\alpha \notin [0.35\pi,\, 1.75\pi]$ dropped | ~14 → 11 vertices |
| SC parameter solve | LM residual cost | $\sim 1.5\times10^{-3}$ |
| Terrain elevation | USGS 3DEP sample points | 81 / 81 (100%) |
| Urban obstacle | OSM commercial polygons | 73 raw → 6-vertex, 0.13 km² |
| Circle separation | $\text{Im}(\zeta_0)/a$ | 5.36 |
| Road intersections | OSM primary/secondary/tertiary | 12 inside polygon, 11 vortices |

---

## Results

### Fig 1 - Boulder Boundary: Original vs. Simplified

![Boulder boundary original vs simplified](figures/fig1_polygon_comparison.png)

1935-vertex TIGER/Line boundary (left) vs. the Douglas-Peucker simplification after
angle smoothing (right). Vertices with interior angle outside $[0.35\pi,\, 1.75\pi]$
are removed iteratively to prevent SC crowding, leaving an 11-vertex polygon.

### Fig 2 - Streamlines ($\psi = \text{const}$)

![Streamlines uniform flow](figures/fig2_streamlines.png)

Level curves of the stream function under $W = U\zeta$. Each curve is the forward image
of a horizontal line $\text{Im}(\zeta) = y_0$ in $\mathbb{H}$. Flow enters from the
left and right boundary segments and converges at the top vertex (the conformal image
of $\zeta \to +\infty$).

### Fig 3 - Equipotential Lines ($\varphi = \text{const}$)

![Equipotentials uniform flow](figures/fig3_equipotentials.png)

Level curves of the velocity potential $\varphi = \text{Re}(W)$. Each curve is the
forward image of a vertical half-line $\text{Re}(\zeta) = x_0$ in $\mathbb{H}$.

### Fig 4 - Combined Streamlines & Equipotentials

![Combined flow uniform](figures/fig4_combined.png)

Overlay of Figs 2 and 3 (blue streamlines, red equipotentials). The two families form
the **conformal grid** — the image of a rectangular grid in $\mathbb{H}$ under $f$.
Orthogonality throughout the interior confirms the map is conformal.

### Fig 5 - Terrain-Informed Flow

![Terrain flow](figures/fig5_terrain_flow.png)

Streamlines (green) and equipotentials (amber) under the terrain-corrected potential.
USGS 3DEP elevation data (81 query points) is fitted with a thin-plate-spline RBF;
the gradient at each polygon vertex drives a source/sink in $\mathbb{H}$. Red triangles
mark the highest vertex (~1770 m, west side); blue triangles the lowest (~1570 m, east).
Streamlines shift visibly toward lower elevation compared to uniform flow.

### Fig 6 - Uniform vs. Terrain-Corrected (side-by-side)

![Flow comparison](figures/fig6_flow_comparison.png)

Direct comparison at identical contour levels. The terrain correction bends streamlines
eastward (downhill), reproducing the slope-driven drainage pattern of Boulder's terrain.

### Fig 7 - Urban Core as Interior Obstacle

![Urban flow](figures/fig7_urban_flow.png)

Doubly-connected flow: the downtown commercial core (OSM landuse query, 6-vertex convex
hull, 0.13 km²) is treated as an impenetrable obstacle. Its pre-image in $\mathbb{H}$
is a circle ($\zeta_0$, radius $a$); the Milne-Thomson circle theorem gives
$W = U\zeta + Ua^2/(\zeta-\zeta_0) + Ua^2/(\bar\zeta-\bar\zeta_0)$.
Streamlines visibly deflect around the purple obstacle.

### Fig 8 - Three-Way Comparison

![Three-way comparison](figures/fig8_three_way_comparison.png)

Side-by-side: uniform flow / terrain-corrected / urban obstacle at the same contour
levels. Each panel shows the progressive physical enrichment of the SC framework.

### Fig 9 - Road-Vortex Flow

![Road flow](figures/fig9_road_flow.png)

Point vortices placed at the 11 highest-degree OSM road intersections (primary through
tertiary roads) inside the polygon. Each intersection's pre-image in $\mathbb{H}$ is
obtained via the SC inverse map. Vortices north of the centroid spin CCW (orange
triangles), south spin CW (blue triangles), producing a shear pattern consistent with
Boulder's prevailing westerly flow. Streamlines show local eddies at each intersection.

### Fig 10 - Uniform vs. Road-Vortex (side-by-side)

![Road vs uniform](figures/fig10_road_vs_uniform.png)

Direct comparison: the road-vortex correction introduces organised local circulation
absent from the uniform baseline, particularly along the Broadway and 28th St corridors.

### Fig 11 - Four-Way Comparison (all models)

![Four-way comparison](figures/fig11_four_way_comparison.png)

Side-by-side: uniform flow / terrain-corrected / urban obstacle / road-vortex at the same
contour levels. Each panel shows the progressive physical enrichment of the SC framework,
from ideal parallel flow to a doubly-connected, terrain- and circulation-informed model.

---

## Project Structure

```
.
├── data/raw/               TIGER/Line shapefile (Boulder, CO)
├── src/
│   ├── polygon.py          Load, simplify, and smooth Boulder polygon
│   ├── angles.py           Interior-angle computation
│   ├── sc_solver.py        SC parameter problem and forward map
│   ├── flow.py             Forward map, parametric curves, and flow grid
│   ├── terrain.py          DEM elevation (RBF) and per-vertex sources
│   ├── urban.py            Urban-core polygon (OSM) and coordinate conversion
│   ├── roads.py            OSM road intersections → point vortices in ℍ
│   ├── sc_solver_dc.py     Circle-theorem obstacle + combined potentials in ℍ
│   └── visualization.py    Figure generation
├── figures/                Generated output (see above)
├── main.py                 Full pipeline CLI
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--shapefile PATH` | - | TIGER/Line shapefile directory |
| `--demo` | off | Built-in hexagon domain, no data needed |
| `--terrain` | off | Terrain-corrected flow (requires USGS network access) |
| `--urban` | off | Urban-obstacle doubly-connected flow (requires OSMnx) |
| `--urban-method` | `osmnx` | `osmnx` (live OSM) or `fallback` (hardcoded polygon) |
| `--roads` | off | Road-vortex flow from OSM intersection data |
| `--road-method` | `osmnx` | `osmnx` (live OSM) or `fallback` (hardcoded intersections) |
| `--road-n-max` | 12 | Maximum road intersections to use as vortex sources |
| `--grid N` | 80 | Flow-grid resolution (N × N) |
| `--min-vertices` | 12 | Min vertices after Douglas-Peucker |
| `--max-vertices` | 16 | Max vertices after Douglas-Peucker |

---

## References

- Driscoll & Trefethen, *Schwarz-Christoffel Mapping*, Cambridge, 2009.
- Ablowitz & Fokas, *Complex Variables*, Cambridge, 2003.
- Milne-Thomson, *Theoretical Hydrodynamics*, 5th ed., Macmillan, 1968.
- US Census Bureau, TIGER/Line Shapefiles, 2025.
- USGS 3DEP Elevation Point Query Service: https://epqs.nationalmap.gov/v1/
- Boeing, G. (2017). OSMnx. *Computers, Environment and Urban Systems*, 65, 126-139.
