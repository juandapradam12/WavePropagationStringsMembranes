# Numerical methodology

This note records the continuous models, the discrete schemes, and the
diagnostics used in the simulations. It is meant to accompany the figures in
the README, not to replace them.

## Continuous problems

### Vibrating string (1D)

$$
\partial_{tt} u = c^2 \partial_{xx} u,
\qquad x \in (0,L),\ t > 0,
$$

with initial data $u(x,0)=u_0(x)$, $\partial_t u(x,0)=0$. Two boundary
regimes are considered:

1. **Fixed ends.** $u(0,t)=u(L,t)=0$.
2. **Driven end.** $u(0,t)=0$ and $u(L,t)=A\sin(\omega_1 t)$ with
   $\omega_1 = \pi c / L$ (the fundamental angular frequency).
3. **Free end.** $u(0,t)=0$ and $u_x(L,t)=0$ (Neumann), implemented with a
   ghost point $u_{N+1}=u_{N-1}$.

Optional linear damping $-\gamma\partial_t u$ uses the centered update

$$
u^{n+1}_i
=
\frac{2u^n_i-(1-\gamma\Delta t/2)u^{n-1}_i+r^2\Delta_h u^n_i}{1+\gamma\Delta t/2}.
$$

The exact fundamental period for fixed ends is $T_1 = 2L/c$.

### Clamped membrane (2D)

$$
\partial_{tt} u = c^2 \nabla^2 u,
\qquad (x,y)\in(0,L)^2,\ t>0,
$$

with either clamped edges $u|_{\partial\Omega}=0$ or free edges
$\partial_n u|_{\partial\Omega}=0$, and vanishing initial velocity.
Rectangular frequencies keep the form

$$
f_{mn} = \frac{c}{2}\sqrt{\Bigl(\frac{m}{L}\Bigr)^2+\Bigl(\frac{n}{L}\Bigr)^2},
$$

with sine factors and $m,n\ge 1$ when clamped, and cosine factors and
$m,n\ge 0$ when free (the rigid mode $(0,0)$ has $f_{00}=0$). Neumann
conditions are imposed by reflecting ghost points across each edge.

## Explicit finite-difference schemes

Uniform grids $\Delta x$ (and $\Delta y=\Delta x$ in 2D) and time step
$\Delta t$ define the Courant number $r = c\,\Delta t/\Delta x$.

### 1D leapfrog update

$$
u^{n+1}_i = 2(1-r^2)\,u^n_i - u^{n-1}_i + r^2\bigl(u^n_{i+1}+u^n_{i-1}\bigr).
$$

The first step from rest uses the centered starter

$$
u^{1}_i = u^{0}_i + \tfrac12 r^2\bigl(u^{0}_{i+1}-2u^{0}_i+u^{0}_{i-1}\bigr).
$$

Von Neumann analysis yields stability for $r\le 1$.

### 2D five-point update

$$
u^{n+1}_{i,j} = 2\bigl(1-2r^2\bigr)\,u^n_{i,j} - u^{n-1}_{i,j} + r^2\bigl(u^n_{i+1,j}+u^n_{i-1,j}+u^n_{i,j+1}+u^n_{i,j-1}\bigr)
$$

stable for $r\le 1/\sqrt{2}$.

## Discrete energy

A consistent discrete mechanical energy

$$
E^n = \frac{1}{2}\sum_i\bigl(\partial_t^{\mathrm{stg}} u\bigr)^2\Delta x + \frac{c^2}{2}\sum_i\bigl(\partial_x u\bigr)^2\Delta x
$$

(and its two-dimensional analogue) is monitored at every time step. For
closed systems (fixed boundaries, no drive) the relative drift
$\max|E^n-E^0|/|E^0|$ is the primary consistency diagnostic reported in the
figures.

## Exact references

With $\partial_t u(x,0)=0$ the fixed-end solution is given by d'Alembert's
formula on the odd $2L$-periodic extension $F$ of the initial data:

$$
u(x,t)=\frac12\bigl[F(x-ct)+F(x+ct)\bigr].
$$

A triangular pluck of height $A$ peaking at mid-span has sine coefficients
$b_n=8A/(n^2\pi^2)$ for odd $n$ (zero for even $n$), used for the modal
spectrum figure. Grid-refinement studies use a pure eigenmode
$A\sin(n\pi x/L)$ so the measured order is not polluted by series truncation.

The leapfrog dispersion relation

$$
\omega_{\mathrm{num}}(\kappa)=\frac{2}{\Delta t}\arcsin\bigl(r\sin(\kappa\Delta x/2)\bigr)
$$

is implemented under `wavefd/analysis/` and plotted by the pipeline.

## Code map

| Module | Role |
|--------|------|
| `wavefd/solvers/string_1d.py` | 1D leapfrog solver and string energy |
| `wavefd/solvers/membrane_2d.py` | 2D leapfrog solver and membrane energy |
| `wavefd/analysis/diagnostics.py` | CFL report, energy drift, period estimate |
| `wavefd/analysis/exact.py` | Modal series reference and $L^2$ errors |
| `wavefd/analysis/convergence.py` | Grid-refinement study |
| `wavefd/analysis/modes.py` | Discrete sine projection |
| `wavefd/analysis/dispersion.py` | Leapfrog dispersion relation |
| `wavefd/analysis/audio.py` | Mid-point WAV export |
| `wavefd/analysis/membrane_exact.py` | Exact $(m,n)$ membrane modes |
| `wavefd/analysis/membrane_modes.py` | Rectangular modal projection |
| `wavefd/analysis/cfl_demo.py` | Stable vs unstable Courant demo |
| `wavefd/viz/figures.py` | Figure suite used by the README |
| `wavefd/viz/extension_figures.py` | Membrane / CFL / damping / free-end panels |
| `scripts/run_pipeline.py` | End-to-end run: solve → diagnose → plot |
