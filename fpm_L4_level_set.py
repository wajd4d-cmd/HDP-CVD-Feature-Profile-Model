"""
fpm_L4_level_set.py
===================

[L4] Level-Set Solver for the HDP-CVD Feature Profile Model (FPM).

Consumes the 1-D net-normal-velocity field ``V_n(z)`` produced by the [L3]
``RateAssemblySolver`` and dynamically advances a 2-D Signed Distance Function
(SDF) phi(x, z, t) that represents the trench / void interface. The zero level
set {phi = 0} is the moving deposition front; tracking it through topological
change (overhang -> pinch-off -> sealed void) is exactly what a marker-string
profiler cannot do, which is the whole reason the level-set method is used for
HDP gap-fill.

Sign convention
---------------
    phi > 0  : OPEN region  (gas / chamber / void)      -- material is removed
    phi < 0  : SOLID region (substrate + deposited film) -- material is present
    phi = 0  : the interface (deposition / etch front)

With the outward (solid->gas) normal n = grad(phi)/|grad(phi)| pointing toward
increasing phi, the interface velocity is v = V_n * n and the level set obeys

        d(phi)/dt + V_n |grad(phi)| = 0                       (Hamilton-Jacobi)

so V_n > 0 (deposition, "fill") advances solid INTO the open region (the open
region shrinks) and V_n < 0 (sputter-dominated, "etch") recedes the solid.
This matches the L3 RateState.V_n sign (R_D - R_S + redep - O*div Js).

Numerics
--------
1. Spatial Hamiltonian: Godunov upwind flux (Osher-Sethian) so that the
   characteristics of |grad phi| are respected and no entropy-violating shocks /
   artificial facets are produced. Two reconstructions of the one-sided
   derivatives are provided:
       'upwind1' -- monotone first-order differences (rock-solid baseline);
       'weno5'   -- fifth-order HJ-WENO (Jiang-Peng) for low-dissipation,
                    facet-free fronts. Default.
2. Time integration: TVD / SSP Runge-Kutta 3 (Shu-Osher) under a CFL condition
   on the extended velocity.
3. Reinitialization: the SDF is periodically rebuilt to |grad phi| = 1 from an
   exact Euclidean distance transform of the sign field, which keeps the band
   well-conditioned without the drift of iterative PDE reinitialization.
4. Velocity extension: V_n is physically defined only ON the interface. A
   fast-sweeping extension solves  grad(F_ext) . grad(phi) = 0  so that F_ext is
   constant along interface normals, giving a continuous velocity throughout the
   2-D narrow band (the standard Adalsteinsson-Sethian construction).

Failure detection (per timestep)
--------------------------------
    pinch-off       : a connected component of OPEN cells that is NOT vented to
                      the chamber (top boundary) -- i.e. a sealed void below a
                      closed mouth, found by connected-component labeling.
    corner clipping : excessive recession (etch) of the field surface at the top
                      trench corners relative to the original substrate plane.

Author role: Principal Scientific Computing Engineer
Layer:       L4 (this file implements ONLY L4; it consumes L3's RateState).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage as _ndi
    _HAVE_SCIPY = True
except Exception:                                            # pragma: no cover
    _HAVE_SCIPY = False


# =============================================================================
# 1. CONFIGURATION CONTAINERS
# =============================================================================
@dataclass
class TrenchGeometry:
    """
    Initial trench (slot) geometry to be tracked by the level set.

    The slot is centered at x = 0, spans the full domain width in solid except
    for the open column |x| <= w0/2 over 0 <= z <= H. z = 0 is the wafer field
    plane (mouth); z increases DOWN into the substrate (consistent with L1-L3).
    Everything at z < 0 is the open chamber above the wafer.
    """
    w0_m: float = 100e-9             # mouth width                       [m]
    aspect_ratio: float = 10.0       # AR = H / w0                       [-]
    x_pad_w0: float = 1.0            # lateral pad each side, in units of w0
    z_pad_top_w0: float = 0.6        # chamber headroom above mouth, in w0
    z_pad_bot_w0: float = 0.8        # substrate below trench floor, in w0

    @property
    def depth_m(self) -> float:
        return self.aspect_ratio * self.w0_m


@dataclass
class LevelSetParams:
    """Discretization and solver controls."""
    nx: int = 121                    # transverse grid points (x)        [-]
    nz: int = 361                    # axial grid points (z)             [-]
    scheme: str = "weno5"            # 'weno5' | 'upwind1'
    cfl: float = 0.4                 # CFL number for SSP-RK3            [-]
    narrow_band_cells: float = 6.0   # half-width of velocity band, in cells
    reinit_every: int = 8            # SDF reinitialization cadence (steps)
    extend_sweeps: int = 4           # fast-sweep passes for velocity extension
    # --- failure thresholds ------------------------------------------------
    void_area_frac_thresh: float = 1.5e-3   # min sealed-open area / trench area
    corner_clip_frac: float = 0.12          # field recession / w0 that flags clip
    min_gap_pinch_frac: float = 0.06        # open mouth gap / w0 below -> sealing


# =============================================================================
# 2. OUTPUT CONTAINER
# =============================================================================
@dataclass
class ProfileState:
    """Immutable snapshot of the evolving profile handed back to the caller / L5."""
    t: float                         # physical time                     [s]
    step: int                        # timestep index                    [-]
    x: np.ndarray                    # transverse axis                   [m]
    z: np.ndarray                    # axial axis (mouth=0 -> down)       [m]
    phi: np.ndarray                  # SDF, shape (nz, nx)               [m]
    V_ext: np.ndarray                # extended normal velocity field     [m/s]
    # --- failure flags / metrics ------------------------------------------
    pinch_off: bool = False          # sealed void detected               [-]
    corner_clipping: bool = False    # top-corner over-etch detected      [-]
    void_area_m2: float = 0.0        # total sealed-open area             [m^2]
    void_fraction: float = 0.0       # sealed-open / trench cross-section  [-]
    seal_depth_m: float = float("nan")  # z of the seal (top of void)     [m]
    min_gap_m: float = float("nan")  # narrowest open span across trench   [m]
    corner_clip_depth_m: float = 0.0 # field recession at corners         [m]
    fill_fraction: float = 0.0       # filled trench / trench area         [-]
    flags: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, float]:
        """Scalar digest for logging / dashboards."""
        return {
            "t_s": self.t,
            "step": self.step,
            "pinch_off": float(self.pinch_off),
            "corner_clipping": float(self.corner_clipping),
            "fill_fraction": self.fill_fraction,
            "void_fraction": self.void_fraction,
            "void_area_nm2": self.void_area_m2 * 1e18,
            "seal_depth_nm": self.seal_depth_m * 1e9,
            "min_gap_nm": self.min_gap_m * 1e9,
            "corner_clip_nm": self.corner_clip_depth_m * 1e9,
        }


# =============================================================================
# 3. HJ-WENO5 / UPWIND ONE-SIDED DERIVATIVES
# =============================================================================
def _weno5_combine(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                   v4: np.ndarray, v5: np.ndarray) -> np.ndarray:
    r"""
    Jiang-Peng HJ-WENO5 convex combination of the three candidate stencils given
    the five consecutive divided differences ``v1..v5`` (already ordered for the
    desired one-sided derivative). Vectorized over all trailing axes.
    """
    S1 = 13.0 / 12.0 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - 4 * v2 + 3 * v3) ** 2
    S2 = 13.0 / 12.0 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
    S3 = 13.0 / 12.0 * (v3 - 2 * v4 + v5) ** 2 + 0.25 * (3 * v3 - 4 * v4 + v5) ** 2

    eps = 1e-6 * np.maximum.reduce([v1 * v1, v2 * v2, v3 * v3,
                                    v4 * v4, v5 * v5]) + 1e-99
    a1 = 0.1 / (S1 + eps) ** 2
    a2 = 0.6 / (S2 + eps) ** 2
    a3 = 0.3 / (S3 + eps) ** 2
    asum = a1 + a2 + a3

    p1 = (v1 / 3.0 - 7.0 * v2 / 6.0 + 11.0 * v3 / 6.0)
    p2 = (-v2 / 6.0 + 5.0 * v3 / 6.0 + v4 / 3.0)
    p3 = (v3 / 3.0 + 5.0 * v4 / 6.0 - v5 / 6.0)
    return (a1 * p1 + a2 * p2 + a3 * p3) / asum


def one_sided_derivs(phi: np.ndarray, h: float, axis: int,
                     scheme: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Dminus, Dplus): the backward- and forward-biased approximations of
    d(phi)/d(axis) at every node, using 'upwind1' or 'weno5'. Boundaries use
    edge-replicated ghost cells (homogeneous Neumann), appropriate for the
    open-chamber / deep-substrate frame.
    """
    phim = np.moveaxis(phi, axis, 0)
    n = phim.shape[0]

    # First-order one-sided differences as the baseline / WENO ingredient.
    f = np.concatenate([phim[:1], phim, phim[-1:]], axis=0)       # 1 ghost each
    Dminus = (f[1:-1] - f[:-2]) / h                               # backward
    Dplus = (f[2:] - f[1:-1]) / h                                 # forward

    if scheme == "upwind1":
        return (np.moveaxis(Dminus, 0, axis), np.moveaxis(Dplus, 0, axis))

    if scheme != "weno5":
        raise ValueError(f"Unknown scheme '{scheme}' (use 'weno5' or 'upwind1').")

    # Consecutive divided differences D_k = (phi_{k+1}-phi_k)/h, length n-1.
    Dc = (phim[1:] - phim[:-1]) / h
    # Edge-replicate 3 ghosts each side -> P length (n-1)+6 = n+5; P[j]=Dc[j-3].
    P = np.concatenate([np.repeat(Dc[:1], 3, axis=0), Dc,
                        np.repeat(Dc[-1:], 3, axis=0)], axis=0)

    # phi_x^- at node i (0..n-1) uses Dc[i-3..i+1] -> P[i..i+4].
    phix_minus = _weno5_combine(P[0:n], P[1:n + 1], P[2:n + 2],
                                P[3:n + 3], P[4:n + 4])
    # phi_x^+ at node i uses Dc[i+2..i-2] (reversed) -> P[i+5,i+4,i+3,i+2,i+1].
    phix_plus = _weno5_combine(P[5:n + 5], P[4:n + 4], P[3:n + 3],
                               P[2:n + 2], P[1:n + 1])

    return (np.moveaxis(phix_minus, 0, axis), np.moveaxis(phix_plus, 0, axis))


# =============================================================================
# 4. LEVEL-SET SOLVER
# =============================================================================
class LevelSetSolver:
    """
    Advance a 2-D SDF of a trench interface under an L3 net-normal velocity.

    Parameters
    ----------
    geometry : TrenchGeometry
    params   : LevelSetParams, optional
    rate_state : L3 ``RateState`` (optional convenience)
        If supplied, the (z, V_n) interface velocity is taken from it directly
        via :meth:`set_velocity`. Otherwise call :meth:`set_velocity` manually.

    Notes
    -----
    The velocity model V_n(z) is parameterized by absolute feature depth z (the
    L3 convention). The interface velocity at any front point is therefore
    V_n(clip(z, 0, H)); the field plane (z<0) inherits the mouth value and the
    sub-floor (z>H) the bottom value. This 1-D-in-z law is then extended off the
    interface into the band by the fast-sweeping construction so that the
    Hamiltonian sees a smooth 2-D speed.
    """

    def __init__(self, geometry: Optional[TrenchGeometry] = None,
                 params: Optional[LevelSetParams] = None,
                 rate_state: Optional[object] = None) -> None:
        self.geom = geometry if geometry is not None else TrenchGeometry()
        self.p = params if params is not None else LevelSetParams()

        self._build_grid()
        self.phi = self._initial_sdf()
        self.phi0 = self.phi.copy()                    # reference (t=0) profile

        # Velocity model V_n(z) (interface law); default zero until provided.
        self._z_src = np.array([0.0, self.geom.depth_m])
        self._Vn_src = np.zeros(2)
        self.V_ext = np.zeros_like(self.phi)

        self.t = 0.0
        self.step = 0

        # Cache the original field-plane (corner) reference for clip detection.
        self._field_surface_z0 = self._field_surface_depth(self.phi0)

        if rate_state is not None:
            self.set_velocity(rate_state.z, rate_state.V_n)

    # ------------------------------------------------------------------ grid
    def _build_grid(self) -> None:
        g, p = self.geom, self.p
        w0, H = g.w0_m, g.depth_m
        x_half = (0.5 + g.x_pad_w0) * w0
        z_lo = -g.z_pad_top_w0 * w0
        z_hi = H + g.z_pad_bot_w0 * w0

        self.x = np.linspace(-x_half, x_half, p.nx)
        self.z = np.linspace(z_lo, z_hi, p.nz)
        self.dx = float(self.x[1] - self.x[0])
        self.dz = float(self.z[1] - self.z[0])
        # phi indexed [iz, ix] -> Z varies down rows, X across columns.
        self.Z, self.X = np.meshgrid(self.z, self.x, indexing="ij")
        self.h_min = min(self.dx, self.dz)

    def _open_mask(self) -> np.ndarray:
        """Boolean OPEN (gas) mask for the initial geometry."""
        g = self.geom
        chamber = self.Z < 0.0
        trench = (np.abs(self.X) <= g.w0_m / 2.0) & (self.Z >= 0.0) \
            & (self.Z <= g.depth_m)
        return chamber | trench

    def _initial_sdf(self) -> np.ndarray:
        """
        Exact signed distance to the trench-wall boundary (phi>0 in open).
        Uses an Euclidean distance transform with physical (dz, dx) sampling so
        |grad phi| ~ 1 from the outset.
        """
        openm = self._open_mask()
        if _HAVE_SCIPY:
            d_open = _ndi.distance_transform_edt(openm, sampling=(self.dz, self.dx))
            d_solid = _ndi.distance_transform_edt(~openm, sampling=(self.dz, self.dx))
            phi = d_open - d_solid
        else:                                                # pragma: no cover
            phi = np.where(openm, 1.0, -1.0) * self.h_min
        return phi.astype(float)

    # -------------------------------------------------------------- velocity
    def set_velocity(self, z_src: np.ndarray, V_n: np.ndarray) -> None:
        """
        Register the L3 interface velocity law V_n(z). Stored as monotone-z
        samples for interpolation; re-callable each outer iteration to feed a
        time-varying L3 field (e.g. as geometry / shadowing evolves).
        """
        z_src = np.asarray(z_src, dtype=float)
        V_n = np.asarray(V_n, dtype=float)
        order = np.argsort(z_src)
        self._z_src = z_src[order]
        self._Vn_src = V_n[order]

    def _interface_velocity_law(self, zq: np.ndarray) -> np.ndarray:
        """V_n at depth zq via clamped linear interpolation of the L3 samples."""
        zc = np.clip(zq, self._z_src[0], self._z_src[-1])
        return np.interp(zc, self._z_src, self._Vn_src)

    def extend_velocity(self) -> np.ndarray:
        r"""
        Build the 2-D extension velocity F_ext satisfying

            grad(F_ext) . grad(phi) = 0

        so that F_ext is constant along the interface normals (Adalsteinsson-
        Sethian). Implementation: seed the narrow band cells adjacent to the
        interface with the physical law V_n(z), then propagate outward with a
        fast-sweeping (Gauss-Seidel) upwind update

            F_ij = (|phi_x| F_x^up + |phi_z| F_z^up) / (|phi_x| + |phi_z|)

        where the "up" neighbor in each axis is the one with SMALLER |phi|
        (i.e. toward the interface), which is the characteristic direction of the
        extension PDE. Four alternating sweep orderings give rapid convergence on
        the band; cells outside the band keep the analytic z-law (a consistent,
        divergence-free continuation for this depth-parameterized model).
        """
        phi = self.phi
        nz, nx = phi.shape
        absphi = np.abs(phi)
        band = self.p.narrow_band_cells * self.h_min

        # Analytic continuation everywhere (depth law); used as far-field value.
        F = self._interface_velocity_law(self.Z).astype(float)

        # Seed: cells straddling the zero set keep the analytic interface value
        # and are held FIXED during sweeping.
        sgn = np.sign(phi)
        seed = np.zeros_like(phi, dtype=bool)
        seed[:-1, :] |= sgn[:-1, :] * sgn[1:, :] < 0
        seed[1:, :] |= sgn[:-1, :] * sgn[1:, :] < 0
        seed[:, :-1] |= sgn[:, :-1] * sgn[:, 1:] < 0
        seed[:, 1:] |= sgn[:, :-1] * sgn[:, 1:] < 0

        active = (absphi <= band) & (~seed)

        # Precompute |phi_x|, |phi_z| (central, floored) for the weights.
        phix = np.zeros_like(phi)
        phiz = np.zeros_like(phi)
        phix[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * self.dx)
        phiz[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * self.dz)
        ax = np.abs(phix) + 1e-30
        az = np.abs(phiz) + 1e-30

        sweeps = [(range(nz), range(nx)), (range(nz - 1, -1, -1), range(nx)),
                  (range(nz), range(nx - 1, -1, -1)),
                  (range(nz - 1, -1, -1), range(nx - 1, -1, -1))]

        for _ in range(self.p.extend_sweeps):
            for zr, xr in sweeps:
                for i in zr:
                    for j in xr:
                        if not active[i, j]:
                            continue
                        # Upwind neighbor toward the interface in each axis.
                        if i > 0 and absphi[i - 1, j] < absphi[i, j]:
                            Fz, az_ij = F[i - 1, j], az[i, j]
                        elif i < nz - 1 and absphi[i + 1, j] < absphi[i, j]:
                            Fz, az_ij = F[i + 1, j], az[i, j]
                        else:
                            Fz, az_ij = 0.0, 0.0
                        if j > 0 and absphi[i, j - 1] < absphi[i, j]:
                            Fx, ax_ij = F[i, j - 1], ax[i, j]
                        elif j < nx - 1 and absphi[i, j + 1] < absphi[i, j]:
                            Fx, ax_ij = F[i, j + 1], ax[i, j]
                        else:
                            Fx, ax_ij = 0.0, 0.0
                        denom = ax_ij + az_ij
                        if denom > 0:
                            F[i, j] = (ax_ij * Fx + az_ij * Fz) / denom

        self.V_ext = F
        return F

    # ------------------------------------------------------- HJ Hamiltonian
    def _godunov_norm_grad(self, F: np.ndarray) -> np.ndarray:
        r"""
        Upwind (Godunov) approximation of |grad phi| consistent with the sign of
        the speed F, for the term F |grad phi| (Osher-Sethian):

            F > 0 :  |grad| = sqrt( max(Dm,0)^2 + min(Dp,0)^2 ) per axis
            F < 0 :  |grad| = sqrt( min(Dm,0)^2 + max(Dp,0)^2 ) per axis

        Returns the signed Hamiltonian  H = F * |grad phi|.
        """
        Dzm, Dzp = one_sided_derivs(self.phi, self.dz, axis=0, scheme=self.p.scheme)
        Dxm, Dxp = one_sided_derivs(self.phi, self.dx, axis=1, scheme=self.p.scheme)

        pos = F > 0
        # Growth side (F>0).
        gx_p = np.maximum(Dxm, 0) ** 2 + np.minimum(Dxp, 0) ** 2
        gz_p = np.maximum(Dzm, 0) ** 2 + np.minimum(Dzp, 0) ** 2
        grad_p = np.sqrt(gx_p + gz_p)
        # Etch side (F<0).
        gx_m = np.minimum(Dxm, 0) ** 2 + np.maximum(Dxp, 0) ** 2
        gz_m = np.minimum(Dzm, 0) ** 2 + np.maximum(Dzp, 0) ** 2
        grad_m = np.sqrt(gx_m + gz_m)

        grad = np.where(pos, grad_p, grad_m)
        return F * grad

    def _max_speed(self) -> float:
        """Bounding speed for CFL: max of the extended field AND the L3 law, so a
        finite dt is available even before the first velocity extension."""
        v_ext = float(np.max(np.abs(self.V_ext))) if self.V_ext.size else 0.0
        v_law = float(np.max(np.abs(self._Vn_src))) if self._Vn_src.size else 0.0
        return max(v_ext, v_law)

    def stable_dt(self) -> float:
        """CFL-limited timestep for the current extended velocity field."""
        vmax = self._max_speed()
        if vmax <= 0:
            return np.inf
        return self.p.cfl * self.h_min / vmax

    # ----------------------------------------------------------- RK3 stepping
    def _rhs(self, F: np.ndarray) -> np.ndarray:
        """d(phi)/dt = -F|grad phi| (Godunov)."""
        return -self._godunov_norm_grad(F)

    def step_once(self, dt: Optional[float] = None) -> float:
        """
        Advance one SSP-RK3 (Shu-Osher) step. Velocity is extended once per step
        (frozen across the RK stages, which is standard and stable for these
        CFL numbers). Returns the dt actually taken.
        """
        self.extend_velocity()
        F = self.V_ext
        if dt is None:
            dt = self.stable_dt()
        if not np.isfinite(dt):
            return 0.0

        phi_n = self.phi
        # Stage 1.
        self.phi = phi_n + dt * self._rhs(F)
        # Stage 2.
        self.phi = 0.75 * phi_n + 0.25 * (self.phi + dt * self._rhs(F))
        # Stage 3.
        self.phi = (phi_n + 2.0 * (self.phi + dt * self._rhs(F))) / 3.0

        self.t += dt
        self.step += 1
        if self.p.reinit_every and (self.step % self.p.reinit_every == 0):
            self.reinitialize()
        return dt

    def reinitialize(self) -> None:
        """
        Rebuild phi as a true signed distance from its current sign field using
        an exact Euclidean distance transform. Robust and drift-free; preserves
        the zero set to sub-cell accuracy for a well-resolved front.
        """
        if not _HAVE_SCIPY:                                  # pragma: no cover
            return
        openm = self.phi > 0
        if openm.all() or (~openm).all():
            return
        d_open = _ndi.distance_transform_edt(openm, sampling=(self.dz, self.dx))
        d_solid = _ndi.distance_transform_edt(~openm, sampling=(self.dz, self.dx))
        self.phi = d_open - d_solid

    # =====================================================================
    # 5. FAILURE DETECTION
    # =====================================================================
    def _field_surface_depth(self, phi: np.ndarray) -> float:
        """
        Depth z of the SOLID surface in the field region (the flat wafer top
        outside the trench mouth), averaged over the outer columns. Measured as
        the shallowest z at which phi changes from open(+) to solid(-) along z.
        Returns 0.0 for the pristine substrate (surface at the mouth plane).
        """
        g = self.geom
        outer = np.abs(self.x) >= (0.5 * g.w0_m + 0.5 * g.x_pad_w0 * g.w0_m)
        cols = np.where(outer)[0]
        if cols.size == 0:
            cols = np.array([0, phi.shape[1] - 1])
        depths = []
        for j in cols:
            col = phi[:, j]
            # First open->solid crossing scanning downward.
            sign_change = np.where((col[:-1] > 0) & (col[1:] <= 0))[0]
            if sign_change.size:
                k = sign_change[0]
                # Linear interpolation of the zero crossing.
                f0, f1 = col[k], col[k + 1]
                zc = self.z[k] + (self.z[k + 1] - self.z[k]) * f0 / (f0 - f1)
                depths.append(zc)
        return float(np.mean(depths)) if depths else 0.0

    def detect_failures(self) -> Dict[str, float]:
        """
        Run the per-timestep void / clip diagnostics on the current phi.

        Returns a flat metrics dict (also surfaced via :meth:`state`).
        """
        g, p = self.geom, self.p
        cell_area = self.dx * self.dz
        trench_area = g.w0_m * g.depth_m

        openm = self.phi > 0

        # ---- pinch-off: sealed open component not vented to the chamber ----
        pinch_off = False
        void_area = 0.0
        seal_depth = float("nan")
        if _HAVE_SCIPY and openm.any():
            lbl, n = _ndi.label(openm)
            vented = set(np.unique(lbl[0, :]))           # labels touching top row
            vented.discard(0)
            void_cells = np.zeros_like(openm)
            for comp in range(1, n + 1):
                if comp in vented:
                    continue
                comp_mask = lbl == comp
                area = comp_mask.sum() * cell_area
                if area >= p.void_area_frac_thresh * trench_area:
                    pinch_off = True
                    void_area += area
                    void_cells |= comp_mask
            if pinch_off:
                zr = np.where(void_cells.any(axis=1))[0]
                seal_depth = float(self.z[zr.min()])     # top of the void

        # ---- minimum open gap across the trench (proximity to sealing) -----
        in_trench = (self.z >= 0) & (self.z <= g.depth_m)
        min_gap = float("nan")
        if in_trench.any():
            gaps = []
            for i in np.where(in_trench)[0]:
                row_open = openm[i, :]
                if row_open.any():
                    gaps.append(row_open.sum() * self.dx)
                else:
                    gaps.append(0.0)
            min_gap = float(np.min(gaps)) if gaps else float("nan")

        # ---- corner clipping: field-surface recession vs. pristine plane ----
        field_z = self._field_surface_depth(self.phi)
        clip_depth = max(field_z - self._field_surface_z0, 0.0)
        corner_clipping = clip_depth >= p.corner_clip_frac * g.w0_m

        # ---- fill fraction (solid inside the original trench footprint) ----
        trench_fp = (np.abs(self.X) <= g.w0_m / 2.0) & (self.Z >= 0) \
            & (self.Z <= g.depth_m)
        solid_in_trench = (self.phi <= 0) & trench_fp
        fill_fraction = float(solid_in_trench.sum() * cell_area / trench_area)

        return {
            "pinch_off": pinch_off,
            "corner_clipping": corner_clipping,
            "void_area_m2": void_area,
            "void_fraction": void_area / trench_area,
            "seal_depth_m": seal_depth,
            "min_gap_m": min_gap,
            "corner_clip_depth_m": clip_depth,
            "fill_fraction": fill_fraction,
        }

    # =====================================================================
    # 6. STATE / DRIVER
    # =====================================================================
    def state(self) -> ProfileState:
        """Bundle the current grid + failure diagnostics into a ProfileState."""
        m = self.detect_failures()
        return ProfileState(
            t=self.t, step=self.step, x=self.x, z=self.z,
            phi=self.phi.copy(), V_ext=self.V_ext.copy(),
            pinch_off=bool(m["pinch_off"]),
            corner_clipping=bool(m["corner_clipping"]),
            void_area_m2=m["void_area_m2"], void_fraction=m["void_fraction"],
            seal_depth_m=m["seal_depth_m"], min_gap_m=m["min_gap_m"],
            corner_clip_depth_m=m["corner_clip_depth_m"],
            fill_fraction=m["fill_fraction"], flags=m,
        )

    def advance(self, t_end: float,
                velocity_callback: Optional[Callable[[float], Tuple[np.ndarray, np.ndarray]]] = None,
                max_steps: int = 100000,
                stop_on_pinch: bool = False) -> ProfileState:
        """
        March to ``t_end``. If ``velocity_callback(t)`` is given it must return
        (z_src, V_n) for the CURRENT time, letting the L3 field evolve as the
        geometry changes. Stops early on pinch-off if requested.
        """
        while self.t < t_end and self.step < max_steps:
            if velocity_callback is not None:
                z_src, V_n = velocity_callback(self.t)
                self.set_velocity(z_src, V_n)
            dt = min(self.stable_dt(), t_end - self.t)
            self.step_once(dt)
            if stop_on_pinch and self.detect_failures()["pinch_off"]:
                break
        return self.state()

    # ---------------------------------------------------- ASCII visualization
    def ascii_profile(self, width: int = 72, rows: int = 30) -> str:
        """Coarse ASCII cross-section: '#' solid, ' ' open, '.' near-interface."""
        zi = np.linspace(0, self.phi.shape[0] - 1, rows).astype(int)
        xi = np.linspace(0, self.phi.shape[1] - 1, width).astype(int)
        sub = self.phi[np.ix_(zi, xi)]
        band = 1.5 * self.h_min
        out = []
        for r in range(rows):
            line = []
            for c in range(width):
                v = sub[r, c]
                line.append("#" if v < -band else (" " if v > band else "."))
            out.append("".join(line))
        return "\n".join(out)


# =============================================================================
# 7. DEMONSTRATION / SELF-TEST
# =============================================================================
def _try_real_L3_velocity(geom: TrenchGeometry):
    """Prefer the validated L0->L3 chain; return (z, V_n) or None."""
    try:
        from fpm_L0_global_plasma import GlobalPlasmaSolver, ReactorInputs
        from fpm_L1_feature_transport import (FeatureTransportSolver,
                                              FeatureGeometry, TransportParams)
        from fpm_L2_surface_kinetics import SurfaceKineticsSolver, SurfaceParams
        from fpm_L3_rate_assembly import RateAssemblySolver
        inp = ReactorInputs(power_w=2500.0, pressure_mtorr=5.0,
                            flows_sccm={"Ar": 100.0, "O2": 50.0, "SiH4": 30.0},
                            rf_bias_v=80.0)
        plasma = GlobalPlasmaSolver(inp).solve()
        fg = FeatureGeometry(aspect_ratio=geom.aspect_ratio,
                             w0_m=geom.w0_m, n_z=200)
        tr = FeatureTransportSolver(plasma, fg, TransportParams(0.10)).solve()
        sf = SurfaceKineticsSolver(plasma, tr, fg, SurfaceParams()).solve()
        rs = RateAssemblySolver(plasma, tr, sf).solve()
        return rs.z, rs.V_n
    except Exception as exc:                                 # pragma: no cover
        print(f"[L4 demo] real L3 chain unavailable ({exc}); using mock V_n(z).")
        return None


def _mock_breadloaf_velocity(geom: TrenchGeometry, t: float, t_seal: float
                             ) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Mock time-stepped L3 field engineered to drive an overhang -> pinch-off
    sequence (the classic "breadloaf" gap-fill failure).

    The depositing flux is geometrically shadowed with depth, so the net normal
    velocity is large at the mouth and falls off into the trench:

        V_n(z) = V_top * [ a_floor + (1 - a_floor) exp(-z / lambda) ]

    A mild time ramp on V_top accelerates the mouth as the overhang grows
    (positive feedback: narrowing aperture -> more shadowing -> faster closure),
    so the top seals while the lower trench is still open -> a void.
    """
    H = geom.depth_m
    z = np.linspace(0.0, H, 200)
    lam = 0.28 * H                         # shadowing decay length
    a_floor = 0.06                         # residual bottom fill fraction
    V_top0 = 42e-9 / 60.0                  # 42 nm/min at the mouth   [m/s]
    ramp = 1.0 + 0.9 * min(t / t_seal, 1.0)
    V_top = V_top0 * ramp
    Vn = V_top * (a_floor + (1.0 - a_floor) * np.exp(-z / lam))
    return z, Vn


def _demo() -> None:
    """
    Drive the level set with a mock, time-stepped V_n(z) and mathematically
    demonstrate a sealed void forming (pinch-off) in a 10:1, 100 nm trench.
    """
    geom = TrenchGeometry(w0_m=100e-9, aspect_ratio=10.0)
    params = LevelSetParams(nx=121, nz=361, scheme="weno5",
                            cfl=0.4, reinit_every=8)
    solver = LevelSetSolver(geom, params)

    print("=" * 74)
    print(" [L4] LEVEL-SET ADVANCE  -  10:1 trench (w0 = 100 nm), HJ-WENO5 + RK3")
    print("=" * 74)
    print(f" Grid            : {params.nx} x {params.nz}  "
          f"(dx={solver.dx*1e9:.2f} nm, dz={solver.dz*1e9:.2f} nm)")
    real = _try_real_L3_velocity(geom)
    if real is not None:
        z3, Vn3 = real
        print(f" L3 coupling     : real RateAssemblySolver  "
              f"(V_n mouth={Vn3[0]*1e9*60:.2f}, bottom={Vn3[-1]*1e9*60:.2f} nm/min)")
        print(" NOTE: a sputter-balanced L3 field fills void-free; the demo below")
        print("       uses a deliberately shadow-limited mock field to force a void.")
    print("-" * 74)

    # Estimated seal time scale: lateral half-gap / mouth speed.
    t_seal = (geom.w0_m / 2.0) / (42e-9 / 60.0)
    t_end = 1.5 * t_seal

    def vcb(t: float):
        return _mock_breadloaf_velocity(geom, t, t_seal)

    report_times = np.linspace(0, t_end, 7)[1:]
    ri = 0
    first_pinch = None
    print(f" {'t[s]':>8}{'step':>6}{'fill%':>8}{'min_gap[nm]':>13}"
          f"{'void%':>8}{'seal_z[nm]':>12}{'clip[nm]':>10}  flags")
    while solver.t < t_end:
        z_src, V_n = vcb(solver.t)
        solver.set_velocity(z_src, V_n)
        dt = min(solver.stable_dt(), t_end - solver.t)
        solver.step_once(dt)

        if ri < len(report_times) and solver.t >= report_times[ri]:
            st = solver.state()
            fl = []
            if st.pinch_off:
                fl.append("PINCH-OFF")
            if st.corner_clipping:
                fl.append("CORNER-CLIP")
            print(f" {st.t:>8.3f}{st.step:>6d}{st.fill_fraction*100:>8.1f}"
                  f"{st.min_gap_m*1e9:>13.2f}{st.void_fraction*100:>8.2f}"
                  f"{st.seal_depth_m*1e9:>12.1f}{st.corner_clip_depth_m*1e9:>10.2f}"
                  f"  {','.join(fl) if fl else '-'}")
            if st.pinch_off and first_pinch is None:
                first_pinch = st
            ri += 1

    final = solver.state()
    print("-" * 74)
    print(" Final profile diagnostics:")
    for k, v in final.summary().items():
        print(f"   {k:<18}: {v:.5g}")
    print("-" * 74)
    if final.pinch_off:
        print(f" RESULT: VOID CONFIRMED -- sealed open region of "
              f"{final.void_area_m2*1e18:.0f} nm^2 "
              f"({final.void_fraction*100:.1f}% of trench) "
              f"capped at z = {final.seal_depth_m*1e9:.0f} nm.")
    else:
        print(" RESULT: no sealed void detected (trench filled / still venting).")
    print("-" * 74)
    print(" Final interface (cross-section; '#'=solid film/substrate, ' '=void/gas):")
    print(solver.ascii_profile(width=72, rows=30))
    print("=" * 74)


if __name__ == "__main__":
    _demo()
