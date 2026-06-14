"""
fpm_L1_feature_transport.py
===========================

[L1] Feature-Local Transport Solver for the HDP-CVD Feature Profile Model (FPM).

Consumes the wafer-uniform ``PlasmaState`` produced by [L0] together with a
``FeatureGeometry`` description of a single high-aspect-ratio (HAR) trench, and
computes the 1-D, vertically-resolved transport fields along the trench depth z
(z = 0 at the trench mouth, z = H = AR * w_0 at the bottom):

    Kn(z)     : local Knudsen number  lambda / w(z)                    [-]
    n_g(z)    : neutral (precursor) density profile down the trench    [m^-3]
    V_trench  : Clausing bottom-to-top arrival ratio  n_g(H)/n_g(0)    [-]
    phi_s(z)  : dielectric surface potential (differential charging)    [V]
    sigma_q(z): surface charge field  ~ eps0 * E_axial                  [C/m^2]
    dtheta(z) : charge-induced ion deflection angle                     [rad]

Physics implemented
-------------------
1. Neutral transport -- free-molecular (Knudsen) diffusion with wall sticking
   loss, solved as a two-point boundary-value problem (scipy.solve_bvp):

       d/dz [ D_K(z) dn_g/dz ] = k_wall(z) * n_g(z)
       D_K(z) = (2/3) w(z) v_th      (Knudsen slot diffusivity)
       k_wall(z) = s_c v_th / (2 w(z))   (two-sidewall sticking loss)

   BCs: n_g(0) = n_g,mouth ;  -D_K n_g'(H) = s_c (1/4) v_th n_g(H)  (bottom sink).

2. Differential charging -- a reduced 1-D floating-dielectric model. Directional
   ions (narrow IADF from L0) reach depth z almost unattenuated; isotropic
   electrons are geometrically shadowed with transmission G(z) = w^2/(w^2+z^2).
   Local current balance on the insulator (J_i = J_e) gives:

       phi_s(z) = T_e * ln[ Gamma_i / ( (1/4) n_e v_e,bar * G(z) ) ]

   -> top stays near the (negative) floating potential; the electron-starved
   bottom charges positive: the classic top-negative / bottom-positive dipole.

3. Ion deflection -- the transverse field E_perp ~ 2 phi_s / w imparts a lateral
   impulse to the descending ion; the cumulative deflection is

       dtheta(z) = arctan[ (1 / (2 V_sh)) * \int_0^z (2 phi_s / w) dz' ].

Fidelity note: the rigorous charging problem is a 2-D Poisson solve coupled to
angular ion/electron transport (deferred to a future L1+). The closures here are
the standard 1-D reductions and are clearly flagged as such.

Author role: Principal Scientific Computing Engineer
Layer:       L1 (this file implements ONLY L1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from scipy import constants as const
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.sparse import diags, identity, kron, csr_matrix
from scipy.sparse.linalg import spsolve

# =============================================================================
# 1. PHYSICAL CONSTANTS  (true SI values from CODATA via scipy.constants)
# =============================================================================
E_CHARGE: float = const.e               # elementary charge            [C]
M_E: float = const.m_e                  # electron mass                [kg]
K_B: float = const.k                    # Boltzmann constant           [J/K]
EPS0: float = const.epsilon_0           # vacuum permittivity          [F/m]
AMU: float = const.physical_constants["atomic mass constant"][0]  # [kg]
EV_TO_J: float = const.e                # 1 eV in Joules               [J/eV]


# =============================================================================
# 2. INPUT CONTAINERS
# =============================================================================
@dataclass
class FeatureGeometry:
    """
    Geometry of a single trench / via to be resolved by L1.

    The width profile w(z) defaults to a constant-width slot but accepts an
    arbitrary callable (e.g. tapered or bowed sidewalls) plus its derivative
    w'(z) so the Knudsen diffusivity gradient is exact inside the BVP.
    """
    aspect_ratio: float                  # AR = H / w_0                    [-]
    w0_m: float                          # top (mouth) width              [m]
    n_z: int = 200                       # number of axial mesh points    [-]
    width_profile: Optional[Callable[[np.ndarray], np.ndarray]] = None
    width_derivative: Optional[Callable[[np.ndarray], np.ndarray]] = None

    @property
    def depth_m(self) -> float:
        """Trench depth H = AR * w_0."""
        return self.aspect_ratio * self.w0_m

    @property
    def z_axis(self) -> np.ndarray:
        """1-D axial mesh from mouth (0) to bottom (H), length n_z."""
        return np.linspace(0.0, self.depth_m, self.n_z)

    def width(self, z: np.ndarray) -> np.ndarray:
        """Local trench width w(z); constant slot by default."""
        z = np.asarray(z, dtype=float)
        if self.width_profile is None:
            return np.full_like(z, self.w0_m)
        return self.width_profile(z)

    def width_grad(self, z: np.ndarray) -> np.ndarray:
        """dw/dz; zero for the default constant slot."""
        z = np.asarray(z, dtype=float)
        if self.width_derivative is None:
            return np.zeros_like(z)
        return self.width_derivative(z)


@dataclass
class TransportParams:
    """
    Gas / surface parameters not carried by the L0 PlasmaState. Defaults are
    representative of an HDP-CVD oxide process; calibrate against fab data.
    """
    sticking_coeff: float = 0.10         # precursor wall sticking s_c    [-]
                                         #   SiO2 HDP precursors ~0.01-0.3
    gas_temp_k: float = 500.0            # neutral gas temperature        [K]
    neutral_mass_amu: float = 32.0       # effective precursor mass       [amu]
                                         #   (SiH4 ~32, O2 ~32)
    neutral_diameter_m: float = 3.7e-10  # kinetic collision diameter     [m]
                                         #   (~3.4-4.0 A for these gases)
    eps_ox_rel: float = 3.9              # dielectric constant (SiO2 ~3.9) [-]


# =============================================================================
# 3. OUTPUT CONTAINER
# =============================================================================
@dataclass
class TransportState:
    """z-resolved L1 transport fields handed to L2/L3."""
    z: np.ndarray                        # axial mesh (mouth->bottom)     [m]
    width: np.ndarray                    # w(z)                           [m]
    Kn: np.ndarray                       # Knudsen number Kn(z)           [-]
    D_K: np.ndarray                      # Knudsen diffusivity D_K(z)     [m^2/s]
    n_g: np.ndarray                      # total neutral density n_g(z)   [m^-3]
    n_dep: np.ndarray                    # depositing-precursor density   [m^-3]
                                         #   = f_precursor * n_g(z); only this
                                         #   Si-bearing fraction feeds R_D.
    phi_s: np.ndarray                    # surface potential phi_s(z)     [V]
    sigma_q: np.ndarray                  # surface charge field           [C/m^2]
    dtheta_charge: np.ndarray            # ion deflection angle           [rad]
    V_trench: float                      # bottom/top neutral arrival     [-]
    V_trench_clausing: float             # closed-form Clausing cross-check [-]
    mean_free_path_m: float              # gas-phase lambda               [m]

    def summary(self) -> Dict[str, float]:
        """Scalar digest for logging / dashboards."""
        return {
            "V_trench_bvp": self.V_trench,
            "V_trench_clausing": self.V_trench_clausing,
            "Kn_top": float(self.Kn[0]),
            "Kn_bottom": float(self.Kn[-1]),
            "n_g_top_m3": float(self.n_g[0]),
            "n_g_bottom_m3": float(self.n_g[-1]),
            "n_dep_top_m3": float(self.n_dep[0]),
            "phi_s_top_V": float(self.phi_s[0]),
            "phi_s_bottom_V": float(self.phi_s[-1]),
            "dtheta_bottom_deg": float(np.degrees(self.dtheta_charge[-1])),
            "mean_free_path_m": self.mean_free_path_m,
        }


# =============================================================================
# 4. FEATURE-LOCAL TRANSPORT SOLVER
# =============================================================================
class FeatureTransportSolver:
    """
    Feature-local transport for one trench, with a *true 2-D Poisson* treatment
    of differential charging.

    The neutral-transport block (Knudsen BVP, Clausing cross-check) is unchanged
    from the baseline L1. The charging block is upgraded: the 1-D closure
    E_perp ~ 2 phi_s / w -- which assumes the full surface potential drops across
    the half-width and badly over-predicts deflection in HAR trenches -- is
    replaced by:

      1. floating dielectric-wall potential phi_wall(z) from local ion/electron
         current balance (this is the Dirichlet data),
      2. a vectorized sparse 5-point FDM solve of  d2phi/dx2 + d2phi/dz2 = -rho/eps0
         on the 2-D trench cross-section,
      3. extraction of the *true* transverse field E_x = -dphi/dx and integration
         of the ion-trajectory deflection from it.

    In a thin HAR trench the transverse potential equilibrates within ~w of the
    mouth (both walls share phi_wall(z)), so E_x is strongly screened at depth
    and the deflection is physically attenuated -- the purpose of the refactor.

    Parameters
    ----------
    plasma : PlasmaState (L0)   -- T_e, n_e, n_g, V_sheath, m_ion_eff_amu, ieadf
    geometry : FeatureGeometry
    params : TransportParams, optional
    poisson_nx : int            -- transverse grid resolution (odd -> has axis)

    Notes
    -----
    The structured 2-D grid assumes near-vertical sidewalls (w(z) ~ w_0), the
    regime of interest for HAR gap-fill. Strongly tapered/bowed profiles need a
    body-fitted mesh (future L1+).
    """

    def __init__(self, plasma, geometry, params=None, poisson_nx: int = 41):
        self.plasma = plasma
        self.geom = geometry
        self.p = params if params is not None else TransportParams()

        # Neutral thermal speed v_th = sqrt(8 kT / (pi m)) [m/s].
        m_n = self.p.neutral_mass_amu * AMU
        self.v_th = np.sqrt(8.0 * K_B * self.p.gas_temp_k / (np.pi * m_n))

        # Gas-phase mean free path lambda = 1/(sqrt(2) n_g sigma_gg) [m].
        sigma_gg = np.pi * self.p.neutral_diameter_m ** 2
        self.mfp = 1.0 / (np.sqrt(2.0) * self.plasma.n_g * sigma_gg)

        # Bohm velocity from L0 ion mass [m/s].
        m_ion = self.plasma.m_ion_eff_amu * AMU
        self.u_B = np.sqrt(E_CHARGE * self.plasma.T_e / m_ion)

        # Mean electron thermal speed v_e,bar = sqrt(8 e T_e/(pi m_e)) [m/s].
        self.v_e_bar = np.sqrt(8.0 * E_CHARGE * self.plasma.T_e / (np.pi * M_E))

        # Mean ion impact angle from the L0 IEADF [rad].
        self.mean_ion_angle_rad = np.radians(self._read_mean_ion_angle())

        # 2-D Poisson grid resolution + diagnostic caches (filled by solve()).
        self.poisson_nx = int(poisson_nx)
        self.X = self.Z = self.phi_2d = self.Ex_2d = self.Ez_2d = None
        self.dtheta_2d = None
        self._dx_poisson = self._dz_poisson = None

    # ------------------------------------------------------------- helpers
    def _read_mean_ion_angle(self) -> float:
        """Mean ion polar angle [deg]; fall back to ~1 deg if unavailable."""
        ieadf = getattr(self.plasma, "ieadf", None)
        if ieadf is not None and hasattr(ieadf, "mean_angle_deg"):
            return float(ieadf.mean_angle_deg())
        return 1.0

    def knudsen_diffusivity(self, z: np.ndarray) -> np.ndarray:
        """Knudsen slot diffusivity D_K(z) = (2/3) w(z) v_th [m^2/s]."""
        return (2.0 / 3.0) * self.geom.width(z) * self.v_th

    def knudsen_number(self, z: np.ndarray) -> np.ndarray:
        """Local Knudsen number Kn(z) = lambda / w(z)."""
        return self.mfp / self.geom.width(z)

    # -------------------------------------------------- neutral transport
    def _solve_neutral_bvp(self) -> np.ndarray:
        r"""
        Solve  d/dz[D_K(z) n'] = k_wall(z) n  for n_g(z) via scipy.solve_bvp,
        non-dimensionalized (zeta = z/H, ntil = n/n_mouth) for conditioning.
        BCs: n(0) = n_mouth ; -D_K(H) n'(H) = s_c (1/4) v_th n(H) (bottom sink).
        """
        z_mesh = self.geom.z_axis
        H = self.geom.depth_m
        n_mouth = self.plasma.n_g
        s_c, v_th = self.p.sticking_coeff, self.v_th
        zeta_mesh = z_mesh / H

        def odes(zeta: np.ndarray, y: np.ndarray) -> np.ndarray:
            z = zeta * H
            w = self.geom.width(z)
            dw = self.geom.width_grad(z)
            D_K = (2.0 / 3.0) * w * v_th
            dD_K = (2.0 / 3.0) * v_th * dw
            k_wall = s_c * v_th / (2.0 * w)
            ntil, dntil = y
            d2 = (k_wall * H ** 2 / D_K) * ntil - (dD_K * H / D_K) * dntil
            return np.vstack((dntil, d2))

        def bcs(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
            w_H = float(self.geom.width(np.array([H]))[0])
            D_K_H = (2.0 / 3.0) * w_H * v_th
            bc_top = ya[0] - 1.0
            bc_bot = -(D_K_H / H) * yb[1] - s_c * 0.25 * v_th * yb[0]
            return np.array([bc_top, bc_bot])

        L_decay = self.geom.w0_m * 2.0 / np.sqrt(max(3.0 * s_c, 1e-6))
        ntil_guess = np.exp(-z_mesh / L_decay)
        dntil_guess = -(H / L_decay) * ntil_guess
        y_guess = np.vstack((ntil_guess, dntil_guess))

        sol = solve_bvp(odes, bcs, zeta_mesh, y_guess, tol=1e-8, max_nodes=20000)
        if not sol.success:
            raise RuntimeError(f"Neutral BVP failed to converge: {sol.message}")
        return sol.sol(zeta_mesh)[0] * n_mouth

    def clausing_transmission(self) -> float:
        """Closed-form Clausing transmission cross-check:
        V = 1 / (1 + 0.5 s_c AR (2 - s_c))."""
        s_c, AR = self.p.sticking_coeff, self.geom.aspect_ratio
        return 1.0 / (1.0 + 0.5 * s_c * AR * (2.0 - s_c))

    # ------------------------------------------- differential charging (2-D)
    def _wall_potential(self, z: np.ndarray) -> np.ndarray:
        r"""
        Floating dielectric-wall potential phi_wall(z) [V] from local current
        balance J_i = J_e (Dirichlet data for the 2-D Poisson solve):

            phi_wall(z) = T_e * ln[ Gamma_i / ((1/4) n_e v_e,bar G(z)) ]

        Directional ions reach all depths (Gamma_i = n_e u_B); isotropic
        electrons are geometrically shadowed by G(z) = w^2/(w^2 + z^2).
        """
        Te = self.plasma.T_e
        Gamma_i = self.plasma.n_e * self.u_B
        Gamma_e0 = 0.25 * self.plasma.n_e * self.v_e_bar
        w = self.geom.width(z)
        G = w ** 2 / (w ** 2 + z ** 2)
        ratio = Gamma_i / (Gamma_e0 * np.clip(G, 1e-30, None))
        return Te * np.log(ratio)

    @staticmethod
    def _laplacian_1d(n: int, h: float):
        """1-D second-difference operator (Dirichlet-ready), spacing h."""
        main = -2.0 / h ** 2 * np.ones(n)
        off = 1.0 / h ** 2 * np.ones(n - 1)
        return diags([off, main, off], offsets=[-1, 0, 1], format="csr")

    def _solve_poisson_2d(self, z: np.ndarray, phi_wall: np.ndarray,
                          rho: "np.ndarray | None" = None) -> dict:
        r"""
        Vectorized sparse 5-point FDM solve of  laplacian(phi) = -rho/eps0  on
        the trench cross-section (x across width in [0, w_0], z down depth).

        Node ordering k = i*nz + j (C-order). The 2-D operator is the tensor sum
            L = Dxx (x) I_z  +  I_x (x) Dzz.
        Dirichlet rows are imposed by the fully vectorized projection
            A = P_int @ L + P_bnd          (no per-node Python loops)
        where P_int / P_bnd are diagonal interior / boundary selectors.

        BCs:  top (z=0)   : phi = 0            (plasma/presheath reference)
              walls (x=0,w0): phi = phi_wall(z) (charged dielectric)
              bottom (z=H) : phi = phi_wall(H)  (charged dielectric)
        """
        nx, nz = self.poisson_nx, z.size
        dx = self.geom.w0_m / (nx - 1)
        dz = float(z[1] - z[0])

        # Tensor-product Laplacian (sparse).
        Dxx = self._laplacian_1d(nx, dx)
        Dzz = self._laplacian_1d(nz, dz)
        Ix = identity(nx, format="csr")
        Iz = identity(nz, format="csr")
        L = kron(Dxx, Iz, format="csr") + kron(Ix, Dzz, format="csr")

        # Boundary mask + Dirichlet values on the (nx, nz) grid.
        mask = np.zeros((nx, nz), dtype=bool)
        val = np.zeros((nx, nz), dtype=float)
        mask[0, :] = mask[-1, :] = True          # side walls
        val[0, :] = val[-1, :] = phi_wall
        mask[:, -1] = True                        # bottom (charged)
        val[:, -1] = phi_wall[-1]
        mask[:, 0] = True                         # top (reference) overrides corners
        val[:, 0] = 0.0
        mask_f = mask.ravel(order="C")
        val_f = val.ravel(order="C")

        # RHS: source defaults to 0 (Debye-screened interior); Dirichlet pinned.
        b = np.zeros(nx * nz, dtype=float)
        if rho is not None:
            b[:] = -rho.ravel(order="C") / EPS0
        b[mask_f] = val_f[mask_f]

        # Vectorized Dirichlet projection.
        P_int = diags((~mask_f).astype(float), format="csr")
        P_bnd = diags(mask_f.astype(float), format="csr")
        A = (P_int @ L + P_bnd).tocsr()

        phi = spsolve(A, b).reshape(nx, nz)

        # True electrostatic field [V/m]; Ex is the transverse (deflecting) part.
        Ex = -np.gradient(phi, dx, axis=0)
        Ez = -np.gradient(phi, dz, axis=1)
        x = np.linspace(0.0, self.geom.w0_m, nx)
        X, Z = np.meshgrid(x, z, indexing="ij")
        return {"X": X, "Z": Z, "phi": phi, "Ex": Ex, "Ez": Ez, "dx": dx, "dz": dz}

    def _solve_charging(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the wall potential (Dirichlet data) and run the 2-D Poisson
        solve. Caches the 2-D fields on self and returns phi_s(z) = phi_wall(z).
        """
        phi_wall = self._wall_potential(z)
        sol = self._solve_poisson_2d(z, phi_wall)
        self.X, self.Z = sol["X"], sol["Z"]
        self.phi_2d, self.Ex_2d, self.Ez_2d = sol["phi"], sol["Ex"], sol["Ez"]
        self._dx_poisson, self._dz_poisson = sol["dx"], sol["dz"]
        return phi_wall

    def _charge_field(self, z: np.ndarray) -> np.ndarray:
        """
        True dielectric surface-charge field from Gauss's law at the wall:
            sigma_q(z) = eps0 eps_r E_n,  E_n = -dphi/dx at the sidewall (x=0).
        """
        E_n_wall = self.Ex_2d[0, :]              # transverse field at left wall
        return EPS0 * self.p.eps_ox_rel * E_n_wall

    def _deflection_angle(self, z: np.ndarray) -> np.ndarray:
        r"""
        Physically-attenuated ion deflection from the *true* transverse field.
        For each entrance column x0 the descending ion (vertical speed u_z,
        with M u_z^2 = 2 e V_sh) accrues

            theta(x0, z) = arctan[ (1/(2 V_sh)) \int_0^z E_x(x0, z') dz' ].

        The z-resolved deflection is the RMS over interior entrance positions
        (the centerline is field-free by symmetry).
        """
        Ex = self.Ex_2d
        dz = self._dz_poisson
        impulse = cumulative_trapezoid(Ex, dx=dz, axis=1, initial=0.0)  # int E_x dz
        theta2d = np.arctan(impulse / (2.0 * self.plasma.V_sheath))
        self.dtheta_2d = theta2d
        return np.sqrt(np.mean(theta2d[1:-1, :] ** 2, axis=0))

    # ----------------------------------------------------------------- solve
    def solve(self) -> "TransportState":
        """Run the full L1 solve and return a TransportState."""
        z = self.geom.z_axis
        w = self.geom.width(z)

        # Neutral transport.
        Kn = self.knudsen_number(z)
        D_K = self.knudsen_diffusivity(z)
        n_g = self._solve_neutral_bvp()
        V_trench = float(n_g[-1] / n_g[0])
        V_clausing = self.clausing_transmission()

        # Depositing-precursor density: only the Si-bearing feed fraction builds
        # film. Total n_g still sets the gas-phase mean free path / Knudsen number;
        # n_dep is what L2/L3 must use for the deposition flux R_D.
        f_precursor = float(getattr(self.plasma, "f_precursor", 1.0))
        n_dep = n_g * f_precursor

        # Differential charging via 2-D Poisson.
        phi_s = self._solve_charging(z)          # caches 2-D fields on self
        sigma_q = self._charge_field(z)
        dtheta = self._deflection_angle(z)

        return TransportState(
            z=z, width=w, Kn=Kn, D_K=D_K, n_g=n_g, n_dep=n_dep, phi_s=phi_s,
            sigma_q=sigma_q, dtheta_charge=dtheta, V_trench=V_trench,
            V_trench_clausing=V_clausing, mean_free_path_m=self.mfp,
        )



# =============================================================================
# 5. DEMONSTRATION / SELF-TEST  (mocks an L0 PlasmaState)
# =============================================================================
def _make_mock_plasma_state():
    """
    Build a lightweight stand-in for the L0 PlasmaState using values that match
    the validated L0 demo (HDP Ar/O2/SiH4, 2.5 kW, 5 mTorr, 80 V bias). Prefers
    the real L0 classes if importable, otherwise a duck-typed namespace.
    """
    try:
        from fpm_L0_global_plasma import GlobalPlasmaSolver, ReactorInputs
        inp = ReactorInputs(
            power_w=2500.0, pressure_mtorr=5.0,
            flows_sccm={"Ar": 100.0, "O2": 50.0, "SiH4": 30.0},
            rf_bias_v=80.0,
        )
        return GlobalPlasmaSolver(inp).solve()
    except Exception:
        from types import SimpleNamespace
        # Mock IEADF exposing only what L1 reads.
        ieadf = SimpleNamespace(mean_angle_deg=lambda: 1.22)
        return SimpleNamespace(
            T_e=2.529, n_e=2.031e17, n_g=9.656e19, V_sheath=91.57,
            V_plasma=11.57, m_ion_eff_amu=32.44, ieadf=ieadf,
            mean_ion_energy_ev=91.57,
        )


def _demo() -> None:
    """Resolve transport down a 10:1 aspect-ratio, 100 nm-wide trench."""
    plasma = _make_mock_plasma_state()

    geom = FeatureGeometry(aspect_ratio=10.0, w0_m=100e-9, n_z=200)
    params = TransportParams(sticking_coeff=0.10)
    state = FeatureTransportSolver(plasma, geom, params).solve()

    print("=" * 66)
    print(" [L1] FEATURE-LOCAL TRANSPORT  -  10:1 trench (w0 = 100 nm)")
    print("=" * 66)
    print(f" Plasma in : T_e={plasma.T_e:.3g} eV, n_e={plasma.n_e:.3g} m^-3, "
          f"n_g={plasma.n_g:.3g} m^-3, V_sh={plasma.V_sheath:.3g} V")
    print(f" Trench    : H={geom.depth_m*1e9:.0f} nm, AR={geom.aspect_ratio:.0f}, "
          f"s_c={params.sticking_coeff:.2f}")
    print("-" * 66)
    for k, v in state.summary().items():
        print(f" {k:<22}: {v:.4g}")
    print("-" * 66)

    # Spot the profiles at mouth / mid / bottom.
    idx = [0, len(state.z) // 2, -1]
    labels = ["mouth", "mid", "bottom"]
    print(f" {'z-station':<10}{'z[nm]':>9}{'Kn':>12}{'n_g/n0':>10}"
          f"{'phi_s[V]':>11}{'dtheta[deg]':>13}")
    for lbl, k in zip(labels, idx):
        print(f" {lbl:<10}{state.z[k]*1e9:>9.1f}{state.Kn[k]:>12.3e}"
              f"{state.n_g[k]/state.n_g[0]:>10.3f}{state.phi_s[k]:>11.3f}"
              f"{np.degrees(state.dtheta_charge[k]):>13.3f}")
    print("-" * 66)

    # Physical sanity flags.
    regime = "free-molecular" if state.Kn[0] > 10 else (
        "transitional" if state.Kn[0] > 0.01 else "continuum")
    print(f" Flow regime (Kn>>1 expected) : {regime}  (Kn_top={state.Kn[0]:.2e})")
    print(f" Charging dipole sign         : top={state.phi_s[0]:+.2f} V -> "
          f"bottom={state.phi_s[-1]:+.2f} V "
          f"({'OK: bottom more positive' if state.phi_s[-1] > state.phi_s[0] else 'CHECK'})")
    print("=" * 66)


if __name__ == "__main__":
    _demo()
