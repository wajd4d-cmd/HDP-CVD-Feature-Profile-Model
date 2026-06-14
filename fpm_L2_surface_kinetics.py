"""
fpm_L2_surface_kinetics.py
==========================

[L2] Surface Kinetics & Heat Solver for the HDP-CVD Feature Profile Model (FPM).

Consumes the L0 ``PlasmaState`` and L1 ``TransportState`` for one trench and
returns the z-resolved surface thermodynamics and adatom kinetics:

    T(z)   : local surface temperature                       [K]
    s_c(z) : dynamic Langmuir-Kisliuk + ion-enhanced sticking [-]
    D_s(z) : Arrhenius adatom surface diffusivity            [m^2/s]
    L_s(z) : adatom diffusion length sqrt(D_s tau_inc)        [m]
    n_s(z) : adatom surface concentration                    [m^-2]

Physics implemented
-------------------
1. Thermal field. The MVP's single global "thermal drift window" is replaced by
   two coupled pieces:
     (a) Global wafer temperature from plasma power loading
            T_bulk = T_chuck + (Gamma_i E_i) * (L_wafer/kappa_Si + 1/h_chuck)
         i.e. the ion power density conducts through the wafer to the back-side
         cooled chuck. T_bulk now *responds* to the plasma instead of being fixed.
     (b) Local surface temperature T(z) from a steady-state fin / extended-surface
         balance along the trench wall:
            kappa*delta * T'' - G_tbc (T - T_bulk) + q_surf(z) = 0
         q_surf = q_ion(z) + q_rxn(z) - q_rad(z). Because Si/SiO2 are excellent
         heat spreaders at the micron scale, the feature-local Delta T is small;
         the solver resolves it exactly and reports its magnitude.

2. Dynamic sticking (Langmuir-Kisliuk + ion activation), per the FPM blueprint:
        s_c = s0 (1-Theta) exp(-E_a/kT) / (1 + K_K Theta)  +  eta_ia Gamma_i sigma_ia
   with coverage Theta = n_s/n_s_max. The first term is thermal adsorption gated
   by a Kisliuk mobile-precursor factor; the second is ion-enhanced activation.

3. Adatom transport. Steady reaction-diffusion on the wall:
        D_s(T) n_s'' + s_c Gamma_g - n_s/tau_inc - n_s nu_d exp(-E_des/kT) = 0
   (zero-flux ends). D_s(T) = (a0^2 nu0 / 4) exp(-E_diff/kT); L_s = sqrt(D_s tau_inc).

Because s_c, T and n_s are mutually coupled (Theta <-> s_c <-> q_rxn <-> T), the
solver runs a damped fixed-point iteration to self-consistency.

Author role: Principal Scientific Computing Engineer
Layer:       L2 (this file implements ONLY L2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy import constants as const
from scipy.integrate import solve_bvp

# =============================================================================
# 1. PHYSICAL CONSTANTS (CODATA via scipy.constants)
# =============================================================================
E_CHARGE: float = const.e                       # elementary charge       [C]
M_E: float = const.m_e                          # electron mass           [kg]
K_B: float = const.k                            # Boltzmann constant      [J/K]
AMU: float = const.physical_constants["atomic mass constant"][0]  # [kg]
EV_TO_J: float = const.e                         # 1 eV in Joules         [J/eV]
SIGMA_SB: float = const.Stefan_Boltzmann         # Stefan-Boltzmann       [W/m^2/K^4]


# =============================================================================
# 2. PARAMETER CONTAINER
# =============================================================================
@dataclass
class SurfaceParams:
    """
    Surface-kinetic and thermal parameters. Defaults are representative of an
    HDP-CVD SiO2 process; calibrate against fab metrology. Literature ranges are
    quoted inline.
    """
    # --- Thermal (global wafer + local fin) -------------------------------
    T_chuck_k: float = 600.0          # back-side chuck setpoint        [K] (300-700)
    kappa_si: float = 150.0           # Si thermal conductivity         [W/m/K]
    wafer_thickness_m: float = 775e-6 # 300 mm wafer thickness          [m]
    h_chuck: float = 1500.0           # back-side (He) conductance       [W/m^2/K] (5e2-3e3)
    kappa_film: float = 1.4           # SiO2 film conductivity          [W/m/K] (1.2-1.4)
    film_delta_m: float = 50e-9       # effective conducting wall layer [m]
    G_tbc: float = 1.0e8              # wall->substrate boundary cond.  [W/m^2/K] (1e7-1e9)
    emissivity: float = 0.7           # surface emissivity              [-]
    T_amb_k: float = 300.0            # radiative ambient (chamber wall) [K]

    # --- Sticking (Langmuir-Kisliuk + ion) --------------------------------
    s0: float = 0.9                   # bare-surface sticking prefactor  [-]
    E_ads_ev: float = 0.10            # adsorption activation            [eV] (0.05-0.3)
    K_kisliuk: float = 2.0            # Kisliuk precursor mobility factor [-] (0.5-5)
    eta_ia_s: float = 2.0e-5          # ion-activation correlation time  [s]
    sigma_ia_m2: float = 1.0e-19      # ion-activation cross-section      [m^2]

    # --- Adatom kinetics --------------------------------------------------
    n_s_max: float = 1.0e19           # saturation site density          [m^-2] (~1e15/cm^2)
    tau_inc_s: float = 1.0e-4         # incorporation time               [s]
    nu_desorb: float = 1.0e13         # desorption attempt frequency     [s^-1]
    E_des_ev: float = 1.00            # desorption energy                [eV] (0.8-1.5)
    a0_jump_m: float = 3.0e-10        # adatom hop distance              [m]
    nu0_diff: float = 1.0e13          # diffusion attempt frequency      [s^-1]
    E_diff_ev: float = 0.50           # surface-diffusion barrier        [eV] (0.3-0.8)
    deltaH_dep_ev: float = 2.0        # exothermic heat per incorporated unit [eV] (1-3)

    # --- Neutral flux ------------------------------------------------------
    neutral_mass_amu: float = 32.0    # depositing-precursor mass        [amu]
    gas_temp_k: float = 500.0         # neutral gas temperature          [K]

    # --- Solver -----------------------------------------------------------
    max_iter: int = 40                # fixed-point iterations
    relax: float = 0.5                # under-relaxation factor          [-]
    tol: float = 1e-6                 # relative convergence tolerance


# =============================================================================
# 3. OUTPUT CONTAINER
# =============================================================================
@dataclass
class SurfaceState:
    """z-resolved L2 surface state handed to L3 (rate assembly)."""
    z: np.ndarray                     # axial mesh (mouth->bottom)      [m]
    T: np.ndarray                     # local surface temperature       [K]
    s_c: np.ndarray                   # dynamic sticking coefficient    [-]
    D_s: np.ndarray                   # surface diffusivity             [m^2/s]
    L_s: np.ndarray                   # adatom diffusion length         [m]
    n_s: np.ndarray                   # adatom concentration            [m^-2]
    coverage: np.ndarray             # fractional coverage Theta        [-]
    T_bulk: float                     # global wafer surface temperature [K]
    Gamma_i: float                    # ion flux to surface             [m^-2 s^-1]
    n_iter: int                       # fixed-point iterations used
    residual: float                   # final relative residual

    def summary(self) -> Dict[str, float]:
        """Scalar digest for logging / dashboards."""
        return {
            "T_bulk_K": self.T_bulk,
            "T_top_K": float(self.T[0]),
            "T_bottom_K": float(self.T[-1]),
            "dT_local_K": float(self.T.max() - self.T.min()),
            "s_c_top": float(self.s_c[0]),
            "s_c_bottom": float(self.s_c[-1]),
            "D_s_top_m2s": float(self.D_s[0]),
            "L_s_top_nm": float(self.L_s[0] * 1e9),
            "coverage_top": float(self.coverage[0]),
            "coverage_bottom": float(self.coverage[-1]),
            "n_s_top_m2": float(self.n_s[0]),
            "n_iter": self.n_iter,
            "residual": self.residual,
        }


# =============================================================================
# 4. SURFACE KINETICS & HEAT SOLVER
# =============================================================================
class SurfaceKineticsSolver:
    """
    Self-consistent surface temperature + adatom kinetics for one trench.

    Parameters
    ----------
    plasma : PlasmaState (L0)     -- T_e, n_e, V_sheath, m_ion_eff_amu,
                                     mean_ion_energy_ev, ieadf.mean_angle_deg()
    transport : TransportState (L1) -- z, n_g(z), width(z)
    geometry : FeatureGeometry (L1)
    params : SurfaceParams, optional
    """

    def __init__(self, plasma, transport, geometry,
                 params: Optional[SurfaceParams] = None) -> None:
        self.plasma = plasma
        self.tr = transport
        self.geom = geometry
        self.p = params if params is not None else SurfaceParams()

        self.z = np.asarray(transport.z, dtype=float)
        self.H = float(self.z[-1])

        # Neutral thermal speed for the precursor flux [m/s].
        m_n = self.p.neutral_mass_amu * AMU
        self.v_th = np.sqrt(8.0 * K_B * self.p.gas_temp_k / (np.pi * m_n))

        # Ion flux (Bohm) from L0 [m^-2 s^-1] and impact energy [J].
        m_ion = self.plasma.m_ion_eff_amu * AMU
        self.u_B = np.sqrt(E_CHARGE * self.plasma.T_e / m_ion)
        self.Gamma_i = self.plasma.n_e * self.u_B
        self.E_i_J = self.plasma.mean_ion_energy_ev * EV_TO_J

        # Mean ion impact angle (sets the grazing sidewall heat fraction).
        self.theta_mean = np.radians(self._read_mean_ion_angle())

        # Neutral precursor flux profile Gamma_g(z) = (1/4) n_g(z) v_th.
        self.Gamma_g = 0.25 * np.asarray(transport.n_g, dtype=float) * self.v_th

    # ----------------------------------------------------------- helpers
    def _read_mean_ion_angle(self) -> float:
        ieadf = getattr(self.plasma, "ieadf", None)
        if ieadf is not None and hasattr(ieadf, "mean_angle_deg"):
            return float(ieadf.mean_angle_deg())
        return 1.0

    # ----------------------------------------------------- global wafer T
    def global_wafer_temperature(self) -> float:
        r"""
        Wafer surface temperature from plasma power loading:
            T_bulk = T_chuck + P_ion (L_wafer/kappa_Si + 1/h_chuck)
        with the planar ion power density P_ion = Gamma_i * E_i [W/m^2].
        This replaces the MVP's fixed thermal window with a plasma-responsive
        absolute temperature.
        """
        P_ion = self.Gamma_i * self.E_i_J
        R_wafer = self.p.wafer_thickness_m / self.p.kappa_si + 1.0 / self.p.h_chuck
        return self.p.T_chuck_k + P_ion * R_wafer

    # --------------------------------------------------------- kinetics
    def diffusivity(self, T: np.ndarray) -> np.ndarray:
        """Arrhenius adatom surface diffusivity D_s(T) = (a0^2 nu0/4) exp(-E_diff/kT)."""
        pref = self.p.a0_jump_m ** 2 * self.p.nu0_diff / 4.0
        return pref * np.exp(-self.p.E_diff_ev * EV_TO_J / (K_B * T))

    def sticking(self, T: np.ndarray, theta_cov: np.ndarray) -> np.ndarray:
        r"""
        Langmuir-Kisliuk precursor-mediated sticking with ion-enhanced activation:

            s_c = s0 (1-Theta) exp(-E_a/kT) / (1 + K_K Theta)
                  + eta_ia Gamma_i sigma_ia

        Returns s_c clipped to the physical range [0, 1].
        """
        thermal = (self.p.s0 * (1.0 - theta_cov)
                   * np.exp(-self.p.E_ads_ev * EV_TO_J / (K_B * T))
                   / (1.0 + self.p.K_kisliuk * theta_cov))
        ion = self.p.eta_ia_s * self.Gamma_i * self.p.sigma_ia_m2
        return np.clip(thermal + ion, 0.0, 1.0)

    # ---------------------------------------------------- adatom transport
    def _solve_adatom_bvp(self, T: np.ndarray, s_c: np.ndarray) -> np.ndarray:
        r"""
        Steady reaction-diffusion for n_s(z):
            D_s n_s'' + F_ads - k_loss n_s = 0,
            F_ads = s_c Gamma_g,  k_loss = 1/tau_inc + nu_d exp(-E_des/kT).
        Non-dimensionalized (zeta=z/H, ntil=n_s/n_ref) with zero-flux ends.
        """
        H = self.H
        D_s = self.diffusivity(T)
        F_ads = s_c * self.Gamma_g
        k_loss = (1.0 / self.p.tau_inc_s
                  + self.p.nu_desorb * np.exp(-self.p.E_des_ev * EV_TO_J / (K_B * T)))

        # Reference scale from the local algebraic balance (avoids stiffness).
        n_ref = float(np.max(F_ads / k_loss))
        n_ref = max(n_ref, 1.0)

        # Interpolators for arbitrary zeta queried by solve_bvp.
        zeta_grid = self.z / H

        def interp(arr, zeta):
            return np.interp(zeta, zeta_grid, arr)

        def odes(zeta: np.ndarray, y: np.ndarray) -> np.ndarray:
            Dl = interp(D_s, zeta)
            kl = interp(k_loss, zeta)
            Fa = interp(F_ads, zeta)
            ntil, dntil = y
            d2 = (H ** 2 / Dl) * (kl * ntil - Fa / n_ref)
            return np.vstack((dntil, d2))

        def bcs(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
            return np.array([ya[1], yb[1]])           # zero-flux both ends

        y0 = np.vstack((interp(F_ads / k_loss, zeta_grid) / n_ref,
                        np.zeros_like(zeta_grid)))
        sol = solve_bvp(odes, bcs, zeta_grid, y0, tol=1e-8, max_nodes=20000)
        if not sol.success:
            raise RuntimeError(f"Adatom BVP failed: {sol.message}")
        return np.clip(sol.sol(zeta_grid)[0] * n_ref, 0.0, None)

    # ------------------------------------------------------------- heat
    def _solve_heat_bvp(self, T_prev: np.ndarray, s_c: np.ndarray,
                        T_bulk: float) -> np.ndarray:
        r"""
        Steady fin balance for the local surface temperature T(z):
            kappa*delta T'' - G_tbc (T - T_bulk) + q_surf(z) = 0,
        with phi = T - T_bulk:  phi'' - beta^2 phi = -q_surf/(kappa delta),
        beta^2 = G_tbc/(kappa delta). Non-dimensionalized in zeta = z/H.

        q_surf = q_ion(z) + q_rxn(z) - q_rad(z):
          q_ion  -- grazing ion power on the vertical wall, Gamma_i E_i sin(theta);
          q_rxn  -- exothermic incorporation, deltaH_dep * s_c Gamma_g;
          q_rad  -- grey-body loss, eps sigma_SB (T_prev^4 - T_amb^4).
        The concentrated bottom (normal) ion flux enters as a localized source.
        """
        H = self.H
        kd = self.p.kappa_film * self.p.film_delta_m
        beta2 = self.p.G_tbc / kd
        zeta_grid = self.z / H

        # Sidewalls receive only the grazing ion-power component; the bottom
        # face (within ~one trench width of the tip) receives the full normal
        # flux. Both enter as the distributed source -- heat leaves transversely
        # through G_tbc, NOT by lateral conduction along the thin wall.
        w_ref = float(self.tr.width[-1])
        bottom_window = np.exp(-(self.H - self.z) / w_ref)
        sin_t = np.sin(self.theta_mean)
        q_ion = self.Gamma_i * self.E_i_J * (sin_t + (1.0 - sin_t) * bottom_window)
        q_rxn = self.p.deltaH_dep_ev * EV_TO_J * s_c * self.Gamma_g
        q_rad = self.p.emissivity * SIGMA_SB * (T_prev ** 4 - self.p.T_amb_k ** 4)
        q_surf = q_ion + q_rxn - q_rad                                # [W/m^2]


        def interp(arr, zeta):
            return np.interp(zeta, zeta_grid, arr)

        def odes(zeta: np.ndarray, y: np.ndarray) -> np.ndarray:
            f = interp(q_surf, zeta) / kd
            phi, dphi = y
            d2 = (beta2 * H ** 2) * phi - f * H ** 2
            return np.vstack((dphi, d2))

        def bcs(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
            # mouth sunk to bulk (phi(0)=0); adiabatic tip (phi'(H)=0). All
            # surface heat (incl. the bottom face) leaves transversely via G_tbc.
            return np.array([ya[0], yb[1]])

        y0 = np.vstack((q_surf / (kd * beta2), np.zeros_like(zeta_grid)))
        sol = solve_bvp(odes, bcs, zeta_grid, y0, tol=1e-8, max_nodes=20000)
        if not sol.success:
            raise RuntimeError(f"Heat BVP failed: {sol.message}")
        return T_bulk + sol.sol(zeta_grid)[0]

    # ----------------------------------------------------------------- solve
    def solve(self) -> SurfaceState:
        """Run the coupled (T, s_c, n_s) fixed-point solve to self-consistency."""
        T_bulk = self.global_wafer_temperature()
        z = self.z

        T = np.full_like(z, T_bulk)                  # init isothermal at bulk
        n_s = np.zeros_like(z)
        theta = np.zeros_like(z)
        residual = np.inf
        it = 0

        for it in range(1, self.p.max_iter + 1):
            s_c = self.sticking(T, theta)
            n_s_new = self._solve_adatom_bvp(T, s_c)
            theta_new = np.clip(n_s_new / self.p.n_s_max, 0.0, 0.999)
            T_new = self._solve_heat_bvp(T, s_c, T_bulk)

            # Under-relaxed updates.
            r = self.p.relax
            n_s = (1 - r) * n_s + r * n_s_new
            theta = (1 - r) * theta + r * theta_new
            T_upd = (1 - r) * T + r * T_new

            # Relative residual on the coupled state (T + coverage).
            dT = np.linalg.norm(T_upd - T) / (np.linalg.norm(T) + 1e-30)
            dC = np.linalg.norm(theta - theta_new) / (np.linalg.norm(theta) + 1e-30)
            T = T_upd
            residual = max(dT, dC)
            if residual < self.p.tol:
                break

        s_c = self.sticking(T, theta)
        D_s = self.diffusivity(T)
        L_s = np.sqrt(D_s * self.p.tau_inc_s)

        return SurfaceState(
            z=z, T=T, s_c=s_c, D_s=D_s, L_s=L_s, n_s=n_s, coverage=theta,
            T_bulk=T_bulk, Gamma_i=self.Gamma_i, n_iter=it, residual=residual,
        )


# =============================================================================
# 5. DEMONSTRATION / SELF-TEST  (mocks L0 + L1)
# =============================================================================
def _make_mock_states():
    """Prefer real L0/L1; otherwise duck-typed stand-ins matching their demos."""
    try:
        from fpm_L0_global_plasma import GlobalPlasmaSolver, ReactorInputs
        from fpm_L1_feature_transport import (FeatureTransportSolver,
                                              FeatureGeometry, TransportParams)
        inp = ReactorInputs(power_w=2500.0, pressure_mtorr=5.0,
                            flows_sccm={"Ar": 100.0, "O2": 50.0, "SiH4": 30.0},
                            rf_bias_v=80.0)
        plasma = GlobalPlasmaSolver(inp).solve()
        geom = FeatureGeometry(aspect_ratio=10.0, w0_m=100e-9, n_z=200)
        transport = FeatureTransportSolver(plasma, geom, TransportParams(0.10)).solve()
        return plasma, transport, geom
    except Exception:
        from types import SimpleNamespace
        ieadf = SimpleNamespace(mean_angle_deg=lambda: 1.22)
        plasma = SimpleNamespace(
            T_e=2.529, n_e=2.031e17, n_g=9.656e19, V_sheath=91.57,
            m_ion_eff_amu=32.44, mean_ion_energy_ev=91.57, ieadf=ieadf)
        z = np.linspace(0.0, 1.0e-6, 200)
        # Neutral profile mimicking L1 (mouth -> ~11% at bottom).
        n_g = 9.656e19 * (0.113 + (1 - 0.113) * np.exp(-z / (z[-1] / 2.7)))
        width = np.full_like(z, 100e-9)
        transport = SimpleNamespace(z=z, n_g=n_g, width=width)
        geom = SimpleNamespace(aspect_ratio=10.0, w0_m=100e-9, depth_m=1.0e-6)
        return plasma, transport, geom


def _demo() -> None:
    """Resolve surface kinetics + heat down a 10:1, 100 nm trench."""
    plasma, transport, geom = _make_mock_states()
    state = SurfaceKineticsSolver(plasma, transport, geom, SurfaceParams()).solve()

    print("=" * 70)
    print(" [L2] SURFACE KINETICS & HEAT  -  10:1 trench (w0 = 100 nm)")
    print("=" * 70)
    print(f" Ion flux Gamma_i        : {state.Gamma_i:.3e} m^-2 s^-1")
    print(f" Ion power density        : {state.Gamma_i * (plasma.mean_ion_energy_ev*EV_TO_J):.3e} W/m^2")
    print(f" Converged in {state.n_iter} iters (residual {state.residual:.2e})")
    print("-" * 70)
    for k, v in state.summary().items():
        print(f" {k:<20}: {v:.5g}")
    print("-" * 70)
    idx = [0, len(state.z) // 2, -1]
    labels = ["mouth", "mid", "bottom"]
    print(f" {'station':<9}{'z[nm]':>8}{'T[K]':>10}{'s_c':>10}"
          f"{'D_s[m2/s]':>13}{'L_s[nm]':>10}{'Theta':>11}")
    for lbl, k in zip(labels, idx):
        print(f" {lbl:<9}{state.z[k]*1e9:>8.1f}{state.T[k]:>10.4f}{state.s_c[k]:>10.4f}"
              f"{state.D_s[k]:>13.3e}{state.L_s[k]*1e9:>10.3f}{state.coverage[k]:>11.3e}")
    print("-" * 70)
    print(f" Global wafer rise above chuck : "
          f"{state.T_bulk - SurfaceParams().T_chuck_k:.3f} K "
          f"(chuck {SurfaceParams().T_chuck_k:.0f} K -> wafer {state.T_bulk:.2f} K)")
    print(f" Feature-local Delta T          : {state.T.max()-state.T.min():.3e} K "
          f"(micron-scale conduction -> near-isothermal)")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
