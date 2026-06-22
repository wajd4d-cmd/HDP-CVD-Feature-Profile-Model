"""
fpm_L3_rate_assembly.py
=======================

[L3] Rate Assembly Solver for the HDP-CVD Feature Profile Model (FPM).

Consumes the L0 ``PlasmaState``, L1 ``TransportState`` and L2 ``SurfaceState``
for one trench and assembles the competing surface fluxes into the net normal
growth velocity along the trench depth z:

    R_D(z)     : gross deposition rate                         [m/s]
    R_S(z)     : angle-resolved sputter rate                   [m/s]
    R_redep(z) : redeposited (re-sputtered) material           [m/s]
    div_Js(z)  : surface-diffusion divergence contribution     [m/s]
    V_n(z)     : net normal velocity  R_D - R_S + F[R_S] - O*div(J_s)
    R_DS(z)    : local deposition/sputter ratio  R_D / R_S      [-]

Physics implemented
-------------------
1. Sputter (Yamamura-Tawara). The energy yield Y(E,0) uses the Thomas-Fermi
   reduced nuclear stopping with the full Yamamura prefactor, threshold, and
   (small) electronic-stopping correction. The angular factor uses the Yamamura
   form Y(theta)/Y(0) = cos(theta)^-f exp[ f cos(theta_opt)(1 - sec theta) ].
   The TRUE local incidence angle at each z is built by folding the L0 IEADF
   angular distribution with the L1 charge-deflection angle dtheta_charge(z) and
   the local surface orientation (vertical sidewall -> grazing; bottom -> normal),
   then integrating Y*cos(theta_inc) over the IEADF.

2. Deposition. R_D = Omega * s_c(z) * Gamma_g(z), with the L2 dynamic sticking
   coefficient and the L1 transmission-attenuated neutral flux
   Gamma_g(z) = (1/4) n_g(z) v_th.

3. Redeposition. Sputtered atoms leave with a cosine distribution and strike the
   opposing wall; the parallel-plate differential view factor
   K(z,z') = (1/2) w^2 / (w^2 + (z-z')^2)^{3/2} redistributes them. A coarse
   5-zone view-factor matrix F_{k->k'} (the blueprint artifact) is also reported.

4. Surface diffusion. div(J_s) = d/dz[ -D_s dn_s/dz ] from L2 fields contributes
   -Omega*div(J_s) to V_n (Fickian; curvature/Gibbs-Thomson enters at L4).

Author role: Principal Scientific Computing Engineer
Layer:       L3 (this file implements ONLY L3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import constants as const

# Module logger for the bias -> ion-energy -> sputter/deposition diagnostic.
# Silent by default; enable with logging.getLogger("fpm.L3").setLevel(logging.DEBUG).
log = logging.getLogger("fpm.L3")

# =============================================================================
# 1. PHYSICAL CONSTANTS (CODATA via scipy.constants)
# =============================================================================
E_CHARGE: float = const.e                        # elementary charge      [C]
K_B: float = const.k                             # Boltzmann constant     [J/K]
AMU: float = const.physical_constants["atomic mass constant"][0]  # [kg]
EV_TO_J: float = const.e                          # 1 eV in Joules        [J/eV]


# =============================================================================
# 2. MATERIAL / PARAMETER CONTAINERS
# =============================================================================
@dataclass(frozen=True)
class SputterTarget:
    """
    Yamamura-Tawara projectile/target pair. Defaults: Ar+ onto an effective
    SiO2 target atom (mass/charge stoichiometry-averaged). U_s and Q are the
    primary calibration handles against fab sputter data.
    """
    M1_amu: float = 39.948       # projectile mass (Ar)               [amu]
    Z1: float = 18.0             # projectile atomic number (Ar)      [-]
    M2_amu: float = 20.0         # effective target mass (SiO2 avg)   [amu]
    Z2: float = 10.0             # effective target Z (SiO2 avg)      [-]
    U_s_ev: float = 4.0          # surface binding energy             [eV] (4-8)
                                 # Calibrated to the lower end of the SiO2 range
                                 # so the IEADF-averaged S/D ratio reaches the
                                 # ~0.1-0.3 responsive band at typical bias; at
                                 # U_s=6.4 the threshold factor suppressed sputter
                                 # to S/D~0.03 and the profile was bias-insensitive.
    Q: float = 1.0               # dimensionless yield factor         [-] (~0.5-1.5)
    s_threshold_exp: float = 2.5 # threshold exponent in (1-sqrt(Eth/E))^s
    f_angular: float = 2.0       # Yamamura angular exponent f        [-]
    theta_opt_deg: float = 70.0  # angle of maximum yield             [deg] (60-80)


@dataclass
class RateParams:
    """Assembly parameters not carried by upstream states."""
    omega_m3: float = 4.5e-29    # deposited-unit volume (SiO2 ~4.5e-29) [m^3]
    s_reemit: float = 0.0        # re-emission prob. of redep flux    [-] (0-0.5)
    neutral_mass_amu: float = 32.0  # depositing precursor mass       [amu]
    gas_temp_k: float = 500.0    # neutral gas temperature            [K]
    n_angle: int = 91            # IEADF angular quadrature points    [-]
    n_zones: int = 5             # redeposition zones (blueprint)     [-]


# =============================================================================
# 3. OUTPUT CONTAINER
# =============================================================================
@dataclass
class RateState:
    """z-resolved L3 rate fields handed to L4 (level-set advance)."""
    z: np.ndarray                # axial mesh (mouth->bottom)         [m]
    R_D: np.ndarray              # gross deposition rate              [m/s]
    R_S: np.ndarray              # sputter rate                       [m/s]
    R_redep: np.ndarray          # redeposited material              [m/s]
    div_Js: np.ndarray           # surface-diffusion divergence       [m^-2 s^-1]
    V_n: np.ndarray              # net normal velocity                [m/s]
    R_DS: np.ndarray             # local D/S ratio                    [-]
    Y_bar: np.ndarray            # IEADF-averaged sputter yield       [atoms/ion]
    theta_inc_eff: np.ndarray    # effective incidence angle          [rad]
    zone_view_factor: np.ndarray # 5x5 redeposition view-factor matrix [-]

    def summary(self) -> Dict[str, float]:
        """Scalar digest for logging / dashboards."""
        return {
            "R_D_top_nm_min": float(self.R_D[0] * 1e9 * 60),
            "R_D_bottom_nm_min": float(self.R_D[-1] * 1e9 * 60),
            "R_S_bottom_nm_min": float(self.R_S[-1] * 1e9 * 60),
            "Y_bar_bottom": float(self.Y_bar[-1]),
            "R_DS_top": float(self.R_DS[0]),
            "R_DS_bottom": float(self.R_DS[-1]),
            "V_n_top_nm_min": float(self.V_n[0] * 1e9 * 60),
            "V_n_bottom_nm_min": float(self.V_n[-1] * 1e9 * 60),
            "redep_frac_of_sputter": float(
                np.trapezoid(self.R_redep, self.z)
                / (np.trapezoid(self.R_S, self.z) + 1e-300)),
        }


# =============================================================================
# 4. RATE ASSEMBLY SOLVER
# =============================================================================
class RateAssemblySolver:
    """
    Assemble deposition, angle-resolved sputter, redeposition and surface
    diffusion into V_n(z) for one trench.

    Parameters
    ----------
    plasma : PlasmaState (L0)     -- n_e, T_e, m_ion_eff_amu, mean_ion_energy_ev,
                                     ieadf (angle_pdf / theta_axis / mean_angle_deg)
    transport : TransportState (L1) -- z, n_g(z), width(z), dtheta_charge(z)
    surface : SurfaceState (L2)   -- s_c(z), D_s(z), n_s(z), Gamma_i (optional)
    target : SputterTarget, optional
    params : RateParams, optional
    """

    def __init__(self, plasma, transport, surface,
                 target: Optional[SputterTarget] = None,
                 params: Optional[RateParams] = None) -> None:
        self.plasma = plasma
        self.tr = transport
        self.sf = surface
        self.t = target if target is not None else SputterTarget()
        self.p = params if params is not None else RateParams()

        self.z = np.asarray(transport.z, dtype=float)
        self.H = float(self.z[-1])
        self.width = np.asarray(transport.width, dtype=float)
        self.dtheta = np.asarray(transport.dtheta_charge, dtype=float)

        # Ion flux (Bohm) and mean impact energy from L0.
        m_ion = self.plasma.m_ion_eff_amu * AMU
        u_B = np.sqrt(E_CHARGE * self.plasma.T_e / m_ion)
        self.Gamma_i = getattr(surface, "Gamma_i", self.plasma.n_e * u_B)
        self.E_i_eV = float(self.plasma.mean_ion_energy_ev)

        # Depositing-precursor flux Gamma_g(z) = (1/4) n_dep(z) v_th. Only the
        # Si-bearing fraction deposits SiO2; fall back to total n_g * f_precursor
        # for duck-typed mock states that predate the n_dep field.
        m_n = self.p.neutral_mass_amu * AMU
        self.v_th = np.sqrt(8.0 * K_B * self.p.gas_temp_k / (np.pi * m_n))
        n_dep = getattr(transport, "n_dep", None)
        if n_dep is None:
            n_dep = (np.asarray(transport.n_g, dtype=float)
                     * float(getattr(self.plasma, "f_precursor", 1.0)))
        self.Gamma_g = 0.25 * np.asarray(n_dep, dtype=float) * self.v_th

        # IEADF angular distribution (forward-peaked) -> quadrature.
        self.theta_ion, self.p_ion = self._ion_angle_distribution()

    # --------------------------------------------------- IEADF angular pdf
    def _ion_angle_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (theta_ion grid [rad], normalized pdf). Pulls the true angular
        marginal from the L0 IEADF at the mean ion energy; falls back to a narrow
        distribution if the IEADF is not exposed.
        """
        theta = np.linspace(0.0, np.pi / 2.0, self.p.n_angle)
        ieadf = getattr(self.plasma, "ieadf", None)
        if ieadf is not None and hasattr(ieadf, "angle_pdf"):
            try:
                p = np.asarray(ieadf.angle_pdf(theta, self.E_i_eV), dtype=float)
            except Exception:
                p = None
        else:
            p = None
        if p is None or not np.all(np.isfinite(p)) or np.trapezoid(p, theta) <= 0:
            # Fallback: collisionless IADF g(th) ~ (E/Ti) sinth costh exp(-(E/Ti)sin^2)
            r = self.E_i_eV / 0.05
            p = r * np.sin(theta) * np.cos(theta) * np.exp(-r * np.sin(theta) ** 2)
        p = p / np.trapezoid(p, theta)
        return theta, p

    # ----------------------------------------------- Yamamura-Tawara yield
    def yamamura_yield(self, E_eV: float, theta_inc: np.ndarray) -> np.ndarray:
        r"""
        Yamamura-Tawara sputter yield Y(E, theta_inc) [atoms/ion].

        Energy term (normal incidence):
            Y0 = 0.042 (Q alpha*/U_s) * S_n(E)/(1 + k_e eps^0.3)
                 * (1 - sqrt(E_th/E))^s
        with reduced energy eps (Lindhard), Thomas-Fermi reduced nuclear stopping
        s_n(eps), nuclear stopping cross-section S_n [eV A^2], Lindhard electronic
        coefficient k_e, and Yamamura threshold E_th.

        Angular term:
            Y(theta)/Y(0) = cos(theta)^-f exp[ f cos(theta_opt)(1 - sec theta) ]
        """
        t = self.t
        M1, Z1, M2, Z2, U_s = t.M1_amu, t.Z1, t.M2_amu, t.Z2, t.U_s_ev
        mu = M2 / M1

        if E_eV <= 0.0:
            return np.zeros_like(theta_inc)

        # Lindhard reduced energy.
        zfac = Z1 * Z2 * np.sqrt(Z1 ** 0.667 + Z2 ** 0.667)
        eps = 0.03255 / zfac * (M2 / (M1 + M2)) * E_eV

        # Thomas-Fermi reduced nuclear stopping (Matsunami fit).
        se = np.sqrt(eps)
        s_n = (3.441 * se * np.log(eps + 2.718)
               / (1.0 + 6.355 * se + eps * (6.882 * se - 1.708)))
        # Nuclear stopping cross-section [eV * A^2].
        S_n = 84.78 * (Z1 * Z2 / np.sqrt(Z1 ** 0.667 + Z2 ** 0.667)) \
            * (M1 / (M1 + M2)) * s_n

        # Lindhard electronic coefficient (small at ~100 eV; included for rigor).
        k_e = (0.079 * (Z1 ** 0.667 * Z2 ** 0.5 * (M1 + M2) ** 1.5)
               / ((Z1 ** 0.667 + Z2 ** 0.667) ** 0.75 * M1 ** 1.5 * M2 ** 0.5))

        # Yamamura threshold energy.
        E_th = U_s * (1.9 + 3.8 / mu + 0.134 * mu ** 1.24)

        # Energy-independent factor alpha*(M2/M1) (Yamamura-Tawara fit).
        alpha_star = 0.249 * mu ** 0.56 + 0.0035 * mu ** 1.5

        bracket = max(1.0 - np.sqrt(E_th / E_eV), 0.0) if E_eV > E_th else 0.0
        Y0 = (0.042 * t.Q * alpha_star / U_s
              * S_n / (1.0 + k_e * eps ** 0.3)
              * bracket ** t.s_threshold_exp)

        # Angular dependence (peaks at theta_opt; clipped away from 90 deg).
        th = np.clip(theta_inc, 0.0, np.radians(89.0))
        theta_opt = np.radians(t.theta_opt_deg)
        f = t.f_angular
        y_ang = np.cos(th) ** (-f) * np.exp(f * np.cos(theta_opt)
                                            * (1.0 - 1.0 / np.cos(th)))
        return Y0 * y_ang

    # ------------------------------------------------ local incidence angle
    def _surface_orientation_window(self) -> np.ndarray:
        """
        Bottom-face weighting W(z) in [0,1]: ~0 on the vertical sidewall, ->1
        within ~one trench width of the tip (where the surface becomes the
        horizontal bottom). Smoothly blends grazing- and normal-incidence.
        """
        w_ref = float(self.width[-1])
        return np.exp(-(self.H - self.z) / w_ref)

    def _incidence_angles(self) -> np.ndarray:
        r"""
        Build the 2-D incidence-angle field theta_inc[z, a] for ion polar angle
        theta_ion[a] (from the IEADF) and depth z, folding in the L1 deflection:

            alpha(z, a) = theta_ion[a] + dtheta_charge(z)        (ion tilt)
            sidewall (W~0):  theta_inc = pi/2 - alpha             (grazing)
            bottom   (W~1):  theta_inc = alpha                    (near normal)
            blended:         theta_inc = (1-W)(pi/2 - alpha) + W*alpha
        """
        W = self._surface_orientation_window()[:, None]          # (n_z, 1)
        alpha = self.theta_ion[None, :] + self.dtheta[:, None]   # (n_z, n_a)
        theta_inc = (1.0 - W) * (np.pi / 2.0 - alpha) + W * alpha
        return np.clip(theta_inc, 0.0, np.pi / 2.0)

    # ------------------------------------------------------- sputter rate
    def _sputter_rate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Angle-resolved sputter rate:
            R_S(z) = Omega * Gamma_i * < Y(E, theta_inc) cos(theta_inc) >_IEADF
        The cos factor projects the (vertical) ion flux onto the local surface.
        Returns (R_S, Y_bar, theta_inc_effective).
        """
        theta_inc = self._incidence_angles()                     # (n_z, n_a)
        Y = self.yamamura_yield(self.E_i_eV, theta_inc)          # (n_z, n_a)
        proj = np.cos(theta_inc)

        # IEADF-weighted integrals over the ion polar angle.
        w = self.p_ion[None, :]
        denom = np.trapezoid(self.p_ion, self.theta_ion)
        Ycos = np.trapezoid(Y * proj * w, self.theta_ion, axis=1) / denom
        Y_bar = np.trapezoid(Y * w, self.theta_ion, axis=1) / denom
        theta_eff = np.trapezoid(theta_inc * w, self.theta_ion, axis=1) / denom

        R_S = self.p.omega_m3 * self.Gamma_i * Ycos
        return R_S, Y_bar, theta_eff

    # ---------------------------------------------------- deposition rate
    def _deposition_rate(self) -> np.ndarray:
        """R_D(z) = Omega * s_c(z) * Gamma_g(z)  [m/s]."""
        s_c = np.asarray(self.sf.s_c, dtype=float)
        return self.p.omega_m3 * s_c * self.Gamma_g

    # ------------------------------------------------------ redeposition
    def _redeposition(self, R_S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Redistribute sputtered material to the opposing wall via the parallel-
        plate differential view factor
            K(z, z') = (1/2) w^2 / (w^2 + (z - z')^2)^{3/2}
        so that R_redep(z') = (1 - s_reemit) * \int K(z,z') R_S(z) dz.
        Also returns the coarse 5-zone view-factor matrix F_{k->k'} (blueprint).
        """
        z = self.z
        w = float(np.mean(self.width))
        dz = np.gradient(z)

        # Node-resolved kernel (n_z, n_z): row z', col z.
        ZP, Z = np.meshgrid(z, z, indexing="ij")
        K = 0.5 * w ** 2 / (w ** 2 + (Z - ZP) ** 2) ** 1.5
        R_redep = (1.0 - self.p.s_reemit) * (K * dz[None, :]) @ R_S

        # Coarse 5-zone view-factor matrix (integrate kernel over z-bands).
        nz_zones = self.p.n_zones
        edges = np.linspace(0.0, self.H, nz_zones + 1)
        idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, nz_zones - 1)
        F = np.zeros((nz_zones, nz_zones))
        for k in range(nz_zones):
            src = idx == k
            if not np.any(src):
                continue
            contrib = (K[:, src] * dz[None, src]).sum(axis=1)    # arrival per z'
            for kp in range(nz_zones):
                F[k, kp] = contrib[idx == kp].mean() if np.any(idx == kp) else 0.0
        return R_redep, F

    # ------------------------------------------------- surface diffusion
    def _surface_diffusion_div(self) -> np.ndarray:
        """div(J_s) = d/dz[ -D_s dn_s/dz ]  [m^-2 s^-1] from L2 fields."""
        D_s = np.asarray(self.sf.D_s, dtype=float)
        n_s = np.asarray(self.sf.n_s, dtype=float)
        J_s = -D_s * np.gradient(n_s, self.z)
        return np.gradient(J_s, self.z)

    # ----------------------------------------------------------------- solve
    def solve(self) -> RateState:
        """Assemble all rates into V_n(z)."""
        R_D = self._deposition_rate()
        R_S, Y_bar, theta_eff = self._sputter_rate()
        R_redep, F_zone = self._redeposition(R_S)
        div_Js = self._surface_diffusion_div()

        # Net normal velocity: deposition - sputter + redeposition - O*div(J_s).
        V_n = R_D - R_S + R_redep - self.p.omega_m3 * div_Js

        # Local D/S ratio (guard against divide-by-zero on near-grazing walls
        # and cap where R_S -> 0, i.e. sub-threshold ions / pure deposition).
        R_DS = np.minimum(R_D / np.clip(R_S, 1e-300, None), 1e9)

        # Diagnostic: track how ion energy translates into sputter leverage.
        # The sputter/deposition ratio S/D is what determines bias responsiveness;
        # it must reach the ~0.1-0.3 regime for gap-fill to "see" the bias slider.
        if log.isEnabledFor(logging.DEBUG):
            rs_max, rd_max = float(np.max(R_S)), float(np.max(R_D))
            sd = rs_max / rd_max if rd_max > 0 else 0.0
            log.debug(
                "E_ion=%.1f eV | f_precursor=%.3f | R_S_max=%.3e R_D_max=%.3e "
                "| S/D=%.3f | Y_bar_max=%.3e",
                self.E_i_eV, float(getattr(self.plasma, "f_precursor", 1.0)),
                rs_max, rd_max, sd, float(np.max(Y_bar)),
            )

        return RateState(
            z=self.z, R_D=R_D, R_S=R_S, R_redep=R_redep, div_Js=div_Js,
            V_n=V_n, R_DS=R_DS, Y_bar=Y_bar, theta_inc_eff=theta_eff,
            zone_view_factor=F_zone,
        )


# =============================================================================
# 5. DEMONSTRATION / SELF-TEST  (mocks L0 + L1 + L2)
# =============================================================================
def _make_mock_states():
    """Prefer real L0/L1/L2; otherwise duck-typed stand-ins."""
    try:
        from fpm_L0_global_plasma import GlobalPlasmaSolver, ReactorInputs
        from fpm_L1_feature_transport import (FeatureTransportSolver,
                                              FeatureGeometry, TransportParams)
        from fpm_L2_surface_kinetics import SurfaceKineticsSolver, SurfaceParams
        inp = ReactorInputs(power_w=2500.0, pressure_mtorr=5.0,
                            flows_sccm={"Ar": 100.0, "O2": 50.0, "SiH4": 30.0},
                            rf_bias_v=80.0)
        plasma = GlobalPlasmaSolver(inp).solve()
        geom = FeatureGeometry(aspect_ratio=10.0, w0_m=100e-9, n_z=200)
        transport = FeatureTransportSolver(plasma, geom, TransportParams(0.10)).solve()
        surface = SurfaceKineticsSolver(plasma, transport, geom, SurfaceParams()).solve()
        return plasma, transport, surface
    except Exception as exc:           # pragma: no cover
        from types import SimpleNamespace
        print(f"[demo] using mock states ({exc})")
        z = np.linspace(0.0, 1.0e-6, 200)
        r = 91.57 / 0.05
        th = np.linspace(0, np.pi / 2, 91)
        ang = r * np.sin(th) * np.cos(th) * np.exp(-r * np.sin(th) ** 2)
        ieadf = SimpleNamespace(theta_axis=th,
                                angle_pdf=lambda t, e: r * np.sin(t) * np.cos(t)
                                * np.exp(-r * np.sin(t) ** 2),
                                mean_angle_deg=lambda: 1.22)
        plasma = SimpleNamespace(T_e=2.529, n_e=2.031e17, m_ion_eff_amu=32.44,
                                 mean_ion_energy_ev=91.57, ieadf=ieadf)
        n_g = 9.656e19 * (0.113 + 0.887 * np.exp(-z / (z[-1] / 2.7)))
        transport = SimpleNamespace(z=z, n_g=n_g, width=np.full_like(z, 100e-9),
                                    dtheta_charge=np.full_like(z, np.radians(2.9)))
        T = np.full_like(z, 605.5)
        surface = SimpleNamespace(s_c=np.full_like(z, 0.133),
                                  D_s=np.full_like(z, 1.55e-11),
                                  n_s=3.0e16 * (n_g / n_g[0]),
                                  Gamma_i=5.57e20)
        return plasma, transport, surface


def _demo() -> None:
    """Assemble rates and validate V_n(z) down a 10:1, 100 nm trench."""
    plasma, transport, surface = _make_mock_states()
    state = RateAssemblySolver(plasma, transport, surface).solve()

    print("=" * 72)
    print(" [L3] RATE ASSEMBLY  -  10:1 trench (w0 = 100 nm)")
    print("=" * 72)
    print(f" Ion flux Gamma_i : {state_gi(state, surface):.3e} m^-2 s^-1 | "
          f"E_i = {plasma.mean_ion_energy_ev:.1f} eV")
    print("-" * 72)
    for k, v in state.summary().items():
        print(f" {k:<24}: {v:.5g}")
    print("-" * 72)
    idx = [0, len(state.z) // 2, -1]
    labels = ["mouth", "mid", "bottom"]
    print(f" {'station':<8}{'z[nm]':>8}{'th_inc[deg]':>12}{'Y_bar':>11}"
          f"{'R_D':>11}{'R_S':>11}{'R_redep':>11}{'V_n[nm/min]':>13}")
    for lbl, k in zip(labels, idx):
        print(f" {lbl:<8}{state.z[k]*1e9:>8.1f}"
              f"{np.degrees(state.theta_inc_eff[k]):>12.2f}{state.Y_bar[k]:>11.3e}"
              f"{state.R_D[k]*1e9*60:>11.3f}{state.R_S[k]*1e9*60:>11.4f}"
              f"{state.R_redep[k]*1e9*60:>11.4f}{state.V_n[k]*1e9*60:>13.3f}")
    print("-" * 72)
    print(" 5-zone redeposition view-factor matrix F_{k->k'} (top->bottom):")
    with np.printoptions(precision=3, suppress=True):
        print(state.zone_view_factor)
    print("-" * 72)
    gapfill = "deposition-dominated (fill)" if state.V_n[-1] > 0 else "etch-dominated"
    print(f" Bottom regime          : V_n={state.V_n[-1]*1e9*60:.2f} nm/min -> {gapfill}")
    print(f" D/S ratio top->bottom  : {state.R_DS[0]:.1f} -> {state.R_DS[-1]:.1f}")
    print("=" * 72)


def state_gi(state, surface) -> float:
    """Helper: report the ion flux used (from surface or recomputed)."""
    return float(getattr(surface, "Gamma_i", np.nan))


if __name__ == "__main__":
    _demo()
