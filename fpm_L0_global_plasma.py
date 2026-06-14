"""
fpm_L0_global_plasma.py
=======================

[L0] Global Plasma Solver for the HDP-CVD Feature Profile Model (FPM).

This module computes the *wafer-uniform* plasma state for an inductively-coupled
High-Density Plasma CVD reactor running an Argon / Silane (SiH4) / Oxygen (O2)
chemistry. It closes the classical ICP "global model" (volume-averaged particle
and power balance) to deliver the four L0 state quantities required by the
downstream FPM layers (L1 transport, L4 level-set, etc.):

    T_e   : electron temperature                          [eV]
    n_e   : plasma (electron) density                     [m^-3]
    x     : EEDF shape parameter (1 = Maxwellian,
            2 = Druyvesteyn)                              [-]
    V_sh  : total sheath potential drop at the wafer       [V]
    g(e,θ): ion energy-angle distribution function (IEADF) [callable + grid]

Physics chain implemented
-------------------------
    source power, pressure, flows
        -> EEDF shape (collisionality closure)
        -> rate coefficients  K_iz(T_e), K_ex(T_e)   (EEDF moments of cross-sections)
        -> particle balance   n_g <K_iz> = u_B / d_eff           => T_e
        -> power balance      P_abs = e n_e u_B A_eff E_T(T_e)    => n_e
        -> sheath solve       V_p, Child-Langmuir s, RF bias      => V_sh
        -> IEADF generation   bimodal RF IEDF  x  collisionless IADF

References (standard texts; values quoted inline with ranges)
    - Lieberman & Lichtenberg, "Principles of Plasma Discharges and Materials
      Processing", 2nd ed. (global model, Bohm flux, edge-to-center factors).
    - Lotz, Z. Phys. 206 (1967) 205  (electron-impact ionization cross-section).
    - Druyvesteyn distribution for elastic-collision-dominated EEDFs.

Author role: Principal Scientific Computing Engineer
Layer:       L0 (this file implements ONLY L0; L1-L6 are out of scope here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np
from scipy import constants as const
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma as gamma_fn

# =============================================================================
# 1. PHYSICAL CONSTANTS  (true SI values from CODATA via scipy.constants)
# =============================================================================
E_CHARGE: float = const.e               # elementary charge            [C]
M_E: float = const.m_e                  # electron mass                [kg]
K_B: float = const.k                    # Boltzmann constant           [J/K]
EPS0: float = const.epsilon_0           # vacuum permittivity          [F/m]
AMU: float = const.physical_constants["atomic mass constant"][0]  # [kg]
EV_TO_J: float = const.e                # 1 eV in Joules               [J/eV]
TORR_TO_PA: float = 133.322             # 1 Torr in Pascal             [Pa/Torr]
MTORR_TO_PA: float = TORR_TO_PA * 1e-3  # 1 mTorr in Pascal            [Pa/mTorr]

# Lotz ionization-cross-section prefactor.
#   sigma_iz(E) = a_LOTZ * q * ln(E/E_iz) / (E * E_iz),  E,E_iz in eV  -> cm^2
#   a_LOTZ ~ 4.5e-14 cm^2 * eV^2 is the canonical Lotz constant.
A_LOTZ_CM2_EV2: float = 4.5e-14         # [cm^2 * eV^2]
A_LOTZ_M2_EV2: float = A_LOTZ_CM2_EV2 * 1e-4  # convert cm^2 -> m^2


# =============================================================================
# 2. SPECIES DATABASE
# =============================================================================
@dataclass(frozen=True)
class Species:
    """
    Electron-collision and transport parameters for one gas species.

    Notes on values (typical literature ranges given in comments). These are
    *defaults*; calibration against fab metrology should refine them.
    """
    name: str
    mass_amu: float          # neutral/ion mass                        [amu]
    E_iz: float              # first ionization energy                 [eV]
    q_iz: float              # equivalent outer-shell electrons (Lotz) [-]
    E_ex: float              # representative inelastic (excitation/
                             # dissociation) threshold                 [eV]
    q_ex: float              # equivalent electrons for that channel   [-]
    sigma_el: float          # representative elastic / CX cross-sec   [m^2]

    @property
    def mass_kg(self) -> float:
        return self.mass_amu * AMU


# Default HDP-CVD chemistry set.
#   Ar  : E_iz = 15.76 eV,  q ~ 6 (3p^6).            sigma_el ~ 1e-19 m^2.
#   O2  : E_iz = 12.06 eV (12.0-12.2). Dissociation/excitation ~6 eV.
#   SiH4: E_iz ~ 11.0 eV (10.5-11.5). Dissociation ~8 eV. Molecular, q approx.
SPECIES_DB: Dict[str, Species] = {
    "Ar":   Species("Ar",   39.948, E_iz=15.76, q_iz=6.0, E_ex=11.55, q_ex=6.0,
                    sigma_el=1.0e-19),
    "O2":   Species("O2",   31.998, E_iz=12.06, q_iz=4.0, E_ex=6.00,  q_ex=4.0,
                    sigma_el=5.0e-19),
    "SiH4": Species("SiH4", 32.117, E_iz=11.00, q_iz=8.0, E_ex=8.00,  q_ex=8.0,
                    sigma_el=8.0e-19),
}


# =============================================================================
# 3. INPUT / OUTPUT CONTAINERS
# =============================================================================
@dataclass
class ReactorInputs:
    """
    Base reactor operating point. Flows are in sccm; only their *ratios*
    set the gas composition (partial pressures), while absolute pressure is
    set explicitly (throttled chamber).
    """
    power_w: float                       # absorbed source (ICP) power   [W]
    pressure_mtorr: float                # chamber pressure              [mTorr]
    flows_sccm: Dict[str, float]         # {species: flow}               [sccm]
    radius_m: float = 0.15               # chamber radius                [m]
    length_m: float = 0.10               # chamber height (gap)          [m]
    gas_temp_k: float = 500.0            # neutral gas temperature       [K]
    ion_temp_ev: float = 0.05            # transverse ion temperature    [eV]
    rf_bias_v: float = 0.0               # peak RF bias amplitude        [V]
    rf_freq_hz: float = 13.56e6          # bias frequency                [Hz]

    def mole_fractions(self) -> Dict[str, float]:
        """Normalize flows to composition fractions f_s (sum = 1)."""
        total = sum(self.flows_sccm.values())
        if total <= 0:
            raise ValueError("Total gas flow must be positive.")
        return {s: q / total for s, q in self.flows_sccm.items()}


@dataclass
class PlasmaState:
    """Solved L0 plasma state handed to downstream FPM layers."""
    T_e: float                           # electron temperature            [eV]
    n_e: float                           # electron/plasma density         [m^-3]
    x_eedf: float                        # EEDF shape parameter            [-]
    V_sheath: float                      # total sheath drop at wafer      [V]
    V_plasma: float                      # plasma (presheath) potential    [V]
    n_g: float                           # neutral density                 [m^-3]
    m_ion_eff_amu: float                 # flux-weighted effective ion mass [amu]
    sheath_thickness_m: float            # Child-Langmuir sheath width     [m]
    debye_length_m: float                # electron Debye length           [m]
    mean_ion_energy_ev: float            # <eps_i> delivered to wafer       [eV]
    ion_energy_spread_ev: float          # IEDF peak-to-peak width dE       [eV]
    ieadf: "IEADF" = field(repr=False)   # ion energy-angle distribution
    f_precursor: float = 1.0             # film-forming (Si-bearing) mole fraction [-]
                                         #   fraction of the neutral flux that
                                         #   actually deposits film; the inert
                                         #   sputter gas (Ar) and excess oxidizer
                                         #   do NOT contribute to R_D.

    def as_dict(self) -> Dict[str, float]:
        """Flat dictionary of scalar outputs (IEADF excluded)."""
        return {
            "T_e_eV": self.T_e,
            "n_e_m3": self.n_e,
            "x_eedf": self.x_eedf,
            "V_sheath_V": self.V_sheath,
            "V_plasma_V": self.V_plasma,
            "n_g_m3": self.n_g,
            "m_ion_eff_amu": self.m_ion_eff_amu,
            "sheath_thickness_m": self.sheath_thickness_m,
            "debye_length_m": self.debye_length_m,
            "mean_ion_energy_eV": self.mean_ion_energy_ev,
            "ion_energy_spread_eV": self.ion_energy_spread_ev,
            "f_precursor": self.f_precursor,
        }


# =============================================================================
# 4. EEDF  (generalized Maxwellian / Druyvesteyn)
# =============================================================================
class EEDF:
    r"""
    Generalized electron energy distribution function:

        g_e(eps) = C * sqrt(eps) * exp[ -(b * eps)^x ]

    normalized so that  \int_0^\infty g_e d(eps) = 1  and the mean energy is
    <eps> = (3/2) T_e. Using the substitution u = (b eps)^x, the closure
    constants follow analytically from Gamma functions:

        b = (2 / (3 T_e)) * Gamma(5/2x) / Gamma(3/2x)
        C = x * b^{3/2} / Gamma(3/2x)

    x = 1 recovers the Maxwellian; x = 2 the Druyvesteyn (elastic-collision
    dominated, depleted high-energy tail).
    """

    def __init__(self, T_e: float, x: float) -> None:
        if T_e <= 0:
            raise ValueError("T_e must be positive.")
        if not (1.0 <= x <= 2.0):
            raise ValueError("EEDF shape x must lie in [1, 2].")
        self.T_e = float(T_e)
        self.x = float(x)
        g3 = gamma_fn(3.0 / (2.0 * x))
        g5 = gamma_fn(5.0 / (2.0 * x))
        self.b = (2.0 / (3.0 * self.T_e)) * (g5 / g3)          # [1/eV]
        self.C = self.x * self.b ** 1.5 / g3                   # [eV^-3/2]

    def pdf(self, eps: "np.ndarray | float") -> "np.ndarray | float":
        """EEDF value g_e(eps); eps in eV."""
        eps = np.asarray(eps, dtype=float)
        return self.C * np.sqrt(np.clip(eps, 0.0, None)) * np.exp(
            -(self.b * eps) ** self.x
        )

    def rate_coefficient(
        self, cross_section: Callable[[float], float], E_threshold: float
    ) -> float:
        r"""
        Maxwellian/Druyvesteyn-averaged rate coefficient:

            K = \int_{E_th}^\infty sigma(eps) * v(eps) * g_e(eps) d(eps)

        with v(eps) = sqrt(2 eps_J / m_e). Returns K in [m^3 / s].
        """
        def integrand(eps_ev: float) -> float:
            v = np.sqrt(2.0 * eps_ev * EV_TO_J / M_E)          # speed [m/s]
            return cross_section(eps_ev) * v * float(self.pdf(eps_ev))

        # Integrate to ~30 thermal energies; tail is exponentially negligible.
        upper = E_threshold + 30.0 * self.T_e
        val, _ = quad(integrand, E_threshold, upper, limit=200)
        return val


def lotz_cross_section(species: Species, channel: str = "iz") -> Callable[[float], float]:
    r"""
    Build a Lotz-form electron-impact cross-section sigma(eps) [m^2] for the
    ionization ("iz") or representative inelastic ("ex") channel:

        sigma(eps) = a_LOTZ * q * ln(eps / E_th) / (eps * E_th),  eps > E_th
                   = 0,                                            eps <= E_th
    """
    if channel == "iz":
        E_th, q = species.E_iz, species.q_iz
    elif channel == "ex":
        E_th, q = species.E_ex, species.q_ex
    else:
        raise ValueError("channel must be 'iz' or 'ex'.")

    def sigma(eps_ev: float) -> float:
        if eps_ev <= E_th:
            return 0.0
        return A_LOTZ_M2_EV2 * q * np.log(eps_ev / E_th) / (eps_ev * E_th)

    return sigma


# =============================================================================
# 5. IEADF  (Ion Energy-Angle Distribution Function)
# =============================================================================
class IEADF:
    r"""
    Ion Energy-Angle Distribution g(eps, theta) delivered to the wafer.

    Energy axis  -- bimodal RF sheath IEDF. For a sinusoidally modulated
    sheath the ion energy oscillates as eps = <eps> + (dE/2) sin(phi), giving

        f_E(eps) = (2 / (pi * dE)) / sqrt(1 - ((eps - <eps>)/(dE/2))^2)

    i.e. the classic saddle structure with two peaks separated by dE.

    Angle axis   -- collisionless-sheath IADF (derived in the FPM blueprint):

        g(theta) ∝ (eps/T_i) sin(theta) cos(theta) exp[-(eps/T_i) sin^2(theta)]

    sharply forward-peaked with half-width ~ sqrt(T_i / 2 eps). It is normalized
    here so that the angular marginal integrates to unity over [0, pi/2].

    The joint distribution is the product f_E(eps) * g(theta | eps).
    """

    def __init__(
        self,
        mean_energy_ev: float,
        spread_ev: float,
        ion_temp_ev: float,
        n_eps: int = 200,
        n_theta: int = 180,
    ) -> None:
        self.mean_energy = float(mean_energy_ev)
        # Guard against a degenerate (zero-width) IEDF when there is no RF bias.
        self.spread = float(max(spread_ev, 1e-6))
        self.T_i = float(ion_temp_ev)

        # Discretized axes for grid/sampling consumers.
        emin = max(self.mean_energy - self.spread / 2.0, 0.0)
        emax = self.mean_energy + self.spread / 2.0
        self.eps_axis = np.linspace(emin, emax, n_eps)         # [eV]
        self.theta_axis = np.linspace(0.0, np.pi / 2.0, n_theta)  # [rad]

    # ---- marginal distributions -------------------------------------------
    def energy_pdf(self, eps: "np.ndarray | float") -> np.ndarray:
        """Bimodal RF-sheath IEDF f_E(eps) (normalized over the energy axis)."""
        eps = np.asarray(eps, dtype=float)
        half = self.spread / 2.0
        z = (eps - self.mean_energy) / half
        with np.errstate(divide="ignore", invalid="ignore"):
            f = (2.0 / (np.pi * self.spread)) / np.sqrt(1.0 - z ** 2)
        return np.where(np.abs(z) < 1.0, f, 0.0)

    def angle_pdf(self, theta: "np.ndarray | float", eps: float) -> np.ndarray:
        """Collisionless IADF g(theta | eps) at fixed ion energy eps."""
        theta = np.asarray(theta, dtype=float)
        r = eps / self.T_i
        # The raw form integrates to (1 - e^-r)/2 over [0, pi/2]; divide by that
        # so g(theta | eps) is a true PDF (integrates to unity).
        norm = (1.0 - np.exp(-r)) / 2.0
        raw = r * np.sin(theta) * np.cos(theta) * np.exp(-r * np.sin(theta) ** 2)
        return raw / norm

    # ---- joint distribution -----------------------------------------------
    def joint_pdf(self, eps: float, theta: float) -> float:
        """g(eps, theta) = f_E(eps) * g(theta | eps)."""
        return float(self.energy_pdf(eps)) * float(self.angle_pdf(theta, eps))

    def grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (EPS, THETA, G) meshgrid arrays for the full joint IEADF,
        suitable for plotting or as a lookup table for L1.
        """
        EPS, THETA = np.meshgrid(self.eps_axis, self.theta_axis, indexing="ij")
        fE = self.energy_pdf(self.eps_axis)[:, None]           # (n_eps, 1)
        r = EPS / self.T_i
        norm = (1.0 - np.exp(-r)) / 2.0                        # per-energy IADF norm
        gA = (r * np.sin(THETA) * np.cos(THETA)
              * np.exp(-r * np.sin(THETA) ** 2)) / norm
        return EPS, THETA, fE * gA

    def mean_angle_deg(self) -> float:
        """Flux-weighted mean impact angle (diagnostic), in degrees."""
        g = self.angle_pdf(self.theta_axis, self.mean_energy)
        norm = np.trapezoid(g, self.theta_axis)
        mean = np.trapezoid(self.theta_axis * g, self.theta_axis) / norm
        return float(np.degrees(mean))


# =============================================================================
# 6. GLOBAL PLASMA SOLVER
# =============================================================================
class GlobalPlasmaSolver:
    """
    Solve the ICP global model for an HDP-CVD operating point.

    Usage
    -----
    >>> inp = ReactorInputs(power_w=2000, pressure_mtorr=5.0,
    ...                     flows_sccm={"Ar": 100, "O2": 40, "SiH4": 30},
    ...                     rf_bias_v=80.0)
    >>> state = GlobalPlasmaSolver(inp).solve()
    >>> state.T_e, state.n_e, state.V_sheath
    """

    def __init__(self, inputs: ReactorInputs) -> None:
        self.inp = inputs
        self.frac = inputs.mole_fractions()
        self.species = {s: SPECIES_DB[s] for s in self.frac}

        # Derived geometric quantities (cylindrical chamber).
        self.R = inputs.radius_m
        self.L = inputs.length_m
        self.volume = np.pi * self.R ** 2 * self.L             # [m^3]

        # Neutral density from ideal gas law: n_g = p / (k_B T_g).
        p_pa = inputs.pressure_mtorr * MTORR_TO_PA
        self.n_g = p_pa / (K_B * inputs.gas_temp_k)            # [m^-3]

        # Composition-averaged neutral/ion mass (mole-fraction weighted seed).
        self.m_ion_amu = sum(self.frac[s] * self.species[s].mass_amu
                             for s in self.frac)
        self.m_ion_kg = self.m_ion_amu * AMU

        # EEDF shape closure (see _estimate_eedf_shape).
        self.x_eedf = self._estimate_eedf_shape()

        # Pre-build cross-section callables (avoid rebuilding inside solver loop).
        self._sigma_iz = {s: lotz_cross_section(self.species[s], "iz")
                          for s in self.frac}
        self._sigma_ex = {s: lotz_cross_section(self.species[s], "ex")
                          for s in self.frac}

    # ------------------------------------------------------------------ EEDF
    def _estimate_eedf_shape(self) -> float:
        """
        Heuristic closure for the EEDF shape parameter x in [1, 2].

        Physics: as pressure rises, elastic collisions dominate and deplete
        the inelastic tail, driving the distribution from Maxwellian (x=1,
        low-pressure / high-T_e) toward Druyvesteyn (x=2, collisional).
        A self-consistent two-term Boltzmann solve (BOLSIG+-style) is the
        rigorous route and is the documented L0 upgrade path; here we use a
        saturating map calibrated to typical HDP behavior.
        """
        p = self.inp.pressure_mtorr
        p_char = 10.0  # characteristic pressure [mTorr] for tail depletion
        return float(np.clip(1.0 + (1.0 - np.exp(-p / p_char)), 1.0, 2.0))

    # ---------------------------------------------------- edge-to-center loss
    def _edge_factors(self) -> Tuple[float, float]:
        """
        Godyak edge-to-center density ratios h_l (axial) and h_R (radial),
        which fold the sheath/presheath density drop into an effective wall
        area. lambda_i is the ion mean free path.
        """
        sigma_i = sum(self.frac[s] * self.species[s].sigma_el for s in self.frac)
        lambda_i = 1.0 / (self.n_g * sigma_i)                  # [m]
        h_l = 0.86 / np.sqrt(3.0 + self.L / (2.0 * lambda_i))
        h_R = 0.80 / np.sqrt(4.0 + self.R / lambda_i)
        return h_l, h_R

    def _effective_area_and_length(self) -> Tuple[float, float]:
        """A_eff (loss-weighted wall area) and d_eff = V / A_eff."""
        h_l, h_R = self._edge_factors()
        A_eff = (2.0 * np.pi * self.R ** 2 * h_l
                 + 2.0 * np.pi * self.R * self.L * h_R)
        d_eff = self.volume / A_eff
        return A_eff, d_eff

    # --------------------------------------------------------- rate moments
    def _mix_rate(self, T_e: float, channel: str) -> float:
        """
        Composition-weighted rate coefficient sum_s f_s K_s(T_e) [m^3/s]
        for 'iz' (ionization) or 'ex' (inelastic) channels.
        """
        eedf = EEDF(T_e, self.x_eedf)
        sig = self._sigma_iz if channel == "iz" else self._sigma_ex
        thr = "E_iz" if channel == "iz" else "E_ex"
        total = 0.0
        for s in self.frac:
            E_th = getattr(self.species[s], thr)
            total += self.frac[s] * eedf.rate_coefficient(sig[s], E_th)
        return total

    def bohm_velocity(self, T_e: float) -> float:
        """Bohm velocity u_B = sqrt(e T_e / m_ion) [m/s] (T_e in eV)."""
        return np.sqrt(E_CHARGE * T_e / self.m_ion_kg)

    # --------------------------------------------------- balance residuals
    def _particle_balance_residual(self, T_e: float, d_eff: float) -> float:
        r"""
        Particle balance: ionization frequency = ambipolar loss frequency.

            n_g * sum_s f_s K_iz,s(T_e)  =  u_B(T_e) / d_eff

        Root in T_e fixes the electron temperature (independent of n_e).
        """
        nu_iz = self.n_g * self._mix_rate(T_e, "iz")
        nu_loss = self.bohm_velocity(T_e) / d_eff
        return nu_iz - nu_loss

    def _collisional_energy_loss(self, T_e: float) -> float:
        r"""
        Energy lost per electron-ion pair created (E_c), assembled from first
        principles rather than a species-specific fit:

            E_c = [ sum_s f_s (E_iz,s K_iz,s + E_ex,s K_ex,s)
                    + (3 m_e / M) T_e * sum_s f_s K_iz,s ]  /  sum_s f_s K_iz,s
        """
        eedf = EEDF(T_e, self.x_eedf)
        num = 0.0
        den = 0.0
        for s in self.frac:
            sp = self.species[s]
            K_iz = eedf.rate_coefficient(self._sigma_iz[s], sp.E_iz)
            K_ex = eedf.rate_coefficient(self._sigma_ex[s], sp.E_ex)
            num += self.frac[s] * (sp.E_iz * K_iz + sp.E_ex * K_ex)
            den += self.frac[s] * K_iz
        # Elastic recoil loss proxy: (3 m_e / M) * T_e per ionization event.
        elastic = (3.0 * M_E / self.m_ion_kg) * T_e
        return num / den + elastic

    def _total_energy_cost(self, T_e: float, V_sh: float) -> float:
        """
        Total energy lost per ion leaving the system:
            E_T = E_c (collisional) + E_e (electron) + E_i (ion at wall)
        with E_e = 2 T_e and E_i = V_sh + T_e/2.  Units: [eV].
        """
        E_c = self._collisional_energy_loss(T_e)
        E_e = 2.0 * T_e
        E_i = V_sh + 0.5 * T_e
        return E_c + E_e + E_i

    # ------------------------------------------------------------- sheath
    def plasma_potential(self, T_e: float) -> float:
        """
        Presheath/plasma potential relative to a floating wall:
            V_p = T_e * ln( sqrt( M_ion / (2 pi m_e) ) )   (~5 T_e for Ar)
        """
        return T_e * np.log(np.sqrt(self.m_ion_kg / (2.0 * np.pi * M_E)))

    def debye_length(self, T_e: float, n_e: float) -> float:
        """Electron Debye length lambda_De = sqrt(eps0 T_e / (e n_e)) [m]."""
        return np.sqrt(EPS0 * T_e / (E_CHARGE * n_e))

    def child_langmuir_sheath(self, T_e: float, n_e: float, V_sh: float) -> float:
        """
        High-voltage (matrix) sheath thickness:
            s = (sqrt(2)/3) lambda_De (2 V_sh / T_e)^{3/4}
        """
        lam_de = self.debye_length(T_e, n_e)
        return (np.sqrt(2.0) / 3.0) * lam_de * (2.0 * V_sh / T_e) ** 0.75

    def ion_transit_time(self, T_e: float, n_e: float, V_sh: float) -> float:
        """Ion transit time across the sheath: tau = 3 s sqrt(M / (2 e V_sh))."""
        s = self.child_langmuir_sheath(T_e, n_e, V_sh)
        return 3.0 * s * np.sqrt(self.m_ion_kg / (2.0 * E_CHARGE * V_sh))

    def ieadf_energy_spread(self, T_e: float, n_e: float, V_sh: float) -> float:
        """
        RF IEDF peak-to-peak energy width:
            dE = 2 V_rf / (omega * tau_ion)     [eV]
        Vanishes for zero RF bias (delta-like IEDF -> mono-energetic ions).
        """
        if self.inp.rf_bias_v <= 0.0:
            return 0.0
        omega = 2.0 * np.pi * self.inp.rf_freq_hz
        tau = self.ion_transit_time(T_e, n_e, V_sh)
        return 2.0 * self.inp.rf_bias_v / (omega * tau)

    # ----------------------------------------------------- effective ion mass
    def _flux_weighted_ion_mass(self, T_e: float) -> float:
        """
        Refine effective ion mass by weighting species masses by their
        *ionization-rate* contribution f_s K_iz,s (the ions actually made),
        rather than by neutral mole fraction alone.
        """
        eedf = EEDF(T_e, self.x_eedf)
        num = 0.0
        den = 0.0
        for s in self.frac:
            K_iz = eedf.rate_coefficient(self._sigma_iz[s], self.species[s].E_iz)
            w = self.frac[s] * K_iz
            num += w * self.species[s].mass_amu
            den += w
        return num / den if den > 0 else self.m_ion_amu

    # ----------------------------------------------------------------- solve
    def solve(self, T_e_bracket: Tuple[float, float] = (0.5, 12.0)) -> PlasmaState:
        """
        Execute the full L0 solve and return a PlasmaState.

        Steps
        -----
        1. Solve particle balance for T_e (Brent's method, robust bracketing).
        2. Sheath potential V_p from T_e; add RF bias amplitude -> V_sh.
        3. Solve power balance algebraically for n_e.
        4. Build sheath geometry (Debye length, Child-Langmuir thickness).
        5. Generate the IEADF g(eps, theta).
        """
        A_eff, d_eff = self._effective_area_and_length()

        # --- 1. Electron temperature from particle balance ------------------
        f_lo = self._particle_balance_residual(T_e_bracket[0], d_eff)
        f_hi = self._particle_balance_residual(T_e_bracket[1], d_eff)
        if f_lo * f_hi > 0:
            raise RuntimeError(
                "Particle-balance root not bracketed in "
                f"{T_e_bracket} eV (residuals {f_lo:.3e}, {f_hi:.3e}). "
                "Check pressure/geometry inputs."
            )
        T_e = brentq(self._particle_balance_residual, *T_e_bracket,
                     args=(d_eff,), xtol=1e-6, rtol=1e-8)

        # Refine effective ion mass with the converged T_e (mild coupling).
        self.m_ion_amu = self._flux_weighted_ion_mass(T_e)
        self.m_ion_kg = self.m_ion_amu * AMU

        # --- 2. Sheath potential -------------------------------------------
        V_p = self.plasma_potential(T_e)
        # Total sheath drop = plasma potential + applied peak RF bias.
        V_sh = V_p + self.inp.rf_bias_v

        # --- 3. Density from power balance ----------------------------------
        u_B = self.bohm_velocity(T_e)
        E_T_eV = self._total_energy_cost(T_e, V_sh)            # energy / ion [eV]
        # Bohm ion flux to the walls: Gamma = n_e * u_B * A_eff  [ions/s].
        # Power balance: P_abs = Gamma * (energy lost per ion in Joules)
        #              = n_e * u_B * A_eff * (E_T_eV * e).
        # EV_TO_J == e, so we convert E_T from eV to Joules exactly ONCE here.
        n_e = self.inp.power_w / (u_B * A_eff * E_T_eV * EV_TO_J)

        # --- 4. Sheath geometry --------------------------------------------
        lam_de = self.debye_length(T_e, n_e)
        s_sheath = self.child_langmuir_sheath(T_e, n_e, V_sh)

        # --- 5. IEADF ------------------------------------------------------
        mean_eps = V_sh                                        # <eps_i> = e V_sh [eV]
        dE = self.ieadf_energy_spread(T_e, n_e, V_sh)
        ieadf = IEADF(mean_energy_ev=mean_eps, spread_ev=dE,
                      ion_temp_ev=self.inp.ion_temp_ev)

        # Film-forming precursor fraction: only the Si-bearing feed (SiH4) builds
        # SiO2 film. The inert sputter gas (Ar) and excess oxidizer do not, so the
        # *depositing* neutral flux is this fraction of the total. Routing R_D
        # through this (instead of the total n_g) restores the physical
        # sputter/deposition balance and makes gap-fill respond to ion energy.
        f_prec = sum(f for s, f in self.frac.items() if "Si" in s)
        if not (0.0 < f_prec <= 1.0):
            f_prec = 1.0                                        # no Si feed -> no-op

        return PlasmaState(
            T_e=T_e, n_e=n_e, x_eedf=self.x_eedf, V_sheath=V_sh, V_plasma=V_p,
            n_g=self.n_g, m_ion_eff_amu=self.m_ion_amu,
            sheath_thickness_m=s_sheath, debye_length_m=lam_de,
            mean_ion_energy_ev=mean_eps, ion_energy_spread_ev=dE, ieadf=ieadf,
            f_precursor=f_prec,
        )


# =============================================================================
# 7. DEMONSTRATION / SELF-TEST
# =============================================================================
def _demo() -> None:
    """Run a representative HDP-CVD oxide-fill operating point and report L0."""
    inp = ReactorInputs(
        power_w=2500.0,
        pressure_mtorr=5.0,
        flows_sccm={"Ar": 100.0, "O2": 50.0, "SiH4": 30.0},
        radius_m=0.15,
        length_m=0.10,
        gas_temp_k=500.0,
        rf_bias_v=80.0,
        rf_freq_hz=13.56e6,
    )

    solver = GlobalPlasmaSolver(inp)
    state = solver.solve()

    print("=" * 64)
    print(" [L0] GLOBAL PLASMA SOLVE  -  HDP-CVD (Ar/O2/SiH4)")
    print("=" * 64)
    print(" Composition (mole frac) : "
          + ", ".join(f"{k}={v:.2f}" for k, v in inp.mole_fractions().items()))
    print(f" Neutral density  n_g    : {state.n_g:.3e}  m^-3")
    print("-" * 64)
    for k, v in state.as_dict().items():
        print(f" {k:<22}: {v:.4g}")
    print("-" * 64)

    # IEADF diagnostics: verify normalization and report mean impact angle.
    EPS, THETA, G = state.ieadf.grid()
    # Integrate joint distribution over the full (eps, theta) grid.
    integral = np.trapezoid(np.trapezoid(G, THETA[0], axis=1), EPS[:, 0])
    print(f" IEADF grid integral     : {integral:.4f}  (~1; <1 = arcsine-edge quadrature)")
    print(f" Mean ion impact angle   : {state.ieadf.mean_angle_deg():.2f} deg")
    print(f" Ionization fraction     : {state.n_e / state.n_g:.3e}")
    print("=" * 64)


if __name__ == "__main__":
    _demo()
