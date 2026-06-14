"""
dashboard.py  --  FabHeat-X :: Interactive Virtual Fab
======================================================

Streamlit front-end for the HDP-CVD Feature Profile Model. The user dials in a
recipe (RF power, RF bias, chamber pressure) from the sidebar, hits *Run
Simulation*, and the app drives the full physics pipeline:

    L0  GlobalPlasmaSolver        (bulk plasma from the recipe)
    L1  FeatureTransportSolver    (in-feature neutral/ion transport)
    L2  SurfaceKineticsSolver     (sticking / re-emission / sputter)
    L3  RateAssemblySolver        (net normal-velocity field V_n(z))
    L4  LevelSetSolver            (HJ-WENO5 interface evolution -> pinch-off)

The frozen interface is rendered as a trench cross-section (matplotlib logic
ported from ``visualizer.py``) and shown inline with ``st.pyplot``. Final
deposition time and the void verdict are surfaced as diagnostics.

This module is intentionally self-contained: it imports only the committed
``fpm_L0..L4`` solver classes, so the dashboard deploys with a single file.
"""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")                      # headless backend; safe under Streamlit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import streamlit as st

try:
    from scipy import ndimage as ndi
except Exception:                          # pragma: no cover
    ndi = None

# --- physics layers (committed alongside this file) ----------------------------
from fpm_L0_global_plasma import GlobalPlasmaSolver, ReactorInputs
from fpm_L1_feature_transport import (FeatureTransportSolver,
                                      FeatureGeometry, TransportParams)
from fpm_L2_surface_kinetics import SurfaceKineticsSolver, SurfaceParams
from fpm_L3_rate_assembly import RateAssemblySolver
from fpm_L4_level_set import LevelSetSolver, TrenchGeometry, LevelSetParams

# =============================================================================
# Fixed process geometry / non-slider recipe knobs
# =============================================================================
W0_M          = 100e-9                     # trench mouth width            [m]
ASPECT_RATIO  = 10.0                       # AR = H / w0                   [-]
WALL_STICKING = 0.10                       # L1 wall sticking coefficient  [-]
FLOWS_SCCM    = {"Ar": 100.0, "O2": 50.0, "SiH4": 30.0}
T_MAX_S       = 30.0                       # guard on simulated time       [s]

C_SOLID = "#3b4a5a"        # deposited film / substrate
C_OPEN  = "#dfe7ef"        # open gas / chamber
C_VOID  = "#d7263d"        # the sealed void (failure)
C_IFACE = "#00d2ff"        # interface outline


# =============================================================================
# Physics pipeline (L0 -> L1 -> L2 -> L3)
# =============================================================================
def build_velocity_field(power_w: float, pressure_mtorr: float, rf_bias_v: float):
    """Solve L0->L3 for the supplied recipe; return the L3 RateState (carries V_n)."""
    inp = ReactorInputs(power_w=power_w,
                        pressure_mtorr=pressure_mtorr,
                        flows_sccm=FLOWS_SCCM,
                        rf_bias_v=rf_bias_v)
    plasma    = GlobalPlasmaSolver(inp).solve()
    geom      = FeatureGeometry(aspect_ratio=ASPECT_RATIO, w0_m=W0_M, n_z=200)
    transport = FeatureTransportSolver(plasma, geom,
                                       TransportParams(WALL_STICKING)).solve()
    surface   = SurfaceKineticsSolver(plasma, transport, geom, SurfaceParams()).solve()
    return RateAssemblySolver(plasma, transport, surface).solve()


# =============================================================================
# L4 advance loop (run until pinch-off or T_MAX)
# =============================================================================
def run_until_pinch(rate):
    """Advance the level set until a void seals (or T_MAX). Return the solver."""
    solver = LevelSetSolver(
        geometry=TrenchGeometry(w0_m=W0_M, aspect_ratio=ASPECT_RATIO),
        params=LevelSetParams(scheme="weno5", cfl=0.4, reinit_every=8),
        rate_state=rate,
    )
    while solver.t < T_MAX_S and not bool(solver.detect_failures()["pinch_off"]):
        solver.advance(solver.t + 0.5, stop_on_pinch=True)
    return solver


# =============================================================================
# Render -> matplotlib Figure (ported from visualizer.render; returns the fig)
# =============================================================================
def build_trench_figure(solver, power_w: float, rf_bias_v: float):
    """Plot solid material, sealed void, and the interface contour; return Figure."""
    state = solver.state()
    phi   = state.phi                                  # (nz, nx) [m]
    x_nm  = solver.x * 1e9
    z_nm  = solver.z * 1e9

    # Identify sealed-void cells (open region NOT vented to the top row).
    void_mask = np.zeros_like(phi, dtype=float)
    openm = phi > 0
    if ndi is not None and openm.any():
        lbl, n = ndi.label(openm)
        vented = set(np.unique(lbl[0, :])) - {0}
        for comp in range(1, n + 1):
            if comp not in vented:
                void_mask[lbl == comp] = 1.0

    fig, ax = plt.subplots(figsize=(5.6, 7.4))

    # Solid vs open fill (phi <= 0 is solid).
    ax.contourf(x_nm, z_nm, phi, levels=[phi.min() - 1, 0.0, phi.max() + 1],
                colors=[C_SOLID, C_OPEN])

    # Highlight the sealed void pocket.
    if void_mask.any():
        ax.contourf(x_nm, z_nm, void_mask, levels=[0.5, 1.5],
                    colors=[C_VOID], alpha=0.85)

    # Interface outline (phi == 0).
    ax.contour(x_nm, z_nm, phi, levels=[0.0], colors=[C_IFACE], linewidths=1.6)

    # Orientation: mouth (z=0) at top, depth increasing downward.
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("x  [nm]")
    ax.set_ylabel("depth  z  [nm]")
    ax.set_title("HDP-CVD Gap-Fill — Trench Cross-Section\n"
                 f"{power_w:.0f} W / {rf_bias_v:.0f} V bias, "
                 f"AR {ASPECT_RATIO:.0f}:1 trench", fontsize=11, fontweight="bold")

    if state.pinch_off:
        ax.annotate(
            f"SEALED VOID\n{state.void_area_m2 * 1e18:,.0f} nm²  "
            f"({state.void_fraction * 100:.0f}% of trench)\n"
            f"seal @ z ≈ {state.seal_depth_m * 1e9:.0f} nm",
            xy=(0.0, state.seal_depth_m * 1e9 + 0.25 * solver.geom.depth_m * 1e9),
            xytext=(0.62, 0.30), textcoords="axes fraction",
            ha="left", va="center", fontsize=9, color=C_VOID, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_VOID, lw=1.4))

    legend = [
        Patch(facecolor=C_SOLID, label="Deposited solid (φ ≤ 0)"),
        Patch(facecolor=C_OPEN,  label="Open / gas (φ > 0)"),
        Patch(facecolor=C_VOID,  label="Sealed void"),
        Line2D([0], [0], color=C_IFACE, lw=1.6, label="Interface (φ = 0)"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


# =============================================================================
# Streamlit application
# =============================================================================
st.set_page_config(page_title="FabHeat-X | Virtual Fab")

st.title("FabHeat-X — Virtual Fab")
st.caption("Interactive HDP-CVD gap-fill simulator · physics pipeline L0 → L4")

# --- Sidebar: recipe controls -------------------------------------------------
st.sidebar.header("Process Recipe")
rf_power = st.sidebar.slider("RF Power", min_value=1000, max_value=5000,
                             value=3000, step=50, format="%d W")
rf_bias  = st.sidebar.slider("RF Bias", min_value=0, max_value=200,
                             value=80, step=5, format="%d V")
pressure = st.sidebar.slider("Pressure", min_value=1.0, max_value=10.0,
                             value=5.0, step=0.5, format="%.1f mTorr")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Fixed: AR {ASPECT_RATIO:.0f}:1 · mouth {W0_M * 1e9:.0f} nm · "
    f"flows Ar/O₂/SiH₄ = "
    f"{FLOWS_SCCM['Ar']:.0f}/{FLOWS_SCCM['O2']:.0f}/{FLOWS_SCCM['SiH4']:.0f} sccm"
)

# --- Main panel: run + results ------------------------------------------------
run = st.button("▶ Run Simulation", type="primary")

if run:
    try:
        with st.spinner("Solving plasma → transport → kinetics → level set …"):
            rate   = build_velocity_field(float(rf_power), float(pressure), float(rf_bias))
            solver = run_until_pinch(rate)
            state  = solver.state()
            fig    = build_trench_figure(solver, float(rf_power), float(rf_bias))
    except Exception as exc:  # surface solver errors instead of a blank screen
        st.exception(exc)
        st.stop()

    col_plot, col_diag = st.columns([3, 2])

    with col_plot:
        st.pyplot(fig)

    with col_diag:
        st.subheader("Diagnostics")
        st.metric("Final deposition time", f"{state.t:.2f} s")
        st.metric("Fill fraction", f"{state.fill_fraction * 100:.1f} %")

        if state.pinch_off:
            st.error(
                f"❌ VOID DETECTED — pinch-off at t = {state.t:.2f} s\n\n"
                f"Void area {state.void_area_m2 * 1e18:,.0f} nm² "
                f"({state.void_fraction * 100:.0f}% of trench), "
                f"seal depth ≈ {state.seal_depth_m * 1e9:.0f} nm."
            )
        else:
            st.success(
                f"✅ VOID-FREE FILL — no pinch-off within {T_MAX_S:.0f} s. "
                f"Trench filled to {state.fill_fraction * 100:.1f}%."
            )

        with st.expander("Full state summary"):
            st.json(state.summary())
else:
    st.info("Set the recipe in the sidebar and press **Run Simulation**.")
