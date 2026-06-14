"""
visualizer.py  --  FPM Virtual Fab :: Pinch-Off Void Renderer
=============================================================

Runs the full HDP-CVD Feature Profile Model (L0 -> L4) for the top-heavy
3000 W / 80 V recipe, advances the level set until the L4 module flags a
pinch-off void, and renders a publication-quality 2-D cross-section of the
frozen interface to ``pinch_off_void.png`` (for the repository README).

The plot shows:
  * the deposited solid / substrate          (phi <= 0)  -- filled
  * the open gas region                       (phi  > 0)  -- light
  * the SEALED VOID                            (enclosed open pocket) -- flagged
  * the interface                             (phi == 0)  -- outlined contour

Author role: Lead Coder
"""

from __future__ import annotations

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")                      # headless / CI-safe backend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from scipy import ndimage as ndi
except Exception:                          # pragma: no cover
    ndi = None

# =============================================================================
# 1. ROBUST MODULE IMPORTS
# =============================================================================
try:
    from fpm_L0_global_plasma import GlobalPlasmaSolver, ReactorInputs
    from fpm_L1_feature_transport import (FeatureTransportSolver,
                                          FeatureGeometry, TransportParams)
    from fpm_L2_surface_kinetics import SurfaceKineticsSolver, SurfaceParams
    from fpm_L3_rate_assembly import RateAssemblySolver
    from fpm_L4_level_set import LevelSetSolver, TrenchGeometry, LevelSetParams
except ImportError as exc:
    sys.stderr.write(
        "\n[FATAL] Could not import the FPM layer modules.\n"
        f"        -> {exc}\n\n"
        "        visualizer.py must run from the directory containing the\n"
        "        fpm_L0..fpm_L4 *.py files.\n\n")
    sys.exit(1)

# =============================================================================
# 2. EXPERIMENT DEFINITION
# =============================================================================
RECIPE = dict(power_w=3000.0, pressure_mtorr=5.0,
              flows_sccm={"Ar": 100.0, "O2": 50.0, "SiH4": 30.0},
              rf_bias_v=80.0)
W0_M = 100e-9
ASPECT_RATIO = 10.0
WALL_STICKING = 0.10

OUTFILE = "pinch_off_void.png"
T_MAX_S = 30.0                              # guard on simulated time


# =============================================================================
# 3. PHYSICS PIPELINE  (L0 -> L1 -> L2 -> L3)
# =============================================================================
def build_velocity_field():
    """Solve L0->L3 for the recipe and return the L3 RateState (carries V_n(z))."""
    inp = ReactorInputs(power_w=RECIPE["power_w"],
                        pressure_mtorr=RECIPE["pressure_mtorr"],
                        flows_sccm=RECIPE["flows_sccm"],
                        rf_bias_v=RECIPE["rf_bias_v"])
    plasma = GlobalPlasmaSolver(inp).solve()
    geom = FeatureGeometry(aspect_ratio=ASPECT_RATIO, w0_m=W0_M, n_z=200)
    transport = FeatureTransportSolver(plasma, geom,
                                       TransportParams(WALL_STICKING)).solve()
    surface = SurfaceKineticsSolver(plasma, transport, geom, SurfaceParams()).solve()
    return RateAssemblySolver(plasma, transport, surface).solve()


# =============================================================================
# 4. L4 ADVANCE LOOP  (run until pinch-off)
# =============================================================================
def is_pinched_off(solver) -> bool:
    """Convenience predicate over the L4 failure detector."""
    return bool(solver.detect_failures()["pinch_off"])


def run_until_pinch(rate):
    """Advance the level set until a void seals (or T_MAX). Return the solver."""
    solver = LevelSetSolver(geometry=TrenchGeometry(w0_m=W0_M,
                                                    aspect_ratio=ASPECT_RATIO),
                            params=LevelSetParams(scheme="weno5", cfl=0.4,
                                                  reinit_every=8),
                            rate_state=rate)
    print(f"[L4] advancing  (V_n mouth/bottom ratio = "
          f"{solver._Vn_src.max()/max(solver._Vn_src.min(), 1e-30):.1f}) ...")
    while solver.t < T_MAX_S and not is_pinched_off(solver):
        solver.advance(solver.t + 0.5, stop_on_pinch=True)
    return solver


# =============================================================================
# 5. RENDER  -> pinch_off_void.png
# =============================================================================
def render(solver, outfile: str = OUTFILE) -> str:
    """Plot the solid material, the sealed void, and the interface contour."""
    state = solver.state()
    phi = state.phi                                   # (nz, nx)  [m]
    x_nm = solver.x * 1e9
    z_nm = solver.z * 1e9

    # Identify the sealed-void cells (open region NOT vented to the top row).
    void_mask = np.zeros_like(phi, dtype=float)
    openm = phi > 0
    if ndi is not None and openm.any():
        lbl, n = ndi.label(openm)
        vented = set(np.unique(lbl[0, :])) - {0}
        for comp in range(1, n + 1):
            if comp not in vented:
                void_mask[lbl == comp] = 1.0

    # --- figure ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 7.4))
    C_SOLID = "#3b4a5a"        # deposited film / substrate
    C_OPEN = "#dfe7ef"         # open gas / chamber
    C_VOID = "#d7263d"         # the sealed void (failure)
    C_IFACE = "#00d2ff"        # interface outline

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
    ax.set_title("HDP-CVD Gap-Fill — Pinch-Off Void\n"
                 f"{RECIPE['power_w']:.0f} W / {RECIPE['rf_bias_v']:.0f} V bias, "
                 f"AR {ASPECT_RATIO:.0f}:1 trench", fontsize=11, fontweight="bold")

    # Annotate the failure.
    if state.pinch_off:
        ax.annotate(
            f"SEALED VOID\n{state.void_area_m2*1e18:,.0f} nm²  "
            f"({state.void_fraction*100:.0f}% of trench)\n"
            f"seal @ z ≈ {state.seal_depth_m*1e9:.0f} nm",
            xy=(0.0, state.seal_depth_m * 1e9 + 0.25 * solver.geom.depth_m * 1e9),
            xytext=(0.62, 0.30), textcoords="axes fraction",
            ha="left", va="center", fontsize=9, color=C_VOID, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_VOID, lw=1.4))

    legend = [
        Patch(facecolor=C_SOLID, label="Deposited solid (φ ≤ 0)"),
        Patch(facecolor=C_OPEN, label="Open / gas (φ > 0)"),
        Patch(facecolor=C_VOID, label="Sealed void"),
        Line2D([0], [0], color=C_IFACE, lw=1.6, label="Interface (φ = 0)"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, fontsize=8, frameon=False)

    fig.tight_layout()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), outfile)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# =============================================================================
# 6. ENTRY POINT
# =============================================================================
def main() -> int:
    print("=" * 70)
    print(" FPM Virtual Fab :: rendering pinch-off void")
    print("=" * 70)
    rate = build_velocity_field()
    solver = run_until_pinch(rate)
    state = solver.state()

    if state.pinch_off:
        print(f"[L4] PINCH-OFF at t = {state.t:.2f} s  |  "
              f"void = {state.void_area_m2*1e18:,.0f} nm² "
              f"({state.void_fraction*100:.1f}% of trench)")
    else:
        print(f"[L4] no void within {T_MAX_S:.0f} s "
              f"(fill = {state.fill_fraction*100:.1f}%); rendering current state.")

    path = render(solver)
    print(f"[OK] figure saved -> {os.path.basename(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
