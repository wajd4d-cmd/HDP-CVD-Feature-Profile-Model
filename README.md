# HDP-CVD Feature Profile Model (FPM)

**A predictive virtual-metrology tool for HDP-CVD gap-fill failures.**

FabHeat-X is an interactive simulator that predicts **trench voiding** and **corner clipping**
during High-Density-Plasma Chemical-Vapor-Deposition (HDP-CVD) dielectric gap-fill. It couples
a multi-layer physics pipeline — from bulk plasma to a moving deposition front — to a live
Streamlit dashboard, so a process engineer can dial in a recipe and watch the trench
cross-section seal (or fill) in real time.

The model targets the central failure mode of HDP-CVD: as the trench mouth deposits faster than
the floor, an overhang forms ("breadloafing") and can seal a void before the feature is filled.
The simulator resolves that competition from first principles and renders the resulting profile.

---

## Reactive UI — real-time rendering on every slider change

The dashboard recomputes the **entire L0 → L4 physics chain** whenever an input changes; there is
no "Run" button to press. Each distinct recipe is solved once and cached
(`st.cache_data`, keyed on the recipe), so revisiting a setting re-renders instantly while new
settings solve on demand. Move a slider, and the trench image, fill fraction, void area, and
seal depth update live.

### Dynamic inputs

| Input | Range | Effect |
|-------|-------|--------|
| **RF Power** | 1000 – 5000 W | Sets plasma density (ion flux); drives deposition and sputter magnitude and the time-to-seal. |
| **RF Bias** | 0 – 200 V | Sets ion-impact energy; the primary lever on sputter yield and therefore the S/D ratio. |
| **Pressure** | 1 – 10 mTorr | Sets neutral density and transport; swings deposition rate ~10x and time-to-seal ~10x. |

Fixed process geometry: 10:1 aspect-ratio trench, 100 nm mouth, Ar/O2/SiH4 = 100/50/30 sccm.

---

## Physics pipeline (L0 → L4)

| Layer | Module | Responsibility |
|-------|--------|----------------|
| **L0** | `fpm_L0_global_plasma.py` | Global plasma solver — bulk `n_e`, `T_e`, sheath, ion energy/angle (IEADF) from the recipe. |
| **L1** | `fpm_L1_feature_transport.py` | In-feature neutral/ion transport — depth-resolved flux and charge-deflection angle. |
| **L2** | `fpm_L2_surface_kinetics.py` | Surface kinetics & heat — self-consistent temperature, Langmuir-Kisliuk sticking, Arrhenius adatom diffusion. |
| **L3** | `fpm_L3_rate_assembly.py` | Rate assembly — Yamamura-Tawara sputter, deposition, redeposition → net normal velocity `V_n(z)`. |
| **L4** | `fpm_L4_level_set.py` | Level-set evolution — HJ-WENO5 + explicit SSP-RK3 advance of the interface to pinch-off. |

### Yamamura-Tawara sputter kinetics and the U_s = 4.0 eV calibration

L3 computes the angle-resolved sputter rate with the **Yamamura-Tawara** yield model: a
Thomas-Fermi reduced nuclear-stopping energy term with a threshold factor
`(1 - sqrt(E_th / E))^s`, folded against the L0 ion-energy-and-angle distribution (IEADF) and
the local surface orientation (grazing on sidewalls, near-normal on the floor). The deposition-
to-sputter balance is captured by the **S/D ratio**, which is what makes the fill outcome respond
to bias and power.

The sputter threshold energy scales with the surface binding energy:
`E_th = U_s * (1.9 + 3.8/mu + 0.134 * mu^1.24)`. At the original `U_s = 6.4 eV`, the threshold sat
just below the typical ion energy, so the threshold factor crushed the yield to **S/D ~ 0.03** and
the profile barely responded to the RF sliders. Calibrating `U_s` to **4.0 eV** — the lower end of
the documented 4–8 eV range for SiO2 — lowers the threshold and brings the IEADF-averaged S/D ratio
into the **responsive ~0.1–0.3 band** at typical bias. With this calibration, RF Power and RF Bias
visibly shift the seal depth, void area, and fill fraction, while the model remains in the
physically realistic breadloaf-and-seal regime.

---

## Quick start

```bash
pip install -r requirements.txt      # numpy, scipy, matplotlib, streamlit
streamlit run dashboard.py
```

Set the recipe in the sidebar (RF Power, RF Bias, Pressure) and the trench cross-section renders
live. The result panel reports final deposition time, fill fraction, and — on failure — the void
area and seal depth.

Each physics layer is also runnable standalone for its own self-test:

```bash
python fpm_L3_rate_assembly.py       # prints the L3 rate-assembly digest, etc.
```

---

## Output

The dashboard renders the trench cross-section with deposited solid, open/gas region, the
interface contour, and any **sealed void** highlighted, alongside diagnostics:

- **Final deposition time** and **fill fraction**
- **Void verdict** — pinch-off detection with void area and seal depth, or a void-free-fill result
- Full state summary (per-layer scalar digest)

---

## Project files

| File | Purpose |
|------|---------|
| `dashboard.py` | Reactive Streamlit UI — drives L0 → L4 and renders the trench cross-section. |
| `fpm_L0_global_plasma.py` | L0 global plasma solver. |
| `fpm_L1_feature_transport.py` | L1 in-feature transport solver. |
| `fpm_L2_surface_kinetics.py` | L2 surface kinetics & heat solver. |
| `fpm_L3_rate_assembly.py` | L3 rate assembly — Yamamura-Tawara sputter and net velocity. |
| `fpm_L4_level_set.py` | L4 level-set interface evolution (HJ-WENO5 / SSP-RK3). |
| `visualizer.py` | Matplotlib trench cross-section rendering. |
| `requirements.txt` | Runtime dependencies. |
