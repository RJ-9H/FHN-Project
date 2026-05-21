"""
main.py
-------
Run and compare both FHN variants on a 2D periodic domain.



Outputs
-------
  fhn_comparison.png        : 4×3 baseline panel
  fhn_phase_plane.png       : nullclines + phase portraits
  fhn_dynamics.png          : wave speed + instability onset (2×4)
  fhn_robustness_reg.png    : Regular FHN Du sweep (Turing onset)
  fhn_robustness_mc.png     : MC-FHN a sweep (morphology / volume fraction)
"""

import numpy as np
import matplotlib.pyplot as plt

from FHNmodel import RegularFHN, MassConservedFHN
from Simulation import FHNSimulation
from Analyser import FHNAnalyser

# ── Shared parameters ──────────────────────────────────────────────────────────
#
# Both models use MC-aligned base parameters so the comparison is apples-to-apples.
# Du is the only parameter that differs meaningfully between the two models:
#   - MC-FHN: Du sets coarsening speed; patterns persist for any Du > 0
#   - Regular FHN: Du must be below the Turing threshold Du* ≈ 0.79 for patterns

BASE_PARAMS = dict(a=0.025, b=1.26, epsilon=0.5, Dv=5.0)

REG_PARAMS = dict(**BASE_PARAMS, Du = 1.0)   
MC_PARAMS  = dict(**BASE_PARAMS, Du=1.0)   

# ── Grid settings ──────────────────────────────────────────────────────────────
#
# Domain size must fit several wavelengths.
# λ* ≈ 8–12 for these parameters so a 50×50 domain holds ~4–6 stripes.
# 256×256 gives ~28 grid points per stripe — well above the ~10 needed for
# spectral accuracy.  Increasing resolution beyond this won't fix artifacts;
# only insufficient T causes the mottled textures seen previously.

GRID = dict(sizex=50.0, sizey=50.0, resx=128, resy=128)

DT         = 0.05
SAVE_EVERY = 20


# ── Helper: build, run, return simulation ──────────────────────────────────────

def run_sim(model, T, grid=None, seed=0):
    """
    Build, initialise, and run a simulation.

    Parameters
    ----------
    model : RegularFHN or MassConservedFHN instance
    T     : total simulation time
    grid  : dict of sizex/sizey/resx/resy  (defaults to global GRID)
    seed  : RNG seed for initial conditions
    """
    g = grid if grid is not None else GRID
    sim = FHNSimulation(model, **g, dt=DT, save_every=SAVE_EVERY)
    sim.set_initial_conditions(seed=seed)
    sim.BuildSpectralMasks()
    sim.run(T)
    return sim


# ── Main simulations ───────────────────────────────────────────────────────────
#
# T=2000 for both main runs.  At Du=0.25 the growth rate σ ≈ 0.025 so the
# e-folding time is ~40 time units; T=2000 gives ~50 e-folds — well past
# saturation into the final attractor.

T_MAIN = 2000

print("Running Regular FHN (main)...")
sim_reg = run_sim(RegularFHN(**REG_PARAMS), T=T_MAIN)

print("Running Mass-Conserved FHN (main)...")
sim_mc = run_sim(MassConservedFHN(**MC_PARAMS), T=T_MAIN)

# ── Mass diagnostics ───────────────────────────────────────────────────────────

mc_mass  = sim_mc.mass()
reg_mass = sim_reg.mass()

print(f"\nMassConservedFHN | initial: {mc_mass[0]:.6f} | "
      f"final: {mc_mass[-1]:.6f} | "
      f"drift: {abs(mc_mass[-1] - mc_mass[0]):.2e}  (expect ~1e-12)")

settled = reg_mass[reg_mass.size // 2:]
print(f"RegularFHN       | mass not conserved by design")
print(f"                 | settled mean ± std: "
      f"{settled.mean():.4f} ± {settled.std():.4f}")

# ── Analysers ──────────────────────────────────────────────────────────────────

ana_reg = FHNAnalyser(sim_reg)
ana_mc  = FHNAnalyser(sim_mc)

# ── Regular FHN: Du sweep straddling the Turing threshold ─────────────────────
#
# Turing thresholds (numerically verified for these parameters):
#   Lower branch u*=-0.83:  Du* ≈ 0.30
#   Upper branch u*=+0.73:  Du* ≈ 0.79
#
# Du ≥ 0.8 → no instability; homogeneous state is linearly stable.
#
# T is set per Du based on the inverse growth rate 5/σ:
#   Du=0.05: σ≈0.176  → T_min≈ 30   → T=500   (fast, lots of headroom)
#   Du=0.15: σ≈0.085  → T_min≈ 60   → T=1000
#   Du=0.25: σ≈0.025  → T_min≈200   → T=3000  (slow — needs long run)
#   Du=0.50: σ≈0.087  → T_min≈ 60   → T=1000
#   Du=0.75: σ≈0.012  → T_min≈400   → T=5000  (near threshold — very slow)
#
# Note: Du=0.75 is close to the upper-branch threshold so the final pattern
# may still be coarsening at T=5000; this is physically correct behaviour,
# not a numerical artifact.
'''
print("\nRunning Regular FHN Du sweep...")

reg_sweep_spec = [
    (0.05,  500,  'Du = 0.05'),
    (0.15, 1000,  'Du = 0.15'),
    (0.25, 3000,  'Du = 0.25 '),
    (0.50, 1000,  'Du = 0.50'),
    (0.75, 4000,  'Du = 0.75'),
]

reg_sweep = []
for Du, T_sw, label in reg_sweep_spec:
    print(f"  Du={Du}  T={T_sw}")
    s = run_sim(RegularFHN(**{**BASE_PARAMS, 'Du': Du}), T=T_sw)
    reg_sweep.append({'param_value': Du, 'sim': s, 'label': f'Du = {Du}'})
'''
# ── MC-FHN: a sweep — volume fraction / morphology transition ─────────────────
#
# a shifts the homogeneous steady state u*, pinning the mean of u (mass is
# conserved).  This produces a morphological transition:
#
#   a=0.005 → u* near lower branch → <10% high-u → isolated droplets
#   a=0.015 → ~30% high-u          → connected clusters
#   a=0.025 → ~50% high-u          → labyrinths  (base param)
#   a=0.040 → ~70% high-u          → holes in high-u sea
#   a=0.060 → >90% high-u          → isolated low-u droplets (inverted)
#
# T=2000 throughout — MC-FHN is always unstable so patterns form quickly
# regardless of a.
'''
print("\nRunning Mass-Conserved FHN a sweep...")

mc_sweep_spec = [
    (0.05,  500,  'Du = 0.05'),
    (0.15, 1000,  'Du = 0.15'),
    (0.25, 3000,  'Du = 0.25 )'),
    (0.50, 1000,  'Du = 0.50'),
    (0.75, 5000,  'Du = 0.75'),
]

mc_sweep = []
for Du, T_sw, label in mc_sweep_spec:
    print(f"  Du={Du} T={T_sw}")
    s = run_sim(MassConservedFHN(**{**BASE_PARAMS, 'Du': MC_PARAMS['Du']}),
                T=T_sw)
    mc_sweep.append({'param_value': Du, 'sim': s, 'label': f'Du = {Du}'})
'''
# ── Plots ──────────────────────────────────────────────────────────────────────

# 1. Baseline 4×3 comparison
fig_cmp = ana_reg.plot_comparison(ana_mc)
fig_cmp.savefig("fhn_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: fhn_comparison.png")

# 2. Phase plane — uses corrected steady_state() starting guess in Analyser
fig_phase = ana_reg.plot_phase_plane(ana_mc)
fig_phase.savefig("fhn_phase_plane.png", dpi=150, bbox_inches="tight")
print("Saved: fhn_phase_plane.png")

'''
# 3. Dynamics: wave speed (row 0) + instability onset (row 1)
fig_dyn, fig_rob_reg, fig_rob_mc = ana_reg.plot_dynamics_comparison(
    other            = ana_mc,
    reg_sweep        = reg_sweep,
    mc_sweep         = mc_sweep,
    sweep_param_name = 'Du',
)

# Patch MC robustness axis labels 

for ax in fig_rob_mc.get_axes():
    if ax.get_xlabel():
        ax.set_xlabel('Du')
    if 'Du' in ax.get_title():
        ax.set_title(ax.get_title().replace('Du', 'a'))
fig_rob_mc.suptitle(
    'Robustness — MassConservedFHN  (sweep: Du)',
    fontsize=12)

fig_dyn.savefig("fhn_dynamics.png",           dpi=150, bbox_inches="tight")
fig_rob_reg.savefig("fhn_robustness_reg.png", dpi=150, bbox_inches="tight")
fig_rob_mc.savefig("fhn_robustness_mc.png",   dpi=150, bbox_inches="tight")

print("Saved: fhn_dynamics.png")
print("Saved: fhn_robustness_reg.png  "
      "(Regular FHN: Du sweep [0.05–0.75], straddling Turing threshold)")
print("Saved: fhn_robustness_mc.png   "
      "(MC-FHN: a sweep [0.005–0.060], droplet→labyrinth→hole)")
'''
plt.show()