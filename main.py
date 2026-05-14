"""
main.py
-------
Run and compare both FHN variants, then produce all dynamics figures.

Outputs
-------
  fhn_comparison.png        : 4x3 baseline panel
  fhn_phase_plane.png       : nullclines + phase portraits
  fhn_dynamics.png          : 2x4 wave-speed + instability-onset panel
  fhn_robustness_reg.png    : Regular FHN parameter sweep
  fhn_robustness_mc.png     : Mass-Conserved FHN parameter sweep
"""

import numpy as np
import matplotlib.pyplot as plt

from FHNmodel import RegularFHN, MassConservedFHN
from Simulation import FHNSimulation
from Analyser import FHNAnalyser

# ── Shared grid and time settings ─────────────────────────────────────────────

GRID = dict(sizex=50.0, sizey=50.0, resx=256, resy=256)
T, DT, SAVE_EVERY = 1251.0, 0.05, 20

# ── Base model parameters ─────────────────────────────────────────────────────
PARAMS  = dict(a=0.025, b=1.26, epsilon=0.5,  Du=1.0,     Dv=5.0)


# ── Helper: build, run, and return a simulation ───────────────────────────────

def run_sim(model, seed=0):
    sim = FHNSimulation(model, **GRID, dt=DT, save_every=SAVE_EVERY)
    sim.set_initial_conditions(seed=seed)
    sim.BuildSpectralMasks()
    sim.run(T)
    return sim


# ── Main simulations ──────────────────────────────────────────────────────────

print("Running Regular FHN...")
sim_reg = run_sim(RegularFHN(**PARAMS))

print("Running Mass-Conserved FHN...")
sim_mc = run_sim(MassConservedFHN(**PARAMS))

# ── Mass diagnostics ──────────────────────────────────────────────────────────

mc_mass  = sim_mc.mass()
reg_mass = sim_reg.mass()

print(f"\nMassConservedFHN | initial: {mc_mass[0]:.6f} | "
      f"final: {mc_mass[-1]:.6f} | "
      f"drift: {abs(mc_mass[-1] - mc_mass[0]):.2e}  (expect ~1e-12)")

settled = reg_mass[reg_mass.size // 2:]
print(f"RegularFHN       | mass not conserved by design")
print(f"                 | settled mean: {settled.mean():.4f} "
      f"± {settled.std():.4f}")

# ── Analysers ─────────────────────────────────────────────────────────────────

ana_reg = FHNAnalyser(sim_reg)
ana_mc  = FHNAnalyser(sim_mc)

# ── Parameter sweep for robustness ───────────────────────────────────────────
#
# Sweep Du for Regular FHN (Du controls Turing band position and width).
# Sweep Du for MC-FHN (controls coarsening speed and domain scale).


T_SWEEP = 2001

print("\nRunning Regular FHN Du sweep...")
reg_Du_values = [0.5, 1.0, 2.0, 5.0, 10.0]
reg_sweep = []
for Du in reg_Du_values:
    params = {**PARAMS, 'Du': Du}
    s = run_sim(RegularFHN(**params))
    reg_sweep.append({
        'param_value': Du,
        'sim':         s,
        'label':       f'Du = {Du}',
    })

print("Running Mass-Conserved FHN Du sweep...")
mc_Du_values = [0.5, 1.0, 2.0, 5.0, 10.0]
mc_sweep = []
for Du in mc_Du_values:
    params = {**PARAMS, 'Du': Du}
    s = run_sim(MassConservedFHN(**params))
    mc_sweep.append({
        'param_value': Du,
        'sim':         s,
        'label':       f'Du = {Du}',
    })

# ── Plots ─────────────────────────────────────────────────────────────────────

# 1. Baseline 4x3 comparison
fig_cmp = ana_reg.plot_comparison(ana_mc)
fig_cmp.savefig("fhn_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: fhn_comparison.png")

# 2. Phase plane
fig_phase = ana_reg.plot_phase_plane(ana_mc)
fig_phase.savefig("fhn_phase_plane.png", dpi=150, bbox_inches="tight")
print("Saved: fhn_phase_plane.png")

# 3. Full dynamics comparison:
#    wave speed (row 0) + instability onset (row 1) + robustness (figs 2-3)
fig_dyn, fig_rob_reg, fig_rob_mc = ana_reg.plot_dynamics_comparison(
    other            = ana_mc,
    reg_sweep        = reg_sweep,
    mc_sweep         = mc_sweep,
    sweep_param_name = 'Du',
)

fig_dyn.savefig("fhn_dynamics.png",        dpi=150, bbox_inches="tight")
fig_rob_reg.savefig("fhn_robustness_reg.png", dpi=150, bbox_inches="tight")
fig_rob_mc.savefig("fhn_robustness_mc.png",  dpi=150, bbox_inches="tight")
print("Saved: fhn_dynamics.png")
print("Saved: fhn_robustness_reg.png")
print("Saved: fhn_robustness_mc.png")

plt.show()