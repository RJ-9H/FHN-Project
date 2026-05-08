"""
main.py
-------
Run and compare both FHN variants on a 2D periodic domain.

File structure (same folder):
    Torus.py
    FHNmodel.py
    Simulation.py
    Analyser.py
    main.py
"""

import matplotlib.pyplot as plt

from FHNmodel import RegularFHN, MassConservedFHN
from Simulation import FHNSimulation
from Analyser import FHNAnalyser

# ── Shared parameters ─────────────────────────────────────────────────────────

# Mass-conserved FHN parameters (eqs. 12 & 13)
PARAMS = dict(a=0.025, b=1.26, epsilon=0.5, Du=1.0, Dv=5.0)

# Grid: 100x100 physical domain at 256x256 resolution
GRID = dict(sizex=50, sizey=50, resx=128, resy=128)

# Simulation time and stepping
T          = 1251.0
DT         = 0.05
SAVE_EVERY = 20

# ── Instantiate models ────────────────────────────────────────────────────────

regular   = RegularFHN(**PARAMS)
conserved = MassConservedFHN(**PARAMS)

# ── Set up simulations ────────────────────────────────────────────────────────

sim_reg = FHNSimulation(regular,   **GRID, dt=DT, save_every=SAVE_EVERY)
sim_mc  = FHNSimulation(conserved, **GRID, dt=DT, save_every=SAVE_EVERY)

# Identical initial conditions for a fair comparison
sim_reg.set_initial_conditions(seed=0)
sim_mc.set_initial_conditions(seed=0)

# Precompute integrating-factor masks on Simulation (owns k2/k4 via Torus2D)
# masks are stored as sim.linear_mask_u / sim.linear_mask_v and passed
# into model.spectral_update each step — the model never owns them
sim_reg.BuildSpectralMasks()
sim_mc.BuildSpectralMasks()

# ── Run ───────────────────────────────────────────────────────────────────────

print("Running Regular FHN...")
sim_reg.run(T)

print("Running Mass-Conserved FHN...")
sim_mc.run(T)

# ── Mass conservation check ───────────────────────────────────────────────────
# For MassConservedFHN, ∫∫u dA should remain constant throughout the run.
# A drift here indicates a bug in the spectral stepping or mask computation.

mc_mass  = sim_mc.mass()
reg_mass = sim_reg.mass()

print(f"\nMass-conserved | initial mass: {mc_mass[0]:.4f} | "
      f"final mass: {mc_mass[-1]:.4f} | "
      f"drift: {abs(mc_mass[-1] - mc_mass[0]):.2e}")

print(f"Regular FHN    | initial mass: {reg_mass[0]:.4f} | "
      f"final mass: {reg_mass[-1]:.4f} | "
      f"drift: {abs(reg_mass[-1] - reg_mass[0]):.2e}")

# ── Analyse and plot ──────────────────────────────────────────────────────────

ana_reg = FHNAnalyser(sim_reg)
ana_mc  = FHNAnalyser(sim_mc)

fig = ana_reg.plot_comparison(ana_mc)
plt.savefig("fhn_2d_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved: fhn_2d_comparison.png")