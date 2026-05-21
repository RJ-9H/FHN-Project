
import numpy as np
import matplotlib.pyplot as plt

from FHNmodel import RegularFHN, MassConservedFHN
from GridSimulation import FHNSimulation
from Analyser import FHNAnalyser

# ── Parameters — paper exact ───────────────────────────────────────────────────
PARAMS = dict(a=0.1, b=1.0, epsilon=0.005, Du=1.0, Dv=1.0)


GRID = dict(sizex=200.0, sizey=200.0, resx=256, resy=256)

DT         = 0.05
SAVE_EVERY = 40          
T          = 1500.0


def make_spiral_ics(sizex, sizey, resx, resy):
    x = np.linspace(-sizex / 2, sizex / 2, resx)   
    y = np.linspace(-sizey / 2, sizey / 2, resy)
    X, Y = np.meshgrid(x, y, indexing='ij')
    width = sizex / 4                               
    u0 = np.tanh(X / width)                         
    v0 = np.tanh(Y / width)                        
    return u0, v0

u0, v0 = make_spiral_ics(**GRID)

# ── Helper ─────────────────────────────────────────────────────────────────────

def run_sim(model, u0, v0):
    sim = FHNSimulation(model, **GRID, dt=DT, save_every=SAVE_EVERY)
    sim.set_initial_conditions(u0=u0, v0=v0)
    sim.BuildSpectralMasks()
    sim.run(T)
    return sim

# ── Run ────────────────────────────────────────────────────────────────────────

print("Running Regular FHN...")
sim_reg = run_sim(RegularFHN(**PARAMS), u0, v0)

print("Running Mass-Conserved FHN...")
sim_mc = run_sim(MassConservedFHN(**PARAMS), u0, v0)

# ── Mass diagnostics ───────────────────────────────────────────────────────────

mc_mass  = sim_mc.mass()
reg_mass = sim_reg.mass()

print(f"\nMassConservedFHN | drift: {abs(mc_mass[-1] - mc_mass[0]):.2e}  (expect ~1e-10)")
print(f"RegularFHN       | final mass: {reg_mass[-1]:.4f}")

# ── Analysers ──────────────────────────────────────────────────────────────────

ana_reg = FHNAnalyser(sim_reg)
ana_mc  = FHNAnalyser(sim_mc)

# ── Helper: find saved index nearest to a target time ─────────────────────────

def nearest_idx(t_history, t_target):
    return int(np.argmin(np.abs(t_history - t_target)))

# ── Plots ──────────────────────────────────────────────────────────────────────

# 1. Baseline 4×3 comparison
fig_cmp = ana_reg.plot_comparison(ana_mc)
fig_cmp.savefig("fhn_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: fhn_comparison.png")

# 2. Phase plane — uses corrected steady_state() starting guess in Analyser
fig_phase = ana_reg.plot_phase_plane(ana_mc)
fig_phase.savefig("fhn_phase_plane.png", dpi=150, bbox_inches="tight")
print("Saved: fhn_phase_plane.png")


plt.show()