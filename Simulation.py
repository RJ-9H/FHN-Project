"""
FHNSimulation.py
----------------
2D spectral simulation of the FitzHugh-Nagumo system.

Inherits Torus2D for the periodic grid setup and spectral operators.
Accepts any FHNBase subclass as the physical model.
"""

import numpy as np
import scipy.fft as fft
from Torus import Torus2D
from FHNmodel import RegularFHN, MassConservedFHN


class FHNSimulation(Torus2D):
    """
    Parameters
    ----------
    model       : RegularFHN or MassConservedFHN instance
    sizex/sizey : physical domain lengths
    resx/resy   : grid resolution (number of points)
    dt          : time step
    save_every  : store state every n steps
    """

    def __init__(self, model, sizex, sizey, resx, resy,
                 dt=0.05, save_every=20):
        super().__init__(sizex, sizey, resx, resy)

        self.model = model
        self.dt = dt
        self.save_every = save_every

        # Fields (set by set_initial_conditions)
        self.u = None
        self.v = None

        # History storage
        self.u_history = []
        self.v_history = []
        self.t_history = []

        # Spectral masks (set by BuildSpectralMasks)
        self.linear_mask_u = None
        self.linear_mask_v = None

    def BuildSpectralMasks(self):
        masks = self.model.build_masks(self.k2, self.k4, self.dt)
        self.model.masks = masks          # stash on the model for SpectralStep

    def SpectralStep(self):
        u_hat = fft.fftn(self.u)
        v_hat = fft.fftn(self.v)
        f_hat = fft.fftn(self.model.f(self.u, self.v))
        g_hat = fft.fftn(self.model.g(self.u, self.v))

        u_hat_new, v_hat_new = self.model.spectral_update(
            u_hat, v_hat, f_hat, g_hat, self.k2, self.dt
        )

        self.u = np.real(fft.ifftn(u_hat_new))
        self.v = np.real(fft.ifftn(v_hat_new))

    # ── Initial conditions ───────────────────────────────────────────────────

    def set_initial_conditions(self, u0=None, v0=None, seed=42):
        """
        Set initial fields.  Default: small random perturbation around (0,0).
        Custom arrays must match (resx, resy).
        """
        rng = np.random.default_rng(seed)
        shape = (self.resx, self.resy)
        self.u = u0 if u0 is not None else 0.1 * rng.standard_normal(shape)
        self.v = v0 if v0 is not None else 0.1 * rng.standard_normal(shape)

    # ── Time-stepping loop ───────────────────────────────────────────────────

    def run(self, T):
        """
        Run the simulation for total time T.

        Parameters
        ----------
        T : total simulation time
        """
        if self.linear_mask_u is None:
            self.BuildSpectralMasks()

        if self.u is None:
            self.set_initial_conditions()

        n_steps = int(T / self.dt)

        for step in range(n_steps):
            self.SpectralStep()

            if step % self.save_every == 0:
                self.u_history.append(self.u.copy())
                self.v_history.append(self.v.copy())
                self.t_history.append(step * self.dt)

        self.u_history = np.array(self.u_history)
        self.v_history = np.array(self.v_history)
        self.t_history = np.array(self.t_history)
        print(f"Done: {self.model} | {n_steps} steps")

    # ── Mass diagnostic ──────────────────────────────────────────────────────

    def mass(self):
        """Total mass ∫∫u dA at each saved time step."""
        return self.u_history.sum(axis=(1, 2)) * self.dx * self.dy