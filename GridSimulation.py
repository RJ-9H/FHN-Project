"""
FHNSimulation.py
----------------
2D pseudo-spectral simulation of the FitzHugh-Nagumo system
with Neumann (zero-flux) boundary conditions.

Inherits Grid2D (DCT-based) instead of Torus2D (FFT-based).
The only numerical change from the periodic version is:
  - fftn/ifftn  →  dctn/idctn
  - FFT wavenumbers (2*pi*n/L)  →  DCT wavenumbers (n*pi/L)

Everything else — model classes, analyser, main — is unchanged.
"""

import numpy as np

from Grid import Grid2D
from FHNmodel import RegularFHN, MassConservedFHN


class FHNSimulation(Grid2D):
    """
    Pseudo-spectral ETD simulation on a 2D grid with Neumann BCs.

    Parameters
    ----------
    model       : RegularFHN or MassConservedFHN instance
    sizex/sizey : physical domain lengths
    resx/resy   : grid resolution
    dt          : time step
    save_every  : store state every n steps
    """

    def __init__(self, model, sizex, sizey, resx, resy,
                 dt=0.05, save_every=20):
        super().__init__(sizex, sizey, resx, resy)

        self.model = model
        self.dt = dt
        self.save_every = save_every

        # Fields
        self.u = None
        self.v = None

        # History
        self.u_history = []
        self.v_history = []
        self.t_history = []

        # Spectral masks (built by BuildSpectralMasks)
        self.linear_mask_u = None
        self.linear_mask_v = None

    # ── Build exponential integrating factor masks ───────────────────────────

    def BuildSpectralMasks(self):
        """
        Precompute exp(-D * k^p * dt) for the linear diffusion term.

        Regular FHN        : p=2  (Laplacian)
        Mass-conserved FHN : p=4  (bilaplacian, from -∇²[..∇²u..])

        Using DCT wavenumbers k_n = n*pi/L encodes Neumann BCs.
        The exponential integrating factor handles the stiff linear
        part exactly — critical for the k^4 operator.
        """
        m = self.model

        if isinstance(m, RegularFHN):
            self.linear_mask_u = np.exp(-m.Du * self.k2 * self.dt)
            self.linear_mask_v = np.exp(-m.Dv * self.k2 * self.dt)

        elif isinstance(m, MassConservedFHN):
            self.linear_mask_u = np.exp(-m.Du * self.k4 * self.dt)
            self.linear_mask_v = np.exp(-m.Dv * self.k4 * self.dt)

        else:
            raise TypeError(f"Unknown model type: {type(m)}")

    # ── One spectral time step ───────────────────────────────────────────────

    def SpectralStep(self):
        """
        Advance u and v by one time step using pseudo-spectral ETD.

        Nonlinear terms f(u,v) and g(u,v) are evaluated in real space
        then transformed to DCT space — pseudo-spectral step.

        Linear diffusion handled exactly via precomputed masks — ETD step.

        Regular FHN (eqs. 8 & 9 in DCT space):
            û_{n+1} = mask_u * (û_n + dt * f̂)
            v̂_{n+1} = mask_v * (v̂_n + dt * ε*ĝ)

        Mass-conserved FHN (eqs. 12 & 13 in DCT space):
            û_{n+1} = mask_u * (û_n + dt * k² * f̂)
            v̂_{n+1} = mask_v * (v̂_n + dt * ε * k² * ĝ)
        """
        m = self.model

        # Transform current fields to DCT space
        u_hat = self.dct(self.u)
        v_hat = self.dct(self.v)

        # Evaluate nonlinear terms in real space, then transform
        f_hat = self.dct(m.f(self.u, self.v))
        g_hat = self.dct(m.g(self.u, self.v))

        if isinstance(m, RegularFHN):
            u_hat_new = self.linear_mask_u * (u_hat + self.dt * f_hat)
            v_hat_new = self.linear_mask_v * (v_hat + self.dt * m.epsilon * g_hat)

        elif isinstance(m, MassConservedFHN):
            u_hat_new = self.linear_mask_u * (u_hat + self.dt * self.k2 * f_hat)
            v_hat_new = self.linear_mask_v * (v_hat + self.dt * m.epsilon * self.k2 * g_hat)

        # Transform back to real space
        self.u = self.idct(u_hat_new)
        self.v = self.idct(v_hat_new)

    # ── Initial conditions ───────────────────────────────────────────────────

    def set_initial_conditions(self, u0=None, v0=None, seed=42):
        """
        Set initial fields. Accepts custom arrays or defaults to
        small random perturbation around (0, 0).
        """
        rng = np.random.default_rng(seed)
        shape = (self.resx, self.resy)
        self.u = u0 if u0 is not None else 0.1 * rng.standard_normal(shape)
        self.v = v0 if v0 is not None else 0.1 * rng.standard_normal(shape)

    # ── Time-stepping loop ───────────────────────────────────────────────────

    def run(self, T):
        """Run simulation for total time T."""
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

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def mass(self):
        """Total mass ∫∫u dA at each saved step."""
        return self.u_history.sum(axis=(1, 2)) * self.dx * self.dy