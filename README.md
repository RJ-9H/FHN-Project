This is a project for a class on Complex Systems. Here we simulate the FitzHugh-Nagumo mode. 
The pdes for both mass conserved and regular FHN are solved using a pseudo-spectral method (Integrating Factors).
This is done on two different boundary conditions (Grid.py for Neumann boundary conditions and torus for periodic boundaries)
main and simulation are for periodic and the python files with grid infront use the grid boundaries.
The models and analyser are the same for both.

For the pseudo spectral method read: https://arxiv.org/pdf/2305.08998
It should also be noted that the Torus2D and Simulation is taken from a project on Incompressible Schrodinger Flow. 
Torus was taken from 3D-2D and simulation uses a different numerical method for PDEs but the structure is similar.

For literature on the FHN-model please consult: https://arxiv.org/pdf/2404.11403

Refer to: https://github.com/zhengshenastro/FHNSim for gpu based speed up. This code is adapted from mine but runs on gpu based code (Jax).
This is for testing the models robustness for parameter changes which is computationally expensive.
