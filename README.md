# Transport Simulation Visualisations

This repository contains Python scripts for simulating and visualising transport phenomena, particularly in the context of porous media or filtration. The codes numerically solve partial differential equations (PDEs) involving advection, diffusion, and moving boundaries using explicit finite difference methods. Visualisations are provided using matplotlib animations.

## File Descriptions

### 1. `characteristic.py`
**Physics/Maths:**  
Simulates the propagation of characteristics for a linear advection equation in 2D, where each characteristic represents the path along which a property (e.g., concentration) is transported with constant velocity.

- **Governing equation:** $\frac{\partial \phi}{\partial t} + u \frac{\partial \phi}{\partial x} + s \frac{\partial \phi}{\partial y} = 0$
- **Numerical method:** Explicit characteristic tracing (Euler forward in time).
- **Physical context:** Illustrates the concept of characteristics in hyperbolic PDEs, relevant to transport in porous media.

### 2. `simple.py`
**Physics/Maths:**  
Solves a 2D advection equation with a moving boundary, representing the advance of a front (e.g., filtration or resin bed) in a porous medium. The velocity field and front position evolve according to the local concentration.

- **Governing equations:**  
  - Advection equation for the volume fraction$\phi$:  
    $\frac{\partial\phi}{\partial t} +\frac{\partial}{\partial x}(u\phi) +S\nabla\cdot(\boldsymbol{s}\phi) = 0$
  - Moving boundary $h(x, t)$ updated via a Stefan-like condition:  
    $\frac{\partial h}{\partial t} \propto \frac{\phi}{\phi_c - \phi}$
  - Velocity: $u = \frac{1}{1 - h}$
- **Numerical method:** Explicit finite difference (forward Euler in time, upwind in space).
- **Physical context:** Models the advance of a reaction or filtration front, with pressure drop calculation across the domain.

### 3. `full.py`
**Physics/Maths:**  
Implements a more advanced 2D advection-diffusion equation with a dynamically evolving boundary and variable velocity, including additional terms for diffusion and non-uniform velocity.

- **Governing equations:**  
  - Advection-diffusion equation for $\phi$:  
    $\frac{\partial \phi}{\partial t} + u \frac{\partial \phi}{\partial x} + s \frac{\partial \phi}{\partial y} = D \frac{\partial^2 \phi}{\partial x^2}$
  - Moving boundary $h(x, t)$ as in `simple.py`, with additional terms for diffusion and velocity gradients.
  - Velocity: $u = \frac{1}{1 - h}$
- **Numerical method:** Explicit finite difference (forward Euler in time, upwind for advection, central difference for diffusion).
- **Physical context:** Captures more realistic transport and reaction in porous media, with evolving boundaries and variable flow.

### 4. `simple.mp4`
A video output demonstrating the results of one of the simulations (possibly from `simple.py`).

## Dependencies

- numpy
- matplotlib
- tqdm

## Numerical Methods

All scripts use explicit finite difference schemes for time integration (forward Euler) and upwind or central differences for spatial derivatives, suitable for simulating advection-dominated transport and moving boundary problems in porous media.
