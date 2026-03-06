# Physics-Guided-Diffusion-with-Kolmogorov-Arnold-Neural-Operators

# Physics-Guided Diffusion with Kolmogorov-Arnold Neural Operators

This project implements a hybrid scientific ML architecture combining:

• Physics-Informed Kolmogorov-Arnold Networks (KAN)  
• Fourier Neural Operators (FNO)  
• Diffusion Models for turbulent flow generation  

The system solves and generates Navier–Stokes flow fields.

## Features

- Navier–Stokes physics residual training
- Reynolds number conditioning
- Fourier Neural Operator spectral layers
- Diffusion-based turbulence generator
- Physics correction using PDE residual minimization
- CFD evaluation metrics (drag / lift)
- Flow visualization (velocity, vorticity, turbulence)
- Symbolic extraction of learned functions

## Pipeline

Noise → Diffusion Model → Generated Flow → Physics Correction → Physically Consistent Flow

## Run Training

Train solver:

python training/train_solver.py

Train diffusion model:

python training/train_diffusion.py

Generate flow:

python generate_flow.py
