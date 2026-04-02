# PCE-ODE: Parametric Equilibrium Analysis via Polynomial Chaos

This repository contains a computational framework for the study of equilibrium states in nonlinear dynamical systems under parameter uncertainty. Using Polynomial Chaos Expansion (PCE), the software tracks steady-state branches and analyzes their stability and bifurcations.

## Mathematical Background

The project addresses the root-finding problem for a parametric vector field $f(u, \bar{\mu} + \sigma \xi) = 0$, where the equilibrium branch is approximated as:

$$u(\xi) \approx \sum_{j=0}^{N} \hat{u}_j \Phi_j(\xi)$$

The theoretical framework, based on the Newton-Kantorovich theorem and specific coercivity constraints, guarantees that the Galerkin projections converge stably to the true equilibrium branches. It also ensures that the discrete system preserves the uniqueness of the continuous problem, avoiding spurious solutions even in multi-dimensional systems.

## Simulation Objectives

The numerical simulations in this repository are designed to empirically validate the theoretical guarantees. Specifically, the software:
* Computes the Galerkin approximations of the equilibrium branches
* Evaluates the residual decay as the polynomial degree $N$ increases to confirm the theoretically expected uniform convergence rates ($O(N^{-s})$).
* Isolates and tracks multiple equilibrium branches near bifurcation points to verify that the polynomial approximation correctly captures the topological structure of the continuous problem.

## Repository Structure

### 1. bifurcation_plots/
Analysis of equilibrium branches for fundamental normal forms:
* pitchfork/, s_shaped/, saddle_node/, transcritical/

### 2. software_1_d/
Core tools for 1D equilibrium analysis:
* least_squares.ipynb: Solver for the initial equilibrium coefficients.
* residuals.ipynb: Accuracy and residual decay analysis.
* spectral_clustering.ipynb: Identification of distinct solution branches using oscillating solutions.
* nf_functions.py: Function definitions.

### 3. software_multi_d/
Multi-dimensional equilibrium studies:
* lorenz.py: Analysis of steady states and sensitivity for the Lorenz system.

### 4. misc/
Miscellaneous codes

## Requirements

* Python 3.8+
* numpy, scipy, matplotlib, sympy, pandas

## Authorship
Author: Giacomo Venier
Focus: Polynomial Chaos Expansion, Equilibrium Stability, and Computational Mathematics.