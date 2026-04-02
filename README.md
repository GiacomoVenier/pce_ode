# PCE-ODE: Parametric Equilibrium Analysis via Polynomial Chaos

This repository contains a computational framework for the study of equilibrium states in nonlinear dynamical systems under parameter uncertainty. Using Polynomial Chaos Expansion (PCE), the software tracks steady-state branches and analyzes their stability and bifurcations.

## Mathematical Background

The project addresses the root-finding problem for a parametric vector field $f(u, \mu + \sigma\xi) = 0$. The equilibrium branch is approximated using a truncated polynomial series:
$$
u_{N}(\xi)=\sum_{j=0}^{N} \hat{u}_{N,j}\,\Phi_j(\xi).
$$

The theoretical framework, based on the Newton-Kantorovich theorem, guarantees that the Galerkin projections converge stably to the true equilibrium branches. It also ensures that the discrete system preserves the uniqueness of the continuous problem, avoiding spurious solutions even in multi-dimensional systems through specific radial coercivity constraints.

## Simulation Objectives

The numerical simulations in this repository are designed to empirically validate the theoretical guarantees:
* Compute the Galerkin approximations of the equilibrium branches, utilizing least-squares approximations as a reliable initial guess for the Newton root-finding method.
* Evaluate the residual decay as the polynomial degree N increases to confirm the theoretically expected uniform convergence rates.
* Isolate and track multiple equilibrium branches near bifurcation points using spectral clustering, verifying that the polynomial approximation correctly captures the topological structure of the continuous problem.

## Repository Structure

### 1. bifurcation_plots/
Analysis of equilibrium branches for fundamental normal forms:
* pitchfork/, s_shaped/, saddle_node/, transcritical/

### 2. software_1_d/
Core computational toolkit for one-dimensional equilibrium analysis.
* least_squares.ipynb: Implements Non-Intrusive Point Collocation (NIPC) via least squares regression to find initial equilibrium coefficients.
* main.ipynb & main_numerical.ipynb: Core numerical solvers generating and validating the accepted polynomial solutions.
* residuals.ipynb: Calculates and plots the residual decay against the polynomial degree.
* spectral_clustering.ipynb: Utilizes scikit-learn clustering to identify and separate distinct solution branches in multi-stable regimes.
* nf_functions.py: Defines the normal form vector fields, Legendre basis matrices, Wigner 3j symbols, and plotting utilities.

### 3. software_multi_d/
Multi-dimensional equilibrium studies:
* lorenz.py: Analysis of steady states and sensitivity for the Lorenz system.

### 4. misc/
Algebraic and functional tools:
* groebner.ipynb & resultants.ipynb: Elimination methods to exactly solve for equilibrium points.

## Requirements

* Python 3.8+
* Dependencies listed in the respective folder requirements files.

## Authorship

Author: Giacomo Venier
Focus: Polynomial Chaos Expansion, Equilibrium Stability, and Computational Mathematics.