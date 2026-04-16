import chaospy as cp
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

np.random.seed(42)

# matplotlib.pyplot options
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'legend.fontsize': 16,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})
# Note: set 'text.usetex': True if you have a LaTeX distribution installed
plt.rcParams['text.usetex'] = False 

class InfiniteBranches1D():
    def __init__(self, mu, seed_rv, n_samples=10000):
        self.mu = mu
        self.seed_rv = seed_rv
        self.n_samples = n_samples

        self.seed_rv_samples = np.atleast_2d(self.seed_rv.sample(n_samples))
        self.mu_samples = self.seed_rv_samples[0] * cp.Std(self.mu) + cp.E(self.mu)

    def f(self, c):
        """Compute the 1D Galerkin projection residual."""
        phi = self.phi(*self.seed_rv_samples) # (n_pc, n_samples)
        x = c @ phi # (n_samples,)
        
        # dx/dt = mu - sin(x)
        x_dot = ((self.mu_samples - np.sin(x)) @ phi.T) / self.n_samples # (n_pc,)
        return x_dot

    def jacobian(self, c):
        """Compute the Jacobian of the Galerkin projection."""
        phi = self.phi(*self.seed_rv_samples) # (n_pc, n_samples)
        x = c @ phi
        
        # Derivative of (mu - sin(x)) w.r.t x is -cos(x)
        J = ((-np.cos(x) * phi) @ phi.T) / self.n_samples
        return J

    def continuation(self, degree_pc, n_branch):
        """Find multiple branches using degree continuation."""
        self.solution = [[] for _ in range(n_branch)]
        
        for i in range(n_branch):
            degree_pc_iter = 0
            counter = 0 
            self.phi, self.phi_norm = cp.generate_expansion(degree_pc_iter, self.seed_rv, retall=True)
            self.n_pc = len(self.phi_norm)
            
            while degree_pc_iter <= degree_pc:
                counter += 1 
                if degree_pc_iter == 0:
                    # Randomly initialize between -3pi and 3pi to catch different branches
                    coeff_init = np.array([np.random.uniform(-3 * np.pi, 3 * np.pi)])
                else:
                    tmp = np.zeros(self.n_pc)
                    tmp[:len(coeff_init)] = coeff_init
                    coeff_init = tmp
                
                sol = root(self.f, coeff_init, method='lm', tol=1e-8, jac=self.jacobian)
                loss = np.sum(np.abs(self.f(sol.x)))
                c = sol.x
                
                # Check if we landed on a unique branch (avoiding duplicates)
                control = True
                if degree_pc_iter == 0 and i > 0:
                    for j in range(i):
                        if len(self.solution[j]) > 0 and np.isclose(c, self.solution[j][0][0], atol=1.0).all():
                            control = False
                            break
                
                if (loss < 1e-6) and control:
                    self.solution[i].append((c.copy(), degree_pc_iter)) 
                    degree_pc_iter += 1
                    counter = 0 
                    coeff_init = c.copy()
                    self.phi, self.phi_norm = cp.generate_expansion(degree_pc_iter, self.seed_rv, retall=True)
                    self.n_pc = len(self.phi_norm)
                
                if counter > 50: 
                    print(f"Stopping branch {i}: reached max attempts.")
                    break
            
            if len(self.solution[i]) > 0:
                print(f"Branch sequence found for branch {i}")

    def plot_x_mu(self, n_branch):
        """Plot the analytical branches against the PCE approximations."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(r"$x$")
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        mu_grid = xi_grid * cp.Std(self.mu) + cp.E(self.mu)
        
        # --- Plot Exact Analytical Branches ---
        for n in range(-3, 4):
            # Stable branches
            ax.plot(mu_grid, np.arcsin(mu_grid) + 2*np.pi*n, 'k', linewidth=3, zorder=1, 
                    label='Exact Stable' if n==0 else "")
            # Unstable branches
            ax.plot(mu_grid, np.pi - np.arcsin(mu_grid) + 2*np.pi*n, 'gray', linewidth=3, 
                    linestyle=':', zorder=1, label='Exact Unstable' if n==0 else "")
            
        # --- Plot PCE approximations ---
        max_deg = max([deg for branch in self.solution for (_, deg) in branch if branch]) 
        colors = plt.cm.coolwarm(np.linspace(0, 1, max_deg + 1))
        
        for i in range(min(n_branch, len(self.solution))):
            for j in range(len(self.solution[i])):
                coeffs, deg = self.solution[i][j]
                
                phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
                approx = coeffs @ phi_eval 
                
                if deg == max_deg:
                    ax.plot(mu_grid, approx, color=colors[deg], linewidth=2.0, zorder=5, linestyle='--',
                            marker='o', markersize=6, markevery=30, 
                            label=rf'PCE $u_{{{deg}}}$' if i==0 else "")

        ax.grid(True, alpha=0.3)
        ax.set_ylim(-10, 10)
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    
    degree_pc = 15
    n_branch_to_approximate = 6
    
    # We restrict mu to [-0.9, 0.9]. At exactly |mu| = 1, the branches undergo 
    # a saddle-node bifurcation (infinite gradient), requiring massive PCE degrees to resolve.
    model = InfiniteBranches1D(
        mu=cp.Uniform(-0.9, 0.9),
        seed_rv=cp.J(cp.Uniform(-np.sqrt(3), np.sqrt(3))),
        n_samples=5000
    )

    print("\n=== Executing Degree Continuation ===")
    model.continuation(degree_pc=degree_pc, n_branch=n_branch_to_approximate)
    model.plot_x_mu(n_branch=n_branch_to_approximate)