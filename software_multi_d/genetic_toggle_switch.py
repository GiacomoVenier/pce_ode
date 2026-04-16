import chaospy as cp
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(241)

# matplotlib.pyplot options
plt.rcParams.update({
    'font.size': 30,
    'axes.titlesize': 30,
    'axes.labelsize': 30,
    'legend.fontsize': 20,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30
})

plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 30
plt.rcParams['font.family'] = 'lmodern'

class ToggleSwitch():
    def __init__(self, mu, seed_rv, n_samples=10000):
        '''
        mu --> {chaospy.Distribution}
        seed_rv --> {chaospy.Distribution} (1D for a single parameter)
        n_samples --> {int}
        '''
        self.mu = mu
        self.seed_rv = seed_rv
        self.n_kl = len(self.seed_rv)
        self.n_samples = n_samples

        # For a 1D random variable, seed_rv.sample returns shape (n_samples,)
        # We wrap it in a 2D array to mimic the structure of the Lorenz code
        self.seed_rv_samples = np.atleast_2d(self.seed_rv.sample(n_samples)) # (1, self.n_samples)
        
        # Affine mapping to physical parameter
        self.mu_samples = self.seed_rv_samples[0] * cp.Std(self.mu) + cp.E(self.mu) # (self.n_samples)

    def f(self, c):
        '''
        compute F(c) = 0 for the Galerkin projection
        '''
        c = c.reshape(2, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) # (self.n_pc, self.n_samples)
        x, y = c.T @ phi # (2, self.n_samples)
        
        # Toggle Switch vector field
        x_dot = ((-x + self.mu_samples / (1 + y**2)) @ phi.T) / self.n_samples # (self.n_pc)
        y_dot = ((-y + self.mu_samples / (1 + x**2)) @ phi.T) / self.n_samples # (self.n_pc)
        
        return np.concatenate([x_dot, y_dot])

    def jacobian(self, c):
        c = c.reshape(2, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) # (self.n_pc, self.n_samples)
        x, y = c.T @ phi

        # Jacobian of the Toggle Switch vector field
        Jxx = (-phi @ phi.T) / self.n_samples
        Jxy = ((-self.mu_samples * 2 * y / (1 + y**2)**2 * phi) @ phi.T) / self.n_samples
        
        Jyx = ((-self.mu_samples * 2 * x / (1 + x**2)**2 * phi) @ phi.T) / self.n_samples
        Jyy = (-phi @ phi.T) / self.n_samples
        
        J = np.block([
                [Jxx, Jxy],
                [Jyx, Jyy]
            ])
        return J

    def run(self, degree_pc, n_init):
        '''
        solve F(c)=0 for n_init different initializations of the nonlinear solver
        '''
        self.phi, self.phi_norm = cp.generate_expansion(degree_pc, self.seed_rv, retall=True)
        self.n_pc = len(self.phi_norm)
        self.solution = np.zeros((n_init, self.n_pc, 2))
        self.coeff_init = []
        count = 0
        
        while count < n_init:
            # Generate smaller random initializations for stability with fractional terms
            self.coeff_init.append(2.0 * np.random.randn(self.n_pc, 2))
            sol = root(self.f, self.coeff_init[-1].ravel(), method='lm', tol=1e-8, jac=self.jacobian)
            loss = np.sum(np.abs(self.f(sol.x)))
            
            if (loss < 1e-6) and (not np.isclose(self.solution[:count], sol.x.reshape(2, self.n_pc).T).all(2).all(1).any()):
                self.solution[count] = sol.x.reshape(2, self.n_pc).T
                count += 1
                print(f"Random Init Found: {count}/{n_init}")

        self.samples_solution = (self.solution.transpose(0, 2, 1) @ self.phi(*self.seed_rv_samples))

    def continuation(self, degree_pc, n_branch):
        self.solution = [[] for _ in range(n_branch)]
        self.samples_solution = [[] for _ in range(n_branch)]
        
        for i in range(n_branch):
            degree_pc_iter = 0
            counter = 0 
            self.phi, self.phi_norm = cp.generate_expansion(degree_pc_iter, self.seed_rv, retall=True)
            self.n_pc = len(self.phi_norm)
            
            while degree_pc_iter <= degree_pc:
                counter += 1 
                if degree_pc_iter == 0:
                    # Different initializations to catch different branches
                    # The symmetric branch is near (1,1), asymmetric are scattered
                    coeff_init = 2.0 * np.random.randn(1, 2)
                else:
                    tmp = np.zeros((self.n_pc, 2))
                    tmp[:len(coeff_init)] = coeff_init
                    coeff_init = tmp.T
                
                sol = root(self.f, coeff_init.ravel(), method='lm', tol=1e-8, jac=self.jacobian)
                loss = np.sum(np.abs(self.f(sol.x)))
                c = sol.x.reshape(2, self.n_pc).T
                
                control = not any(np.isclose(c, self.solution[j][0][0]).all() for j in range(i) if self.solution[j]) if (degree_pc_iter == 0 and i > 0) else True
                
                if (loss < 1e-6) and control:
                    self.solution[i].append((c, degree_pc_iter)) 
                    self.samples_solution[i].append((c.T @ self.phi(*self.seed_rv_samples)))
                    degree_pc_iter += 1
                    counter = 0 
                    coeff_init = c.copy()
                    self.phi, self.phi_norm = cp.generate_expansion(degree_pc_iter, self.seed_rv, retall=True)
                    self.n_pc = len(self.phi_norm)
                
                if counter > 50: 
                    print(f"Stopping branch {i}: reached max attempts.")
                    break
            print(f"Branch sequence found for branch {i}")

    def plot_xy_mu(self, n_branch):
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].set_xlabel(r"$\mu$")
        ax[1].set_xlabel(r"$\mu$")
        ax[0].set_ylabel(r"$x$")
        ax[1].set_ylabel(r"$y$")
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        mu_grid = xi_grid * cp.Std(self.mu) + cp.E(self.mu)
        
        # --- Compute the Exact Branches ---
        sym_x = np.zeros_like(mu_grid)
        for idx, m in enumerate(mu_grid):
            # Symmetric root: x^3 + x - mu = 0
            roots = np.roots([1, 0, 1, -m])
            sym_x[idx] = np.real(roots[np.isreal(roots)][0])
            
        ax[0].plot(mu_grid, sym_x, 'k', linewidth=4.0, zorder=1, label=r'Exact $\bar{u}$')
        ax[1].plot(mu_grid, sym_x, 'k', linewidth=4.0, zorder=1)
        
        # Asymmetric branches (exist for mu >= 2)
        mu_valid = mu_grid[mu_grid >= 2.0]
        if len(mu_valid) > 0:
            asym_x1 = (mu_valid + np.sqrt(mu_valid**2 - 4)) / 2
            asym_x2 = (mu_valid - np.sqrt(mu_valid**2 - 4)) / 2
            
            ax[0].plot(mu_valid, asym_x1, 'k', linewidth=4.0, zorder=1)
            ax[0].plot(mu_valid, asym_x2, 'k', linewidth=4.0, zorder=1)
            ax[1].plot(mu_valid, asym_x2, 'k', linewidth=4.0, zorder=1) # y is flipped
            ax[1].plot(mu_valid, asym_x1, 'k', linewidth=4.0, zorder=1)
        
        # --- Plot the PCE approximations ---
        max_deg = max([deg for branch in self.solution for (_, deg) in branch]) if self.solution[0] else 0
        colors = plt.cm.coolwarm(np.linspace(0, 1, max_deg + 1))
        
        for i in range(min(n_branch, len(self.solution))):
            for j in range(len(self.solution[i])):
                coeffs, deg = self.solution[i][j]
                
                phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
                approx = coeffs.T @ phi_eval # Shape: (2, 500)
                
                if deg == max_deg:
                    label = rf'PCE $u_{{{deg}}}$' if i == 0 else None
                    ax[0].plot(mu_grid, approx[0], color=colors[deg], linewidth=2.0, zorder=5, linestyle='--',
                     marker='o', markersize=6, markevery=30, label=label)
                    ax[1].plot(mu_grid, approx[1], color=colors[deg], linewidth=2.0, zorder=5, linestyle='--',
                     marker='o', markersize=6, markevery=30)
                else:
                    ax[0].plot(mu_grid, approx[0], color=colors[deg], linewidth=1, linestyle='--', alpha=0.6, zorder=2)
                    ax[1].plot(mu_grid, approx[1], color=colors[deg], linewidth=1, linestyle='--', alpha=0.6, zorder=2)

        for i in range(2):
            ax[i].grid(True, alpha=0.3)
            
        ax[0].legend()
        fig.tight_layout()
        fig.savefig(f"plots/toggle_poly_({self.mu.lower.item():.3g},{self.mu.upper.item():.3g})_{degree_pc=}.png", bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    
    degree_pc = 10
    n_branch_to_approximate = 3
    
    # We span across mu=2 to capture the pitchfork bifurcation
    model = ToggleSwitch(
        mu=cp.Uniform(1.7, 3),
        seed_rv=cp.J(cp.Uniform(-np.sqrt(3), np.sqrt(3))), # 1D chaos
        n_samples=1000
    )

    print("\n=== Executing Degree Continuation ===")
    model.continuation(degree_pc=degree_pc, n_branch=n_branch_to_approximate)
    model.plot_xy_mu(n_branch=n_branch_to_approximate)

    model = ToggleSwitch(
        mu=cp.Uniform(-4, 2),
        seed_rv=cp.J(cp.Uniform(-np.sqrt(3), np.sqrt(3))), # 1D chaos
        n_samples=1000
    )

    print("\n=== Executing Degree Continuation ===")
    model.continuation(degree_pc=degree_pc, n_branch=n_branch_to_approximate)
    model.plot_xy_mu(n_branch=n_branch_to_approximate)