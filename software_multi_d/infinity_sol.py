import chaospy as cp
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Necessario per proiezioni 3D in alcune versioni di matplotlib

np.random.seed(241)

# matplotlib.pyplot options
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
})

plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'lmodern'

class BifurcationSystem():
    def __init__(self, eps, seed_rv, n_samples=10000):
        self.eps = eps
        self.seed_rv = seed_rv
        self.n_kl = len(self.seed_rv)
        self.n_samples = n_samples

        self.seed_rv_samples = np.atleast_2d(self.seed_rv.sample(n_samples))
        self.eps_samples = self.seed_rv_samples[0] * cp.Std(self.eps) + cp.E(self.eps)

    def f(self, c):
        c = c.reshape(2, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) 
        x, y = c.T @ phi 
        
        # Campo vettoriale x*(1-x^2-y^2)+eps , y*(1-x^2-y^2)
        x_dot = ((x * (1 - x**2 - y**2) + self.eps_samples) @ phi.T) / self.n_samples 
        y_dot = ((y * (1 - x**2 - y**2)) @ phi.T) / self.n_samples 
        
        return np.concatenate([x_dot, y_dot])

    def jacobian(self, c):
        c = c.reshape(2, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) 
        x, y = c.T @ phi

        Jxx = (((1 - 3*x**2 - y**2) * phi) @ phi.T) / self.n_samples
        Jxy = (((-2*x*y) * phi) @ phi.T) / self.n_samples
        Jyx = (((-2*x*y) * phi) @ phi.T) / self.n_samples
        Jyy = (((1 - x**2 - 3*y**2) * phi) @ phi.T) / self.n_samples
        
        J = np.block([
                [Jxx, Jxy],
                [Jyx, Jyy]
            ])
        return J

    def run(self, degree_pc, n_init):
        self.phi, self.phi_norm = cp.generate_expansion(degree_pc, self.seed_rv, retall=True)
        self.n_pc = len(self.phi_norm)
        self.solution = np.zeros((n_init, self.n_pc, 2))
        self.coeff_init = []
        count = 0
        
        while count < n_init:
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
                    current_guess = 2.0 * np.random.randn(2)
                else:
                    new_guess = np.zeros((2, self.n_pc))
                    old_n_pc = c.shape[0]
                    new_guess[0, :old_n_pc] = c[:, 0]
                    new_guess[1, :old_n_pc] = c[:, 1]
                    current_guess = new_guess.ravel()

                sol = root(self.f, current_guess, method='lm', tol=1e-8, jac=self.jacobian)
                loss = np.sum(np.abs(self.f(sol.x)))
                
                c = sol.x.reshape(2, self.n_pc).T 
                
                control = not any(np.isclose(c[0, 0], self.solution[j][0][0][0,0], atol=1e-2) 
                                 for j in range(i) if self.solution[j]) if (degree_pc_iter == 0 and i > 0) else True
                
                if (loss < 1e-6) and control:
                    self.solution[i].append((c, degree_pc_iter)) 
                    self.samples_solution[i].append((c.T @ self.phi(*self.seed_rv_samples)))
                    
                    degree_pc_iter += 1
                    counter = 0 
                    self.phi, self.phi_norm = cp.generate_expansion(degree_pc_iter, self.seed_rv, retall=True)
                    self.n_pc = len(self.phi_norm)
                
                if counter > 200: 
                    print(f"Stopping branch {i}: reached max attempts.")
                    break
            print(f"Branch sequence found for branch {i}")

    def plot_xy_eps(self, n_branch):
        fig, ax = plt.subplots(1, 2, figsize=(20, 7))
        ax[0].set_xlabel(r"$\epsilon$")
        ax[1].set_xlabel(r"$\epsilon$")
        ax[0].set_ylabel(r"$x$")
        ax[1].set_ylabel(r"$y$")
        
        x_exact = np.linspace(-2.0, 2.0, 1000)
        eps_exact = x_exact**3 - x_exact
        
        ax[0].plot(eps_exact, x_exact, 'k', linewidth=4.0, zorder=1, label='Sol. Esatta')
        ax[1].plot(eps_exact, np.zeros_like(x_exact), 'k', linewidth=4.0, zorder=1)
        
        theta = np.linspace(0, 2*np.pi, 500)
        x_circ = np.cos(theta)
        y_circ = np.sin(theta)
        ax[0].plot(np.zeros_like(x_circ), x_circ, 'k', linewidth=4.0, zorder=1)
        ax[1].plot(np.zeros_like(y_circ), y_circ, 'k', linewidth=4.0, zorder=1)
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        eps_grid = xi_grid * cp.Std(self.eps) + cp.E(self.eps)
        
        max_deg = max([deg for branch in self.solution for (_, deg) in branch]) if self.solution[0] else 0
        branch_colors = ["#065895", "#f79a25", "#77ac30"]
        
        for i in range(min(n_branch, len(self.solution))):
            for j in range(len(self.solution[i])):
                coeffs, deg = self.solution[i][j]
                if deg == max_deg:
                    phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
                    approx = coeffs.T @ phi_eval
                    
                    b_color = branch_colors[i % len(branch_colors)]
                    label = rf'$u_{{{deg}}}$ Branch {i}' 
                    
                    ax[0].plot(eps_grid, approx[0], color=b_color, linewidth=2.0, zorder=5, linestyle='--', marker='o', markersize=6, markevery=30, label=label)
                    ax[1].plot(eps_grid, approx[1], color=b_color, linewidth=2.0, zorder=5, linestyle='--', marker='o', markersize=6, markevery=30)

        for i in range(2):
            ax[i].grid(True, alpha=0.3)
            ax[i].set_xlim([np.min(eps_grid), np.max(eps_grid)])
            
        fig.tight_layout()
        plt.show()

    def plot_3d_bifurcation(self, n_branch):
        """Nuovo metodo per plottare in 3D sia le soluzioni esatte che quelle approssimate"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$\epsilon$")
        ax.set_title("Diagramma di Biforcazione 3D e Approssimazioni PCE")
        
        # 1. Soluzioni Esatte
        # Curva cubica nel piano y=0
        x_exact = np.linspace(-1.5, 1.5, 500)
        eps_exact = x_exact**3 - x_exact
        ax.plot(x_exact, np.zeros_like(x_exact), eps_exact, 'k', linewidth=4.0, zorder=1, label='Ramo y=0 (Esatto)')
        
        # Cerchio unitario nel piano eps=0
        theta = np.linspace(0, 2*np.pi, 500)
        x_circ = np.cos(theta)
        y_circ = np.sin(theta)
        ax.plot(x_circ, y_circ, np.zeros_like(x_circ), 'k', linewidth=4.0, zorder=1, label='Anello Degenere (Esatto)')
        
        # 2. Soluzioni Approssimate (PCE)
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        eps_grid = xi_grid * cp.Std(self.eps) + cp.E(self.eps)
        
        max_deg = max([deg for branch in self.solution for (_, deg) in branch]) if self.solution[0] else 0
        branch_colors = ["#065895", "#f79a25", "#77ac30"]
        
        for i in range(min(n_branch, len(self.solution))):
            for j in range(len(self.solution[i])):
                coeffs, deg = self.solution[i][j]
                if deg == max_deg:
                    phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
                    approx = coeffs.T @ phi_eval # Shape: (2, 500)
                    
                    b_color = branch_colors[i % len(branch_colors)]
                    label = rf'$u_{{{deg}}}$ Branch {i}' 
                    
                    # Plot in 3D: x=approx[0], y=approx[1], z=eps_grid
                    ax.plot(approx[0], approx[1], eps_grid, color=b_color, linewidth=2.5, 
                            linestyle='--', marker='o', markersize=6, markevery=30, label=label, zorder=5)

        # Aggiusta la visuale iniziale per mostrare bene l'intersezione
        ax.view_init(elev=20, azim=-60)
        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    degree_pc = 20
    n_branch_to_approximate = 3
    
    model = BifurcationSystem(
        eps=cp.Uniform(0.3, 1),
        seed_rv=cp.J(cp.Uniform(-np.sqrt(3), np.sqrt(3))), # 1D chaos
        n_samples=1000
    )

    print("\n=== Executing Degree Continuation ===")
    model.continuation(degree_pc=degree_pc, n_branch=n_branch_to_approximate)
    
    # Plot 2D standard
    model.plot_xy_eps(n_branch=n_branch_to_approximate)
    
    # Nuovo Plot 3D interattivo
    model.plot_3d_bifurcation(n_branch=n_branch_to_approximate)