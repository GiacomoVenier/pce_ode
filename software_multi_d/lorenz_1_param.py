import chaospy as cp
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

os.makedirs("plots", exist_ok=True)

np.random.seed(241)

# Colours
bluemathlab = "#065895"
orangemathlab = "#f79a25"
greenmathlab = "#77ac30"

# Matplotlib options (Nuovo Standard)
plt.rcParams.update({
    "font.size": 10,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}",
    "font.family": "serif",
    'legend.fontsize': 'x-large',
    'axes.labelsize': 'x-large',
    'axes.titlesize': 'xx-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large',
    'lines.linewidth': 3,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

class Lorenz():
    def __init__(self, gamma, rho, theta, seed_rv, n_samples=10000):
        '''
        gamma --> {float}
        rho --> {chaospy.Distribution}
        theta --> {float}
        seed_rv --> {chaospy.Distribution} (1D)
        n_samples --> {int}
        '''
        self.gamma = gamma
        self.rho = rho
        self.theta = theta
        self.seed_rv = seed_rv
        self.n_kl = len(self.seed_rv)
        self.n_samples = n_samples

        # 1D chaos for rho
        self.seed_rv_samples = np.atleast_2d(self.seed_rv.sample(n_samples)) # (1, self.n_samples)
        self.rho_samples = self.seed_rv_samples[0] * cp.Std(self.rho) + cp.E(self.rho) # (self.n_samples)

    def f(self, c):
        c = c.reshape(3, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) # (self.n_pc, self.n_samples)
        x, y, z = c.T @ phi # (3, self.n_samples)
        
        x_dot = ((self.gamma * (y - x)) @ phi.T) / self.n_samples 
        y_dot = ((self.rho_samples * x - y - x * z) @ phi.T) / self.n_samples 
        z_dot = ((x * y - self.theta * z) @ phi.T) / self.n_samples 
        
        return np.concatenate([x_dot, y_dot, z_dot])

    def jacobian(self, c):
        c = c.reshape(3, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) 
        x, y, z = c.T @ phi

        Jxx = (-(self.gamma * phi) @ phi.T) / self.n_samples
        Jxy = ((self.gamma * phi) @ phi.T) / self.n_samples
        Jxz = np.zeros((self.n_pc, self.n_pc))

        Jyx = (((self.rho_samples - z) * phi) @ phi.T) / self.n_samples
        Jyy = -phi @ phi.T / self.n_samples
        Jyz = (-(x * phi) @ phi.T) / self.n_samples

        Jzx = ((y * phi) @ phi.T) / self.n_samples
        Jzy = ((x * phi) @ phi.T) / self.n_samples
        Jzz = ((-self.theta * phi) @ phi.T) / self.n_samples
        
        J = np.block([
                [Jxx, Jxy, Jxz],
                [Jyx, Jyy, Jyz],
                [Jzx, Jzy, Jzz]
            ])
        return J

    def run(self, degree_pc, n_init):
        self.phi, self.phi_norm = cp.generate_expansion(degree_pc, self.seed_rv, retall=True)
        self.n_pc = len(self.phi_norm)
        self.solution = np.zeros((n_init, self.n_pc, 3))
        self.coeff_init = []
        count = 0
        
        while count < n_init:
            self.coeff_init.append(5.0 * np.random.randn(self.n_pc, 3))
            sol = root(self.f, self.coeff_init[-1].ravel(), method='lm', tol=1e-8, jac=self.jacobian)
            loss = np.sum(np.abs(self.f(sol.x)))
            
            if (loss < 1e-6) and (not np.isclose(self.solution[:count], sol.x.reshape(3, self.n_pc).T).all(2).all(1).any()):
                self.solution[count] = sol.x.reshape(3, self.n_pc).T
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
                    coeff_init = 50.0 * np.random.randn(1, 3)
                else:
                    tmp = np.zeros((self.n_pc, 3))
                    tmp[:len(coeff_init)] = coeff_init
                    coeff_init = tmp.T
                
                sol = root(self.f, coeff_init.ravel(), method='lm', tol=1e-8, jac=self.jacobian)
                loss = np.sum(np.abs(self.f(sol.x)))
                c = sol.x.reshape(3, self.n_pc).T
                
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

    def plot_xyz_rho_2(self, n_branch):
        """Plot 2D aggiornato agli standard del Toggle Switch"""
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax[0].set_xlabel(r"$\rho$"); ax[0].set_ylabel(r"$x$")
        ax[1].set_xlabel(r"$\rho$"); ax[1].set_ylabel(r"$y$")
        ax[2].set_xlabel(r"$\rho$"); ax[2].set_ylabel(r"$z$")
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        rho_grid = xi_grid * cp.Std(self.rho) + cp.E(self.rho)
        theta_mean = self.theta 
        
        for i in range(3):
            ax[i].plot(rho_grid, np.zeros_like(rho_grid), 'k', zorder=1, label=r'$\bar{u}$' if i==0 else None)
            
        rho_valid = rho_grid[rho_grid >= 1.0]
        if len(rho_valid) > 0:
            x_branch = np.sqrt(theta_mean * (rho_valid - 1.0))
            z_branch = rho_valid - 1.0
            
            ax[0].plot(rho_valid, x_branch, 'k', zorder=1)
            ax[0].plot(rho_valid, -x_branch, 'k', zorder=1)
            ax[1].plot(rho_valid, x_branch, 'k', zorder=1) 
            ax[1].plot(rho_valid, -x_branch, 'k', zorder=1)
            ax[2].plot(rho_valid, z_branch, 'k', zorder=1)
        
        max_deg = max([deg for branch in self.solution for (_, deg) in branch]) if self.solution[0] else 0
        branch_colors = ["#f79a25", "#065895", "#77ac30", "#d9534f"]
        
        for i in range(min(n_branch, len(self.solution))):
            for j in range(len(self.solution[i])):
                coeffs, deg = self.solution[i][j]
                
                if deg == max_deg:
                    phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
                    approx = coeffs.T @ phi_eval 
                    b_color = branch_colors[i % len(branch_colors)]
                    label = rf'$N={{{deg}}}$' 
                    
                    for k in range(3):
                        ax[k].plot(rho_grid, approx[k], color=b_color,  zorder=5, linestyle='--',
                                   marker='o', markersize=5, markevery=40, label=label if k==0 else None)

        for i in range(3):
            ax[i].grid(True, alpha=0.3)
            ax[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax[i].set_xlim([np.min(rho_grid), np.max(rho_grid)])
            
        ax[0].legend(loc="upper left", borderpad=0.2, labelspacing=0.2, handlelength=1.5)
        fig.tight_layout()
        fig.savefig(f"plots/Lorenz_2d_approx_N_{max_deg}_rho_{self.rho}.pdf", bbox_inches="tight")
        plt.show()

    def plot_3d_xy_rho(self, n_branch):
        """Plot 3D nello spazio (x, y, rho)"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.zaxis.set_rotate_label(False) 
        ax.set_zlabel(r"$\rho$", rotation=90, labelpad=15)
        
        ax.tick_params(axis='x', pad=-3)
        ax.tick_params(axis='y', pad=-1)
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        rho_grid = xi_grid * cp.Std(self.rho) + cp.E(self.rho)
        theta_mean = self.theta
        
        # --- Soluzioni Esatte ---
        ax.plot(np.zeros_like(rho_grid), np.zeros_like(rho_grid), rho_grid, 'k', zorder=1, label=r'$\bar{u}$')
        
        rho_valid = rho_grid[rho_grid >= 1.0]
        if len(rho_valid) > 0:
            x_branch = np.sqrt(theta_mean * (rho_valid - 1.0))
            ax.plot(x_branch, x_branch, rho_valid, 'k', zorder=1)
            ax.plot(-x_branch, -x_branch, rho_valid, 'k', zorder=1)

        # --- Soluzioni Approssimate PCE ---
        max_deg = max([deg for branch in self.solution for (_, deg) in branch]) if self.solution[0] else 0
        branch_colors = ["#f79a25", "#065895", "#77ac30", "#d9534f"]
        
        for i in range(min(n_branch, len(self.solution))):
            for j in range(len(self.solution[i])):
                coeffs, deg = self.solution[i][j]
                if deg == max_deg:
                    phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
                    approx = coeffs.T @ phi_eval 
                    b_color = branch_colors[i % len(branch_colors)]
                    
                    ax.plot(approx[0], approx[1], rho_grid, color=b_color,
                            marker='o', markersize=5, markevery=40, linestyle='--', zorder=5)

        ax.view_init(elev=17, azim=-110)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
        
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect(None, zoom=0.90)
        else:
            ax.dist = 11
                
        # Aumentato leggermente pad_inches a 0.15 per garantire che la \rho non venga sfiorata dal taglio
        fig.savefig(f"plots/Lorenz_3d_xy_rho_N_{max_deg}_rho_{self.rho}.pdf", bbox_inches='tight', pad_inches=0.1)
        plt.show()

    def plot_3d_xz_rho(self, n_branch):
        """Plot 3D nello spazio (x, z, rho)"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$z$")
        ax.zaxis.set_rotate_label(False) 
        ax.set_zlabel(r"$\rho$", rotation=90, labelpad=15)
        
        ax.tick_params(axis='x', pad=-3)
        ax.tick_params(axis='y', pad=-1)
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        rho_grid = xi_grid * cp.Std(self.rho) + cp.E(self.rho)
        theta_mean = self.theta
        
        # --- Soluzioni Esatte ---
        ax.plot(np.zeros_like(rho_grid), np.zeros_like(rho_grid), rho_grid, 'k', zorder=1, label=r'$\bar{u}$')
        
        rho_valid = rho_grid[rho_grid >= 1.0]
        if len(rho_valid) > 0:
            x_branch = np.sqrt(theta_mean * (rho_valid - 1.0))
            z_branch = rho_valid - 1.0
            # Nel piano xz, i due rami hanno la stessa z
            ax.plot(x_branch, z_branch, rho_valid, 'k', zorder=1)
            ax.plot(-x_branch, z_branch, rho_valid, 'k', zorder=1)

        # --- Soluzioni Approssimate PCE ---
        max_deg = max([deg for branch in self.solution for (_, deg) in branch]) if self.solution[0] else 0
        branch_colors = ["#f79a25", "#065895", "#77ac30", "#d9534f"]
        
        for i in range(min(n_branch, len(self.solution))):
            for j in range(len(self.solution[i])):
                coeffs, deg = self.solution[i][j]
                if deg == max_deg:
                    phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
                    approx = coeffs.T @ phi_eval 
                    b_color = branch_colors[i % len(branch_colors)]
                    
                    # Plot di x (approx[0]) e z (approx[2]) contro rho
                    ax.plot(approx[0], approx[2], rho_grid, color=b_color,
                            marker='o', markersize=5, markevery=40, linestyle='--', zorder=5)

        
        ax.view_init(elev=17, azim=-110)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
        
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect(None, zoom=0.90)
        else:
            ax.dist = 11
        
        fig.savefig(f"plots/Lorenz_3d_xz_rho_N_{max_deg}_rho_{self.rho}.pdf", bbox_inches='tight', pad_inches=0.1)
        plt.show()
        

if __name__ == "__main__":
    
    degree_pc = 20
    n_init = 1
    n_branch_to_approximate = 3

    model = Lorenz(
        gamma=10.0, 
        rho=cp.Uniform(1.0, 2.0),
        theta=8.0/3.0,
        seed_rv=cp.J(cp.Uniform(-np.sqrt(3), np.sqrt(3))), # 1D chaos
        n_samples=1000
    )

    RUN_RANDOM_INIT = False 
    RUN_CONTINUATION = True

    if RUN_RANDOM_INIT:
        print("\n=== Executing Random Initialization ===")
        model.run(degree_pc=degree_pc, n_init=n_init)
        run_results_formatted = [[(sol, degree_pc)] for sol in model.solution]
        model.solution = run_results_formatted
        model.plot_xyz_rho_2(n_branch=n_init) 

    if RUN_CONTINUATION:
        print("\n=== Executing Degree Continuation ===")
        model.continuation(degree_pc=degree_pc, n_branch=n_branch_to_approximate)
        model.plot_xyz_rho_2(n_branch=n_branch_to_approximate)
        model.plot_3d_xy_rho(n_branch=n_branch_to_approximate)
        model.plot_3d_xz_rho(n_branch=n_branch_to_approximate)
    
    degree_pc = 20
    n_init = 1
    n_branch_to_approximate = 3

    model = Lorenz(
        gamma=10.0, 
        rho=cp.Uniform(0.0, 1.0),
        theta=8.0/3.0,
        seed_rv=cp.J(cp.Uniform(-np.sqrt(3), np.sqrt(3))), # 1D chaos
        n_samples=1000
    )

    RUN_RANDOM_INIT = False 
    RUN_CONTINUATION = True

    if RUN_RANDOM_INIT:
        print("\n=== Executing Random Initialization ===")
        model.run(degree_pc=degree_pc, n_init=n_init)
        run_results_formatted = [[(sol, degree_pc)] for sol in model.solution]
        model.solution = run_results_formatted
        model.plot_xyz_rho_2(n_branch=n_init) 

    if RUN_CONTINUATION:
        print("\n=== Executing Degree Continuation ===")
        model.continuation(degree_pc=degree_pc, n_branch=n_branch_to_approximate)
        model.plot_xyz_rho_2(n_branch=n_branch_to_approximate)
        model.plot_3d_xy_rho(n_branch=n_branch_to_approximate)
        model.plot_3d_xz_rho(n_branch=n_branch_to_approximate)