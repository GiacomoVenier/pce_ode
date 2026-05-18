import chaospy as cp
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(241)

plt.rcParams.update({
    "font.size": 10,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern}",
    "font.family": "serif",

    'legend.fontsize': 'x-large',
    'axes.labelsize': 'x-large',
    'axes.titlesize':'xx-large',
    'xtick.labelsize':'x-large',
    'ytick.labelsize':'x-large',
    'lines.linewidth': 3,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


class ToggleSwitch():
    def __init__(self, mu, seed_rv, n_samples=10000):
        self.mu = mu
        self.seed_rv = seed_rv
        self.n_kl = len(self.seed_rv)
        self.n_samples = n_samples

        self.seed_rv_samples = np.atleast_2d(self.seed_rv.sample(n_samples)) 
        self.mu_samples = self.seed_rv_samples[0] * cp.Std(self.mu) + cp.E(self.mu) 

    def f(self, c):
        c = c.reshape(2, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) 
        x, y = c.T @ phi 
        
        x_dot = ((-x + self.mu_samples / (1 + y**2)) @ phi.T) / self.n_samples 
        y_dot = ((-y + self.mu_samples / (1 + x**2)) @ phi.T) / self.n_samples 
        
        return np.concatenate([x_dot, y_dot])

    def jacobian(self, c):
        c = c.reshape(2, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) 
        x, y = c.T @ phi

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

    def get_exact_branches(self, mu_range):
        """Calcola dinamicamente tutti i rami esatti (simmetrici e asimmetrici) per un dato intervallo di mu."""
        mu_exact = np.linspace(np.min(mu_range) - 0.5, np.max(mu_range) + 0.5, 2000)
        
        # Ramo 1: Soluzione simmetrica (x = y) => x^3 + x - mu = 0
        sym_x = np.zeros_like(mu_exact)
        for idx, m in enumerate(mu_exact):
            roots = np.roots([1, 0, 1, -m])
            sym_x[idx] = np.real(roots[np.isreal(roots)][0])
            
        # Rami 2 e 3: Inizializziamo tutto con NaN per "spezzare" le linee
        asym_x1 = np.full_like(mu_exact, np.nan)
        asym_y1 = np.full_like(mu_exact, np.nan)
        asym_x2 = np.full_like(mu_exact, np.nan)
        asym_y2 = np.full_like(mu_exact, np.nan)
        
        # Maschera: valide solo per |mu| >= 2
        mask_valid = np.abs(mu_exact) >= 2.0
        
        if np.any(mask_valid):
            mu_valid = mu_exact[mask_valid]
            asym_x1[mask_valid] = (mu_valid + np.sqrt(mu_valid**2 - 4)) / 2
            asym_y1[mask_valid] = (mu_valid - np.sqrt(mu_valid**2 - 4)) / 2
            
            asym_x2[mask_valid] = (mu_valid - np.sqrt(mu_valid**2 - 4)) / 2
            asym_y2[mask_valid] = (mu_valid + np.sqrt(mu_valid**2 - 4)) / 2
            
        return mu_exact, sym_x, asym_x1, asym_y1, asym_x2, asym_y2

    def plot_xy_mu(self, n_branch):
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        mu_grid = xi_grid * cp.Std(self.mu) + cp.E(self.mu)
        
        # --- Soluzioni Esatte Dinamiche ---
        # Nota: mu_asym non serve più, usiamo mu_exact per tutto
        mu_exact, sym_x, asym_x1, asym_y1, asym_x2, asym_y2 = self.get_exact_branches(mu_grid)
        
        # ---------------------------------------------------------
        # Figura 1: Approssimazione per la variabile x
        # ---------------------------------------------------------
        fig1, ax1 = plt.subplots()
        ax1.set_xlabel(r"$\mu$")
        ax1.set_ylabel(r"$x$")
        
        # Ramo esatto principale
        ax1.plot(mu_exact, sym_x, 'k', zorder=1, label=r'$\bar{u}$')
        
        # Disegna rami asimmetrici (i NaN spezzeranno automaticamente la linea)
        if not np.all(np.isnan(asym_x1)):
            ax1.plot(mu_exact, asym_x1, 'k', zorder=1)
            ax1.plot(mu_exact, asym_x2, 'k', zorder=1)
        
        # ---------------------------------------------------------
        # Figura 2: Approssimazione per la variabile y
        # ---------------------------------------------------------
        fig2, ax2 = plt.subplots()
        ax2.set_xlabel(r"$\mu$")
        ax2.set_ylabel(r"$y$")
        
        # Ramo esatto principale
        ax2.plot(mu_exact, sym_x, 'k', zorder=1, label=r'$\bar{u}$')
        
        # Disegna rami asimmetrici (i NaN spezzeranno automaticamente la linea)
        if not np.all(np.isnan(asym_y1)):
            ax2.plot(mu_exact, asym_y1, 'k', zorder=1)
            ax2.plot(mu_exact, asym_y2, 'k', zorder=1)
        
        # --- Soluzioni Approssimate PCE ---
        max_deg = max([deg for branch in self.solution for (_, deg) in branch]) if self.solution[0] else 0
        branch_colors = ["#f79a25", "#065895", "#77ac30", "#d9534f"]

        
        for i in range(min(n_branch, len(self.solution))):
            for j in range(len(self.solution[i])):
                coeffs, deg = self.solution[i][j]
                
                phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
                approx = coeffs.T @ phi_eval 
                
                if deg == max_deg:
                    b_color = branch_colors[i % len(branch_colors)]
                    label = rf'$N={{{deg}}}$' 
                    
                    # Plot x
                    ax1.plot(mu_grid, approx[0], color=b_color, zorder=5, 
                             marker='o', markersize=5, markevery=40, linestyle='--', alpha=1, label=label)
                    # Plot y
                    ax2.plot(mu_grid, approx[1], color=b_color, zorder=5, 
                             marker='o', markersize=5, markevery=40, linestyle='--', alpha=1, label=label)

        all_x_vals = []
        all_y_vals = []
        for i in range(len(self.solution)):
            for j in range(len(self.solution[i])):
                coeffs, deg = self.solution[i][j]
                if deg == max_deg:
                    phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
                    approx = coeffs.T @ phi_eval
                    all_x_vals.extend(approx[0])
                    all_y_vals.extend(approx[1])

        # Imposta i limiti con un margine del 5% per non toccare i bordi
        for ax, data in zip([ax1, ax2], [all_x_vals, all_y_vals]):
            ax.grid(True, alpha=0.3)
            ax.set_xlim([np.min(mu_grid), np.max(mu_grid)])
            
            if data:
                min_v, max_v = np.min(data), np.max(data)
                margin = (max_v - min_v) * 0.05
                ax.set_ylim([min_v - margin, max_v + margin])
            
            ax.legend(loc="upper left", borderpad=0.2, labelspacing=0.2, handlelength=1.5)
            
        fig1.tight_layout()
        fig1.savefig(f"plots/Genetic_Toggle_Switch_x_mu_N_{degree_pc}_mu_{self.mu}.pdf", bbox_inches='tight')

        fig2.tight_layout()
        fig2.savefig(f"plots/Genetic_Toggle_Switch_y_mu_N_{degree_pc}_mu_{self.mu}.pdf", bbox_inches='tight')
        plt.show()

    def plot_3d_bifurcation(self, n_branch):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.zaxis.set_rotate_label(False) 
        ax.set_zlabel(r"$\mu$", rotation=90, labelpad=15)
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        mu_grid = xi_grid * cp.Std(self.mu) + cp.E(self.mu)
        
        # --- Soluzioni Esatte Dinamiche ---
        # Riceviamo gli array completi con i NaN già inseriti nella zona vuota
        mu_exact, sym_x, asym_x1, asym_y1, asym_x2, asym_y2 = self.get_exact_branches(mu_grid)
        
        # Ramo principale simmetrico (x = y)
        ax.plot(sym_x, sym_x, mu_exact, 'k', zorder=1, label=r'$\bar{u}$')
        
        # Rami asimmetrici (i NaN spezzeranno automaticamente le linee in 3D)
        if not np.all(np.isnan(asym_x1)):
            ax.plot(asym_x1, asym_y1, mu_exact, 'k', zorder=1)
            ax.plot(asym_x2, asym_y2, mu_exact, 'k', zorder=1)

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
                    label = rf'$N={{{deg}}}$' 
                    
                    # Uniformato stile: linea 2.5, marker ogni 40 punti
                    ax.plot(approx[0], approx[1], mu_grid, color=b_color, 
                             marker='o', markersize=5, markevery=40, linestyle='--', alpha=1, label=label, zorder=5)

        ax.view_init(elev=17, azim=-110)
        
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect(None, zoom=0.90)
        else:
            ax.dist = 11
        
        # Salva "ritagliando" il grafico, con un padding minimo per non tagliare le etichette
        fig.savefig(f"plots/Genetic_Toggle_Switch_3d_N_{degree_pc}_mu_{self.mu}.pdf", bbox_inches='tight', pad_inches=0.1)
        plt.show()

if __name__ == "__main__":
    
    degree_pc = 20
    n_branch_to_approximate = 3

    # Utilizziamo un dominio che attraversa il punto di biforcazione (mu >= 2) 
    # in modo da verificare che i 3 rami appaiano dove teoricamente previsti.
    model = ToggleSwitch(
        mu=cp.Uniform(-6,15), 
        seed_rv=cp.J(cp.Uniform(-np.sqrt(3), np.sqrt(3))), # 1D chaos
        n_samples=1000
    )

    print("\n=== Executing Degree Continuation ===")
    model.continuation(degree_pc=degree_pc, n_branch=n_branch_to_approximate)
    model.plot_xy_mu(n_branch=n_branch_to_approximate)
    model.plot_3d_bifurcation(n_branch=n_branch_to_approximate)
    
    degree_pc = 20
    n_branch_to_approximate = 3

    # Utilizziamo un dominio che attraversa il punto di biforcazione (mu >= 2) 
    # in modo da verificare che i 3 rami appaiano dove teoricamente previsti.
    model = ToggleSwitch(
        mu=cp.Uniform(-2,2), 
        seed_rv=cp.J(cp.Uniform(-np.sqrt(3), np.sqrt(3))), # 1D chaos
        n_samples=1000
    )

    print("\n=== Executing Degree Continuation ===")
    model.continuation(degree_pc=degree_pc, n_branch=n_branch_to_approximate)
    model.plot_xy_mu(n_branch=n_branch_to_approximate)
    model.plot_3d_bifurcation(n_branch=n_branch_to_approximate)