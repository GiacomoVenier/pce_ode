import chaospy as cp
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import os
from scipy.optimize import least_squares

os.makedirs("plots", exist_ok=True)
np.random.seed(42)

# Nuove impostazioni grafiche standardizzate
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

class RingBifurcationUnified():
    def __init__(self, eps, seed_rv, n_samples=10000):
        self.eps = eps
        self.seed_rv = seed_rv
        self.n_kl = len(self.seed_rv)
        self.n_samples = n_samples

        self.seed_rv_samples = np.atleast_2d(self.seed_rv.sample(n_samples)) 
        self.eps_samples = self.seed_rv_samples[0] * cp.Std(self.eps) + cp.E(self.eps) 
        
        # Inizializza un array vuoto per contenere le soluzioni, indipendentemente dal metodo usato
        self.solution = []

    def update_stochastic_amplitude(self, new_std):
        """Modifica dinamicamente la deviazione standard per l'omotopia (Fase 1 e 2)."""
        self.eps_samples = self.seed_rv_samples[0] * new_std + cp.E(self.eps)

    def f(self, c):
        c = c.reshape(2, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) 
        x, y = c.T @ phi 
        
        x_dot = (((x - self.eps_samples**3) * (1 - x**2 - y**2 - self.eps_samples**2)) @ phi.T) / self.n_samples 
        y_dot = ((y * (1 - x**2 - y**2 - self.eps_samples**2)) @ phi.T) / self.n_samples 
        
        return np.concatenate([x_dot, y_dot])

    def jacobian(self, c):
        c = c.reshape(2, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) 
        x, y = c.T @ phi

        Jxx = (((1 - 3*x**2 - y**2 - self.eps_samples**2 + 2*x*self.eps_samples**3) * phi) @ phi.T) / self.n_samples
        Jxy = (((-2*x*y + 2*y*self.eps_samples**3) * phi) @ phi.T) / self.n_samples
        Jyx = (((-2*x*y) * phi) @ phi.T) / self.n_samples
        Jyy = (((1 - x**2 - 3*y**2 - self.eps_samples**2) * phi) @ phi.T) / self.n_samples
        
        J = np.block([
                [Jxx, Jxy],
                [Jyx, Jyy]
            ])
        return J

    def continuation(self, degree_pc, n_branch, start_eps_max=0.1):
        start_std = start_eps_max / np.sqrt(3)
        target_std = float(cp.Std(self.eps))
        target_eps_max = target_std * np.sqrt(3)
        
        if target_std <= start_std + 1e-6:
            start_std = target_std
            
        n_eq = 2
        self.solution = []

        print(f"\n--- FASE 1: Ricerca di {n_branch} branch unici a N=0 (eps limitato a {start_eps_max:.2f}) ---")
        self.update_stochastic_amplitude(start_std)
        self.phi, self.phi_norm = cp.generate_expansion(0, self.seed_rv, retall=True)
        self.n_pc = len(self.phi_norm)
        
        seeds = []
        attempts = 0
        max_attempts = n_branch * 300
        
        while len(seeds) < n_branch and attempts < max_attempts:
            attempts += 1
            guess = 3.0 * np.random.randn(n_eq, self.n_pc)
            sol = root(self.f, guess.ravel(), method='lm', tol=1e-9, jac=self.jacobian)
            loss = np.sum(np.abs(self.f(sol.x)))
            
            if loss < 1e-6:
                c_sol = sol.x.reshape(n_eq, self.n_pc).T
                if not any(np.isclose(c_sol, s, atol=1e-1).all() for s in seeds):
                    seeds.append(c_sol)
                    print(f"  -> Trovato seed per Branch {len(seeds)}/{n_branch} (Tentativo {attempts})")
                    
        if not seeds:
            print("ERRORE: Nessun seed trovato a N=0.")
            return

        print(f"\n--- FASE 2: Espansione del grado PCE (0 -> {degree_pc}) a eps costante ---")
        seeds_at_max_N = []
        
        for b_idx, c_seed in enumerate(seeds):
            c_current = c_seed.copy()
            success_N = True
            
            for n_iter in range(1, degree_pc + 1):
                self.phi, self.phi_norm = cp.generate_expansion(n_iter, self.seed_rv, retall=True)
                self.n_pc = len(self.phi_norm)
                
                new_guess = np.zeros((n_eq, self.n_pc))
                old_n_pc = c_current.shape[0]
                new_guess[:, :old_n_pc] = c_current.T
                
                current_guess = new_guess.ravel()
                
                def f_reg(c_flat):
                    return np.concatenate([self.f(c_flat), 1e-4 * (c_flat - current_guess)])

                def jac_reg(c_flat):
                    return np.vstack([self.jacobian(c_flat), 1e-4 * np.eye(len(c_flat))])

                sol = least_squares(f_reg, current_guess, jac=jac_reg, method='lm', xtol=1e-9, ftol=1e-9)
                loss = np.sum(np.abs(self.f(sol.x)))
                
                if loss < 1e-6:
                    c_current = sol.x.reshape(n_eq, self.n_pc).T
                else:
                    print(f"    ! Fallimento espansione a N={n_iter} per Branch {b_idx}")
                    success_N = False
                    break
            
            seeds_at_max_N.append(c_current if success_N else None)

        self.solution = [[] for _ in range(len(seeds))]

        if target_std <= start_std + 1e-6:
            for b_idx, c_max in enumerate(seeds_at_max_N):
                if c_max is not None:
                    self.solution[b_idx].append((c_max, degree_pc))
            self.update_stochastic_amplitude(target_std)
            return

        print(f"\n--- FASE 3: Continuazione Adattiva Sigma ({start_eps_max:.2f} -> {target_eps_max:.2f}) a N={degree_pc} ---")
        self.phi, self.phi_norm = cp.generate_expansion(degree_pc, self.seed_rv, retall=True)
        self.n_pc = len(self.phi_norm)
        
        ds_init = 0.01   
        ds_max = 0.1     
        ds_min = 1e-5    
        
        for b_idx, c_max_N in enumerate(seeds_at_max_N):
            if c_max_N is None:
                continue
                
            c_current = c_max_N.copy()
            current_sigma = start_std
            ds = ds_init
            success_sigma = True
            
            while target_std - current_sigma > 1e-8:
                if current_sigma + ds > target_std:
                    ds = target_std - current_sigma
                
                next_sigma = current_sigma + ds
                self.update_stochastic_amplitude(next_sigma)
                
                current_guess = c_current.ravel()
                
                def f_reg(c_flat):
                    return np.concatenate([self.f(c_flat), 1e-4 * (c_flat - current_guess)])

                def jac_reg(c_flat):
                    return np.vstack([self.jacobian(c_flat), 1e-4 * np.eye(len(c_flat))])

                sol = least_squares(f_reg, current_guess, jac=jac_reg, method='lm', xtol=1e-14, ftol=1e-14)
                loss = np.sum(np.abs(self.f(sol.x)))
                
                if sol.success and loss < 1e-6:
                    c_current = sol.x.reshape(n_eq, self.n_pc).T
                    current_sigma = next_sigma
                    ds = min(ds * 1.5, ds_max)
                else:
                    ds *= 0.3
                    if ds < ds_min:
                        print(f"    ! Tracciamento perso a eps limite ~ {(current_sigma*np.sqrt(3)):.4f}")
                        success_sigma = False
                        break
            
            if success_sigma:
                print(f"    -> Branch {b_idx} atterrato al target (eps = {target_eps_max:.2f}) con successo!")
                self.solution[b_idx].append((c_current, degree_pc))
                
        self.update_stochastic_amplitude(target_std)

    def run_random_searches(self, degree_pc, n_init, max_attempts=500):
        """Cerca soluzioni da guess puramente casuali al grado PCE desiderato."""
        self.phi, self.phi_norm = cp.generate_expansion(degree_pc, self.seed_rv, retall=True)
        self.n_pc = len(self.phi_norm)
        
        # Assicuriamoci di essere sull'ampiezza stocastica corretta
        target_std = float(cp.Std(self.eps))
        self.update_stochastic_amplitude(target_std)
        
        count = 0
        attempts = 0
        print(f"\n--- Ricerca Random (N={degree_pc}, Target {n_init} branch) ---")
        
        while count < n_init and attempts < max_attempts:
            attempts += 1
            guess = 3.0 * np.random.randn(self.n_pc, 2)
            
            sol = root(self.f, guess.ravel(), method='lm', tol=1e-8, jac=self.jacobian)
            loss = np.sum(np.abs(self.f(sol.x)))
            
            if loss < 1e-6:
                c_sol = sol.x.reshape(2, self.n_pc).T
                
                is_unique = True
                for branch in self.solution:
                    if branch and np.isclose(branch[0][0], c_sol, atol=1e-2).all():
                        is_unique = False
                        break
                
                if is_unique:
                    # Salviamo con la stessa struttura della continuation: lista di tuple (coeff, grado)
                    self.solution.append([(c_sol, degree_pc)])
                    count += 1
                    print(f"  -> Soluzione casuale {count}/{n_init} trovata (Tentativo {attempts})")
                    
        print(f"  -> Ricerca Random terminata.")

    def plot_xy_eps(self):
        """Plotta le soluzioni in spazio (x, eps) e (y, eps) usando la struttura dati unificata."""
        if not self.solution or all(not branch for branch in self.solution):
            print("Nessuna soluzione da plottare.")
            return
            
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].set_xlabel(r"$\epsilon$")
        ax[1].set_xlabel(r"$\epsilon$")
        ax[0].set_ylabel(r"$x$")
        ax[1].set_ylabel(r"$y$")
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        eps_grid = xi_grid * cp.Std(self.eps) + cp.E(self.eps)
                
        # --- Soluzioni Esatte ---
        eps_exact = np.linspace(np.min(eps_grid), np.max(eps_grid), 1000)
        ax[0].plot(eps_exact, eps_exact**3, 'k',  zorder=10, label=r'$\bar{u}$ (Ramo $E_1$)')
        ax[1].plot(eps_exact, np.zeros_like(eps_exact), 'k',  zorder=10)
        
        eps_ring = np.linspace(-1.0, 1.0, 500)
        ring_radius = np.sqrt(1 - eps_ring**2)
        
        ax[0].plot(eps_ring, ring_radius, 'k',  zorder=10, label=r'Anello Degenere')
        ax[0].plot(eps_ring, -ring_radius, 'k',  zorder=10)
        ax[1].plot(eps_ring, ring_radius, 'k',  zorder=10)
        ax[1].plot(eps_ring, -ring_radius, 'k',  zorder=10)
        
        # --- Soluzioni Approssimate PCE ---
        valid_branches = [b for b in self.solution if b]
        colors = plt.cm.jet(np.linspace(0, 1, max(1, len(valid_branches))))
        
        for i, branch in enumerate(valid_branches):
            for j in range(len(branch)):
                coeffs, deg = branch[j]
                
                # Dinamicamente generiamo il set di phi corretto per plottare
                phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
                approx = coeffs.T @ phi_eval 
                
                b_color = colors[i]
                label = rf'$N={{{deg}}}$ Branch {i}' 
                
                ax[0].plot(eps_grid, approx[0], color=b_color, zorder=5, linestyle='--',
                    marker='o', markersize=5, markevery=40, label=label if j==0 else None)
                ax[1].plot(eps_grid, approx[1], color=b_color, zorder=5, linestyle='--',
                    marker='o', markersize=5, markevery=40)

        for i in range(2):
            ax[i].grid(True, alpha=0.3)
            ax[i].set_xlim([np.min(eps_grid), np.max(eps_grid)])
            ax[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
            
        handles, labels = ax[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if len(by_label) < 15:
            ax[0].legend(by_label.values(), by_label.keys(), loc='upper left', borderpad=0.2, labelspacing=0.2, handlelength=1.5)
            
        fig.tight_layout()
        fig.savefig(f"plots/RingBifurcation_2d_unified.pdf", bbox_inches='tight')
        plt.show()

    def plot_3d_bifurcation(self):
        """Plot 3D universale per le soluzioni unificate."""
        if not self.solution or all(not branch for branch in self.solution):
            return
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel(r"$x$", labelpad=10)
        ax.set_ylabel(r"$y$", labelpad=10)
        ax.zaxis.set_rotate_label(False) 
        ax.set_zlabel(r"$\epsilon$", rotation=90, labelpad=15)
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        eps_grid = xi_grid * cp.Std(self.eps) + cp.E(self.eps)
        
        # --- Soluzioni Esatte 3D ---
        eps_exact = np.linspace(np.min(eps_grid)-0.05, np.max(eps_grid)+0.05, 500)
        ax.plot(eps_exact**3, np.zeros_like(eps_exact), eps_exact, 'k',  zorder=10, label=r'$\bar{u}$ (Ramo $E_1$)')
        
        eps_ring = np.linspace(-1.0, 1.0, 30)
        theta = np.linspace(0, 2*np.pi, 40)
        EPS, THETA = np.meshgrid(eps_ring, theta)
        R = np.sqrt(np.clip(1 - EPS**2, 0, None))
        X_ring = R * np.cos(THETA)
        Y_ring = R * np.sin(THETA)
        ax.plot_wireframe(X_ring, Y_ring, EPS, color='gray', alpha=0.3, linewidth=1.0, zorder=1)

        # --- Soluzioni Approssimate PCE ---
        valid_branches = [b for b in self.solution if b]
        colors = plt.cm.jet(np.linspace(0, 1, max(1, len(valid_branches))))
        
        for i, branch in enumerate(valid_branches):
            for j in range(len(branch)):
                coeffs, deg = branch[j]
                phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
                approx = coeffs.T @ phi_eval 
                
                b_color = colors[i]
                label = rf'$N={{{deg}}}$ Branch {i}' 
                
                ax.plot(approx[0], approx[1], eps_grid, color=b_color, 
                        linestyle='--', marker='o', markersize=5, markevery=40, label=label if j==0 else None, zorder=5)

        ax.view_init(elev=17, azim=-110)
        
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='x', pad=-2)
        ax.tick_params(axis='y', pad=-2)
        ax.tick_params(axis='z', pad=-2)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if len(by_label) < 15:
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', borderpad=0.2, labelspacing=0.2, handlelength=1.5)
        
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect(None, zoom=0.90)
        else:
            ax.dist = 11
            
        fig.savefig(f"plots/RingBifurcation_3d_unified.pdf", bbox_inches='tight', pad_inches=0.1)
        plt.show()

if __name__ == "__main__":
    degree_pc = 4
    
    model = RingBifurcationUnified(
        eps=cp.Uniform(0.0, 0.6), 
        seed_rv=cp.J(cp.Uniform(-np.sqrt(3), np.sqrt(3))),
        n_samples=1000
    )

    RUN_CONTINUATION = True
    RUN_RANDOM_SEARCH = False

    if RUN_CONTINUATION:
        model.continuation(
            degree_pc=degree_pc, 
            n_branch=3,
            start_eps_max=0.3
        )
        
    if RUN_RANDOM_SEARCH:
        model.run_random_searches(
            degree_pc=degree_pc, 
            n_init=10,
            max_attempts=300
        )
    
    # Genera i grafici finali cumulativi
    print("\nGenerazione grafici in corso...")
    model.plot_xy_eps()
    model.plot_3d_bifurcation()