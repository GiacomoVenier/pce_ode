import chaospy as cp
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import os

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

class SphericalSystem():
    def __init__(self, mu, seed_rv, n_samples=10000):
        self.mu = mu
        self.seed_rv = seed_rv
        self.n_kl = len(self.seed_rv)
        self.n_samples = n_samples

        self.seed_rv_samples = np.atleast_2d(self.seed_rv.sample(n_samples)) 
        self.mu_samples = self.seed_rv_samples[0] * cp.Std(self.mu) + cp.E(self.mu) 
        
        # Inizializza un array vuoto per contenere le soluzioni, indipendentemente dal metodo usato
        self.solution = []

    def update_stochastic_amplitude(self, new_std):
        """Modifica dinamicamente la deviazione standard per l'omotopia (Fase 1 e 2)."""
        self.mu_samples = self.seed_rv_samples[0] * new_std + cp.E(self.mu)

    def f(self, c):
        c = c.reshape(2, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) 
        x, y = c.T @ phi 
        
        x_dot = (((x - self.mu_samples**3) * (1 - x**2 - y**2 - self.mu_samples**2)) @ phi.T) / self.n_samples 
        y_dot = ((y * (1 - x**2 - y**2 - self.mu_samples**2)) @ phi.T) / self.n_samples 
        
        return np.concatenate([x_dot, y_dot])

    def jacobian(self, c):
        c = c.reshape(2, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) 
        x, y = c.T @ phi

        Jxx = (((1 - 3*x**2 - y**2 - self.mu_samples**2 + 2*x*self.mu_samples**3) * phi) @ phi.T) / self.n_samples
        Jxy = (((-2*x*y + 2*y*self.mu_samples**3) * phi) @ phi.T) / self.n_samples
        Jyx = (((-2*x*y) * phi) @ phi.T) / self.n_samples
        Jyy = (((1 - x**2 - 3*y**2 - self.mu_samples**2) * phi) @ phi.T) / self.n_samples
        
        J = np.block([
                [Jxx, Jxy],
                [Jyx, Jyy]
            ])
        return J

    def continuation(self, degree_pc, n_branch, start_std=0.057735):
        """
        Continuazione stocastica con STEP ADATTIVO.
        Il solutore accelera nelle zone facili e rallenta nelle zone difficili.
        """
        target_std = float(cp.Std(self.mu))
        n_eq = 2
        
        if target_std <= start_std:
            start_std = target_std
            
        self.solution = []

        # =========================================================================
        # FASE 1: Isolamento dei Seed Branch a N=0
        # =========================================================================
        print(f"\n--- FASE 1: Ricerca di {n_branch} branch unici a N=0 (std iniziale = {start_std:.4f}) ---")
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
                if not any(np.isclose(c_sol, s, atol=1e-2).all() for s in seeds):
                    seeds.append(c_sol)
                    print(f"  -> Trovato seed per Branch {len(seeds)}/{n_branch} (Tentativo {attempts})")
                    
        if not seeds:
            print("ERRORE: Nessun seed trovato a N=0. Verifica il valore di start_std.")
            return

        actual_branches = len(seeds)
        self.solution = [[] for _ in range(actual_branches)]

        # =========================================================================
        # FASE 2: Innalzamento del Grado Spettrale (N=0 -> N=degree_pc)
        # =========================================================================
        print(f"\n--- FASE 2: Espansione del grado PCE (0 -> {degree_pc}) a std costante ---")
        seeds_at_max_N = []
        
        for b_idx, c_seed in enumerate(seeds):
            print(f"  Innalzamento grado per Branch {b_idx}...")
            c_current = c_seed.copy()
            success_N = True
            
            for n_iter in range(1, degree_pc + 1):
                self.phi, self.phi_norm = cp.generate_expansion(n_iter, self.seed_rv, retall=True)
                self.n_pc = len(self.phi_norm)
                
                new_guess = np.zeros((n_eq, self.n_pc))
                old_n_pc = c_current.shape[0]
                new_guess[:, :old_n_pc] = c_current.T
                
                sol = root(self.f, new_guess.ravel(), method='lm', tol=1e-9, jac=self.jacobian)
                loss = np.sum(np.abs(self.f(sol.x)))
                
                if loss < 1e-6:
                    c_current = sol.x.reshape(n_eq, self.n_pc).T
                else:
                    sol = root(self.f, new_guess.ravel(), method='hybr', tol=1e-7, jac=self.jacobian)
                    loss = np.sum(np.abs(self.f(sol.x)))
                    if loss < 1e-6:
                        c_current = sol.x.reshape(n_eq, self.n_pc).T
                    else:
                        print(f"    ! Fallimento espansione a N={n_iter} per il Branch {b_idx}")
                        success_N = False
                        break
            
            seeds_at_max_N.append(c_current if success_N else None)

        # =========================================================================
        # FASE 3: OMOTOPIA ADATTIVA (Step-size control)
        # =========================================================================
        print(f"\n--- FASE 3: Continuazione Adattiva in Sigma ({start_std:.4f} -> {target_std:.4f}) a N={degree_pc} ---")
        self.phi, self.phi_norm = cp.generate_expansion(degree_pc, self.seed_rv, retall=True)
        self.n_pc = len(self.phi_norm)
        
        # Parametri dell'algoritmo adattivo
        ds_init = 0.02   # Passo iniziale
        ds_max = 0.1     # Passo massimo (per evitare che salti da 0 a 10 in un colpo solo)
        ds_min = 1e-5    # Sotto questa soglia dichiariamo la divergenza
        
        for b_idx, c_max_N in enumerate(seeds_at_max_N):
            if c_max_N is None:
                continue
                
            print(f"  Dilatazione stocastica per Branch {b_idx}...")
            c_current = c_max_N.copy()
            current_sigma = start_std
            ds = ds_init
            success_sigma = True
            
            # Loop adattivo: finché non siamo arrivati al target
            while target_std - current_sigma > 1e-8:
                # Evita di superare il target al passo finale
                if current_sigma + ds > target_std:
                    ds = target_std - current_sigma
                
                next_sigma = current_sigma + ds
                self.update_stochastic_amplitude(next_sigma)
                
                # Prova lo step
                sol = root(self.f, c_current.ravel(), method='lm', tol=1e-9, jac=self.jacobian)
                loss = np.sum(np.abs(self.f(sol.x)))
                
                if sol.success and loss < 1e-6:
                    # SUCCESS: Accetta il passo e ACCELERA per la prossima volta!
                    c_current = sol.x.reshape(n_eq, self.n_pc).T
                    current_sigma = next_sigma
                    ds = min(ds * 1.5, ds_max) # Moltiplica per 1.5, senza superare ds_max
                    
                else:
                    # FALLIMENTO: Ripiega sul solutore ibrido prima di arrendersi
                    sol_hybr = root(self.f, c_current.ravel(), method='hybr', tol=1e-7, jac=self.jacobian)
                    loss_hybr = np.sum(np.abs(self.f(sol_hybr.x)))
                    
                    if loss_hybr < 1e-6:
                        # Salvati all'ultimo! Accetta il passo ma mantieni la stessa velocità (non accelerare)
                        c_current = sol_hybr.x.reshape(n_eq, self.n_pc).T
                        current_sigma = next_sigma
                    else:
                        # RIFIUTO TOTALE: Riduci il passo drasticamente e riprova
                        ds *= 0.3 # Dimezza o riduci di più (es. 0.3 per tagliare la testa al toro)
                        if ds < ds_min:
                            print(f"    ! Tracciamento perso irrimediabilmente a std={current_sigma:.4f} (passo < {ds_min})")
                            success_sigma = False
                            break
            
            if success_sigma:
                print(f"    -> Branch {b_idx} ancorato al target finale con successo!")
                self.solution[b_idx].append((c_current, degree_pc))
                
        # Ripristino finale dello stato
        self.update_stochastic_amplitude(target_std)

    def run_random_searches(self, degree_pc, n_init, max_attempts=500):
        """Cerca soluzioni da guess puramente casuali al grado PCE desiderato."""
        self.phi, self.phi_norm = cp.generate_expansion(degree_pc, self.seed_rv, retall=True)
        self.n_pc = len(self.phi_norm)
        
        # Assicuriamoci di essere sull'ampiezza stocastica corretta
        target_std = float(cp.Std(self.mu))
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

    def plot_xy_mu(self):
        """Plotta le soluzioni in spazio (x, mu) e (y, mu) usando la struttura dati unificata."""
        if not self.solution or all(not branch for branch in self.solution):
            print("Nessuna soluzione da plottare.")
            return
            
        fig, ax = plt.subplots(1, 2)
        ax[0].set_xlabel(r"$\mu$")
        ax[1].set_xlabel(r"$\mu$")
        ax[0].set_ylabel(r"$x$")
        ax[1].set_ylabel(r"$y$")
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        mu_grid = xi_grid * cp.Std(self.mu) + cp.E(self.mu)
                
        # --- Soluzioni Esatte ---
        mu_exact = np.linspace(np.min(mu_grid), np.max(mu_grid), 1000)
        ax[0].plot(mu_exact, mu_exact**3, 'k',  zorder=10, label=r'$\bar{u}$ (Ramo $E_1$)')
        ax[1].plot(mu_exact, np.zeros_like(mu_exact), 'k',  zorder=10)
        
        mu_ring = np.linspace(-1.0, 1.0, 500)
        ring_radius = np.sqrt(1 - mu_ring**2)
        
        ax[0].plot(mu_ring, ring_radius, 'k',  zorder=10, label=r'Anello Degenere')
        ax[0].plot(mu_ring, -ring_radius, 'k',  zorder=10)
        ax[1].plot(mu_ring, ring_radius, 'k',  zorder=10)
        ax[1].plot(mu_ring, -ring_radius, 'k',  zorder=10)
        
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
                
                ax[0].plot(mu_grid, approx[0], color=b_color, zorder=5, linestyle='--',
                    marker='o', markersize=5, markevery=40, label=label if j==0 else None)
                ax[1].plot(mu_grid, approx[1], color=b_color, zorder=5, linestyle='--',
                    marker='o', markersize=5, markevery=40)

        for i in range(2):
            ax[i].grid(True, alpha=0.3)
            ax[i].set_xlim([np.min(mu_grid), np.max(mu_grid)])
            ax[i].xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
            
        # handles, labels = ax[0].get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # if len(by_label) < 15:
        #     ax[0].legend(by_label.values(), by_label.keys(), loc='upper left', borderpad=0.2, labelspacing=0.2, handlelength=1.5)
            
        fig.tight_layout()
        fig.savefig(f"plots/SphericalSystem_2d_xy_rho_N_{degree_pc}_rho_{self.mu}_N_init_{len(valid_branches)}.pdf", bbox_inches='tight')
        plt.show()

    def plot_3d_bifurcation(self):
        """Plot 3D universale per le soluzioni unificate."""
        if not self.solution or all(not branch for branch in self.solution):
            return
            
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel(r"$x$", labelpad=10)
        ax.set_ylabel(r"$y$", labelpad=10)
        ax.zaxis.set_rotate_label(False) 
        ax.set_zlabel(r"$\mu$", rotation=90, labelpad=15)
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        mu_grid = xi_grid * cp.Std(self.mu) + cp.E(self.mu)
        
        # --- Soluzioni Esatte 3D ---
        mu_exact = np.linspace(np.min(mu_grid)-0.05, np.max(mu_grid)+0.05, 500)
        ax.plot(mu_exact**3, np.zeros_like(mu_exact), mu_exact, 'k',  zorder=10, label=r'$\bar{u}$ (Ramo $E_1$)')
        
        mu_min_plot = max(-1.0, np.min(mu_grid) - 0.05)
        mu_max_plot = min(1.0, np.max(mu_grid) + 0.05)
        
        if mu_min_plot < mu_max_plot:
            mu_ring = np.linspace(mu_min_plot, mu_max_plot, 30)
            
            # Lascio np.pi se vuoi vederla tagliata a metà (a "guscio aperto"), 
            # altrimenti rimetti 2*np.pi per averla chiusa
            theta = np.linspace(0, 2*np.pi, 40) 
            
            MU_MESH, THETA = np.meshgrid(mu_ring, theta)
            R = np.sqrt(np.clip(1 - MU_MESH**2, 0, None))
            X_ring = R * np.cos(THETA)
            Y_ring = R * np.sin(THETA)
            ax.plot_wireframe(X_ring, Y_ring, MU_MESH, color='gray', alpha=0.3, linewidth=1.0, zorder=1)
        

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
                
                ax.plot(approx[0], approx[1], mu_grid, color=b_color, 
                        linestyle='--', marker='o', markersize=5, markevery=40, label=label if j==0 else None, zorder=5)

        ax.view_init(elev=17, azim=-110)
        
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='x', pad=-3)
        ax.tick_params(axis='y', pad=-1)
        ax.tick_params(axis='z')
        
        # handles, labels = ax.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # if len(by_label) < 15:
        #     ax.legend(by_label.values(), by_label.keys(), loc='upper left', borderpad=0.2, labelspacing=0.2, handlelength=1.5)
        
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect(None, zoom=0.90)
        else:
            ax.dist = 11
            
        fig.savefig(f"plots/SphericalSystem_3d_xy_rho_N_{degree_pc}_rho_{self.mu}_N_init_{len(valid_branches)}.pdf", bbox_inches='tight', pad_inches=0.1)
        plt.show()

if __name__ == "__main__":
    degree_pc = 10
    
    model = SphericalSystem(
        mu=cp.Uniform(-2, 2), 
        seed_rv=cp.J(cp.Uniform(-np.sqrt(3), np.sqrt(3))),
        n_samples=1000
    )

    RUN_CONTINUATION = False
    RUN_RANDOM_SEARCH = True

    if RUN_CONTINUATION:
        model.continuation(
            degree_pc=degree_pc, 
            n_branch=20,
            start_std=0.1/np.sqrt(3)
        )
        
    if RUN_RANDOM_SEARCH:
        model.run_random_searches(
            degree_pc=degree_pc, 
            n_init=1,
            max_attempts=300
        )
    
    # Genera i grafici finali cumulativi
    print("\nGenerazione grafici in corso...")
    model.plot_xy_mu()
    model.plot_3d_bifurcation()