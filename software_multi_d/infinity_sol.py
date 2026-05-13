# import chaospy as cp
# import numpy as np
# from scipy.optimize import root
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# np.random.seed(241)

# # Impostazioni grafiche
# plt.rcParams.update({
#     'font.size': 20,
#     'axes.titlesize': 20,
#     'axes.labelsize': 20,
#     'legend.fontsize': 15,
#     'xtick.labelsize': 15,
#     'ytick.labelsize': 15
# })

# plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.size'] = 20
# plt.rcParams['font.family'] = 'lmodern'

# class RingBifurcationSystem():
#     def __init__(self, eps, seed_rv, n_samples=10000):
#         self.eps = eps
#         self.seed_rv = seed_rv
#         self.n_kl = len(self.seed_rv)
#         self.n_samples = n_samples

#         self.seed_rv_samples = np.atleast_2d(self.seed_rv.sample(n_samples)) 
#         self.eps_samples = self.seed_rv_samples[0] * cp.Std(self.eps) + cp.E(self.eps) 

#     def f(self, c):
#         c = c.reshape(2, self.n_pc).T
#         phi = self.phi(*self.seed_rv_samples) 
#         x, y = c.T @ phi 
        
#         # Nuovo campo vettoriale: 
#         # dx = (x - eps^3) * (1 - x^2 - y^2 - eps^2)
#         # dy = y * (1 - x^2 - y^2 - eps^2)
#         x_dot = (((x - self.eps_samples**3) * (1 - x**2 - y**2 - self.eps_samples**2)) @ phi.T) / self.n_samples 
#         y_dot = ((y * (1 - x**2 - y**2 - self.eps_samples**2)) @ phi.T) / self.n_samples 
        
#         return np.concatenate([x_dot, y_dot])

#     def jacobian(self, c):
#         c = c.reshape(2, self.n_pc).T
#         phi = self.phi(*self.seed_rv_samples) 
#         x, y = c.T @ phi

#         # Derivate parziali analitiche
#         Jxx = (((1 - 3*x**2 - y**2 - self.eps_samples**2 + 2*x*self.eps_samples**3) * phi) @ phi.T) / self.n_samples
#         Jxy = (((-2*x*y + 2*y*self.eps_samples**3) * phi) @ phi.T) / self.n_samples
#         Jyx = (((-2*x*y) * phi) @ phi.T) / self.n_samples
#         Jyy = (((1 - x**2 - 3*y**2 - self.eps_samples**2) * phi) @ phi.T) / self.n_samples
        
#         J = np.block([
#                 [Jxx, Jxy],
#                 [Jyx, Jyy]
#             ])
#         return J

#     def run(self, degree_pc, n_init):
#         self.phi, self.phi_norm = cp.generate_expansion(degree_pc, self.seed_rv, retall=True)
#         self.n_pc = len(self.phi_norm)
#         self.solution = np.zeros((n_init, self.n_pc, 2))
#         self.coeff_init = []
#         count = 0
        
#         while count < n_init:
#             self.coeff_init.append(2.0 * np.random.randn(self.n_pc, 2))
#             sol = root(self.f, self.coeff_init[-1].ravel(), method='lm', tol=1e-8, jac=self.jacobian)
#             loss = np.sum(np.abs(self.f(sol.x)))
            
#             if (loss < 1e-6) and (not np.isclose(self.solution[:count], sol.x.reshape(2, self.n_pc).T).all(2).all(1).any()):
#                 self.solution[count] = sol.x.reshape(2, self.n_pc).T
#                 count += 1
#                 print(f"Random Init Found: {count}/{n_init}")

#         self.samples_solution = (self.solution.transpose(0, 2, 1) @ self.phi(*self.seed_rv_samples))

#     def continuation(self, degree_pc, n_branch):
#         self.solution = [[] for _ in range(n_branch)]
#         self.samples_solution = [[] for _ in range(n_branch)]
        
#         for i in range(n_branch):
#             degree_pc_iter = 0
#             counter = 0 
#             self.phi, self.phi_norm = cp.generate_expansion(degree_pc_iter, self.seed_rv, retall=True)
#             self.n_pc = len(self.phi_norm)
            
#             while degree_pc_iter <= degree_pc:
#                 counter += 1 

#                 if degree_pc_iter == 0:
#                     current_guess = 2.0 * np.random.randn(2)
#                 else:
#                     new_guess = np.zeros((2, self.n_pc))
#                     old_n_pc = c.shape[0]
#                     new_guess[0, :old_n_pc] = c[:, 0] 
#                     new_guess[1, :old_n_pc] = c[:, 1] 
#                     current_guess = new_guess.ravel()

#                 sol = root(self.f, current_guess, method='lm', tol=1e-8, jac=self.jacobian)
#                 loss = np.sum(np.abs(self.f(sol.x)))
                
#                 c = sol.x.reshape(2, self.n_pc).T 
                
#                 control = not any(np.isclose(c[0, 0], self.solution[j][0][0][0,0], atol=1e-2) 
#                                  for j in range(i) if self.solution[j]) if (degree_pc_iter == 0 and i > 0) else True
                
#                 if (loss < 1e-6) and control:
#                     self.solution[i].append((c, degree_pc_iter)) 
#                     self.samples_solution[i].append((c.T @ self.phi(*self.seed_rv_samples)))
                    
#                     degree_pc_iter += 1
#                     counter = 0 
#                     self.phi, self.phi_norm = cp.generate_expansion(degree_pc_iter, self.seed_rv, retall=True)
#                     self.n_pc = len(self.phi_norm)
                
#                 if counter > 200: 
#                     print(f"Stopping branch {i}: reached max attempts.")
#                     break
#             print(f"Branch sequence found for branch {i}")

#     def plot_xy_eps(self, n_branch):
#         fig, ax = plt.subplots(1, 2, figsize=(20, 7))
#         ax[0].set_xlabel(r"$\epsilon$")
#         ax[1].set_xlabel(r"$\epsilon$")
#         ax[0].set_ylabel(r"$x$")
#         ax[1].set_ylabel(r"$y$")
        
#         xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
#         grid_eval = np.atleast_2d(xi_grid)
#         eps_grid = xi_grid * cp.Std(self.eps) + cp.E(self.eps)
        
#         # --- Soluzioni Esatte ---
#         eps_exact = np.linspace(np.min(eps_grid)-0.2, np.max(eps_grid)+0.2, 1000)
        
#         # 1. Ramo Isolato
#         x_point = eps_exact**3
#         ax[0].plot(eps_exact, x_point, 'k', linewidth=4.0, zorder=1, label=r'Ramo $E_1$')
#         ax[1].plot(eps_exact, np.zeros_like(eps_exact), 'k', linewidth=4.0, zorder=1)
        
#         # 2. Anello (Proiezioni sui piani principali)
#         eps_ring = np.linspace(-1.0, 1.0, 500)
#         ring_radius = np.sqrt(1 - eps_ring**2)
        
#         ax[0].plot(eps_ring, ring_radius, 'gray', linewidth=3.0, zorder=1, label=r'Anello Degenere')
#         ax[0].plot(eps_ring, -ring_radius, 'gray', linewidth=3.0, zorder=1)
#         ax[1].plot(eps_ring, ring_radius, 'gray', linewidth=3.0, zorder=1)
#         ax[1].plot(eps_ring, -ring_radius, 'gray', linewidth=3.0, zorder=1)
        
#         # --- Soluzioni Approssimate PCE ---
#         max_deg = max([deg for branch in self.solution for (_, deg) in branch]) if self.solution[0] else 0
#         branch_colors = ["#065895", "#f79a25", "#77ac30", "#d9534f", "#8e44ad"]
        
#         for i in range(min(n_branch, len(self.solution))):
#             for j in range(len(self.solution[i])):
#                 coeffs, deg = self.solution[i][j]
                
#                 phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
#                 approx = coeffs.T @ phi_eval 
                
#                 if deg == max_deg:
#                     b_color = branch_colors[i % len(branch_colors)]
#                     label = rf'$u_{{{deg}}}$ Branch {i}' 
                    
#                     ax[0].plot(eps_grid, approx[0], color=b_color, linewidth=2.0, zorder=5, linestyle='--',
#                      marker='o', markersize=6, markevery=30, label=label)
#                     ax[1].plot(eps_grid, approx[1], color=b_color, linewidth=2.0, zorder=5, linestyle='--',
#                      marker='o', markersize=6, markevery=30)

#         for i in range(2):
#             ax[i].grid(True, alpha=0.3)
#             ax[i].set_xlim([np.min(eps_grid), np.max(eps_grid)])
            
#         fig.tight_layout()
#         plt.show()

#     def plot_3d_bifurcation(self, n_branch):
#         fig = plt.figure(figsize=(12, 10))
#         ax = fig.add_subplot(111, projection='3d')
        
#         ax.set_xlabel(r"$x$")
#         ax.set_ylabel(r"$y$")
#         ax.set_zlabel(r"$\epsilon$")
#         ax.set_title("Diagramma 3D: Curva Cubica e Sfera Degenere")
        
#         xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
#         grid_eval = np.atleast_2d(xi_grid)
#         eps_grid = xi_grid * cp.Std(self.eps) + cp.E(self.eps)
        
#         # --- Soluzioni Esatte 3D ---
#         # 1. Ramo Isolato E1 = (eps^3, 0)
#         eps_exact = np.linspace(np.min(eps_grid)-0.2, np.max(eps_grid)+0.2, 500)
#         ax.plot(eps_exact**3, np.zeros_like(eps_exact), eps_exact, 'k', linewidth=4.0, zorder=1, label=r'Ramo $E_1$')
        
#         # 2. Superficie dell'Anello Degenere (Una "sfera deformata")
#         eps_ring = np.linspace(-1.0, 1.0, 30)
#         theta = np.linspace(0, 2*np.pi, 40)
#         EPS, THETA = np.meshgrid(eps_ring, theta)
#         R = np.sqrt(1 - EPS**2)
#         X_ring = R * np.cos(THETA)
#         Y_ring = R * np.sin(THETA)
        
#         ax.plot_wireframe(X_ring, Y_ring, EPS, color='gray', alpha=0.2, linewidth=1.0)
#         ax.plot_surface(X_ring, Y_ring, EPS, color='red', alpha=0.1, edgecolor='none')

#         # --- Soluzioni Approssimate PCE ---
#         max_deg = max([deg for branch in self.solution for (_, deg) in branch]) if self.solution[0] else 0
#         branch_colors = ["#065895", "#f79a25", "#77ac30", "#d9534f", "#8e44ad"]
        
#         for i in range(min(n_branch, len(self.solution))):
#             for j in range(len(self.solution[i])):
#                 coeffs, deg = self.solution[i][j]
#                 if deg == max_deg:
#                     phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
#                     approx = coeffs.T @ phi_eval 
                    
#                     b_color = branch_colors[i % len(branch_colors)]
#                     label = rf'$u_{{{deg}}}$ Branch {i}' 
                    
#                     ax.plot(approx[0], approx[1], eps_grid, color=b_color, linewidth=2.5, 
#                             linestyle='--', marker='o', markersize=6, markevery=30, label=label, zorder=5)

#         ax.view_init(elev=20, azim=-55)
        
#         # Pulisco la legenda per evitare doppioni generati dalla superficie 3D
#         handles, labels = ax.get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         ax.legend(by_label.values(), by_label.keys())
        
#         plt.tight_layout()
#         plt.show()

# if __name__ == "__main__":
#     degree_pc = 6
#     # Imposto a 4 per tentare di catturare il ramo principale + intersezioni multiple sull'anello
#     n_branch_to_approximate = 100

#     # Range che comprende la nascita dell'anello (-1, 1) e l'attraversamento
#     model = RingBifurcationSystem(
#         eps=cp.Uniform(-0.1, 0.1), 
#         seed_rv=cp.J(cp.Uniform(-np.sqrt(3), np.sqrt(3))), # 1D chaos
#         n_samples=1000
#     )

#     print("\n=== Executing Degree Continuation ===")
#     model.continuation(degree_pc=degree_pc, n_branch=n_branch_to_approximate)
#     model.plot_xy_eps(n_branch=n_branch_to_approximate)
#     model.plot_3d_bifurcation(n_branch=n_branch_to_approximate)

import chaospy as cp
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

# Impostazioni grafiche
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

class RingRandomSolver():
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

    def run_random_searches(self, degree_pc, n_init, max_attempts=500):
        """Esegue n_init ricerche di radici a partire da guess completamente casuali."""
        self.phi, self.phi_norm = cp.generate_expansion(degree_pc, self.seed_rv, retall=True)
        self.n_pc = len(self.phi_norm)
        self.solution = []
        
        count = 0
        attempts = 0
        
        print(f"Ricerca di {n_init} soluzioni casuali al grado PCE {degree_pc}...")
        while count < n_init and attempts < max_attempts:
            attempts += 1
            guess = 3.0 * np.random.randn(self.n_pc, 2)
            
            sol = root(self.f, guess.ravel(), method='lm', tol=1e-8, jac=self.jacobian)
            loss = np.sum(np.abs(self.f(sol.x)))
            
            if loss < 1e-6:
                c_sol = sol.x.reshape(2, self.n_pc).T
                
                is_unique = True
                for existing_sol in self.solution:
                    if np.isclose(existing_sol, c_sol, atol=1e-2).all():
                        is_unique = False
                        break
                
                if is_unique:
                    self.solution.append(c_sol)
                    count += 1
                    print(f"Soluzione {count}/{n_init} trovata (Tentativo {attempts})")
                    
        print(f"Ricerca terminata. Trovate {len(self.solution)} soluzioni uniche.")

    def plot_random_solutions_2d(self):
        """Plotta tutte le soluzioni trovate sovrapposte alla soluzione esatta in 2D."""
        fig, ax = plt.subplots(1, 2, figsize=(20, 7))
        ax[0].set_xlabel(r"$\epsilon$")
        ax[1].set_xlabel(r"$\epsilon$")
        ax[0].set_ylabel(r"$x$")
        ax[1].set_ylabel(r"$y$")
        ax[0].set_title(f"Soluzioni Casuali PCE (x vs $\epsilon$)")
        ax[1].set_title(f"Soluzioni Casuali PCE (y vs $\epsilon$)")
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        eps_grid = xi_grid * cp.Std(self.eps) + cp.E(self.eps)
        
        # --- Soluzioni Esatte (Riferimento) ---
        eps_exact = np.linspace(np.min(eps_grid)-0.2, np.max(eps_grid)+0.2, 1000)
        ax[0].plot(eps_exact, eps_exact**3, 'k', linewidth=4.0, zorder=10, label=r'Ramo Esatto $E_1$')
        ax[1].plot(eps_exact, np.zeros_like(eps_exact), 'k', linewidth=4.0, zorder=10)
        
        eps_ring = np.linspace(-1.0, 1.0, 500)
        ring_radius = np.sqrt(1 - eps_ring**2)
        ax[0].plot(eps_ring, ring_radius, 'k', linewidth=4.0, zorder=10, label=r'Anello Degenere')
        ax[0].plot(eps_ring, -ring_radius, 'k', linewidth=4.0, zorder=10)
        ax[1].plot(eps_ring, ring_radius, 'k', linewidth=4.0, zorder=10)
        ax[1].plot(eps_ring, -ring_radius, 'k', linewidth=4.0, zorder=10)
        
        # --- Soluzioni PCE Trovate ---
        phi_eval = cp.generate_expansion(len(self.phi_norm)-1, self.seed_rv, retall=True)[0](*grid_eval)
        colors = plt.cm.jet(np.linspace(0, 1, len(self.solution)))
        
        for idx, coeff in enumerate(self.solution):
            approx = coeff.T @ phi_eval
            
            ax[0].plot(eps_grid, approx[0], color=colors[idx], alpha=0.6, linewidth=1.5, zorder=5)
            ax[1].plot(eps_grid, approx[1], color=colors[idx], alpha=0.6, linewidth=1.5, zorder=5)

        for i in range(2):
            ax[i].grid(True, alpha=0.3)
            ax[i].set_xlim([np.min(eps_grid), np.max(eps_grid)])
            
        handles, labels = ax[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0].legend(by_label.values(), by_label.keys(), loc='upper left')
        
        fig.tight_layout()
        plt.show()

    def plot_random_solutions_3d(self):
        """Plot in 3D delle soluzioni casuali trovate dal solutore."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$\epsilon$")
        ax.set_title("Soluzioni Casuali PCE in 3D")
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.atleast_2d(xi_grid)
        eps_grid = xi_grid * cp.Std(self.eps) + cp.E(self.eps)
        
        # --- Soluzioni Esatte 3D ---
        # 1. Ramo Isolato
        eps_exact = np.linspace(np.min(eps_grid)-0.2, np.max(eps_grid)+0.2, 500)
        ax.plot(eps_exact**3, np.zeros_like(eps_exact), eps_exact, 'k', linewidth=5.0, zorder=1, label=r'Ramo Esatto $E_1$')
        
        # 2. Sfera Degenere
        eps_ring = np.linspace(-1.0, 1.0, 30)
        theta = np.linspace(0, 2*np.pi, 40)
        EPS, THETA = np.meshgrid(eps_ring, theta)
        R = np.sqrt(np.clip(1 - EPS**2, 0, None))
        X_ring = R * np.cos(THETA)
        Y_ring = R * np.sin(THETA)
        
        ax.plot_wireframe(X_ring, Y_ring, EPS, color='gray', alpha=0.3, linewidth=1.0)
        
        # --- Soluzioni PCE Trovate ---
        phi_eval = cp.generate_expansion(len(self.phi_norm)-1, self.seed_rv, retall=True)[0](*grid_eval)
        colors = plt.cm.jet(np.linspace(0, 1, len(self.solution)))
        
        for idx, coeff in enumerate(self.solution):
            approx = coeff.T @ phi_eval 
            ax.plot(approx[0], approx[1], eps_grid, color=colors[idx], alpha=0.8, linewidth=2.0, zorder=5)

        ax.view_init(elev=20, azim=-55)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    degree_pc = 4     
    n_random_inits = 10 # Cambia questo numero per cercare più o meno "fili" sull'anello
    
    model = RingRandomSolver(
        eps=cp.Uniform(-0.1, 0.1), 
        seed_rv=cp.J(cp.Uniform(-np.sqrt(3), np.sqrt(3))),
        n_samples=1000
    )

    print("\n=== Executing Random PCE Search ===")
    model.run_random_searches(degree_pc=degree_pc, n_init=n_random_inits)
    
    # Plot 2D
    model.plot_random_solutions_2d()
    
    # Nuovo Plot 3D
    model.plot_random_solutions_3d()