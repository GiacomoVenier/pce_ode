import chaospy as cp
import numpy as np
from scipy.optimize import root
from scipy.stats import gaussian_kde, wasserstein_distance
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as patches

np.random.seed(241)

# matplotlib.pyplot options
plt.rcParams.update({
    'font.size': 30,           # Dimensione generale del font
    'axes.titlesize': 30,      # Titolo dell'asse
    'axes.labelsize': 30,      # Etichette degli assi
    'legend.fontsize': 20,     # Legenda
    'xtick.labelsize': 30,     # Etichette asse x
    'ytick.labelsize': 30      # Etichette asse y
})


plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 30
plt.rcParams['font.family'] = 'lmodern'

class Lorenz():
    def __init__(self, gamma, rho, theta, seed_rv, n_samples=10000):
        '''
        gamma --> {chaospy.Distribution}
        rho --> {chaospy.Distribution}
        theta --> {chaospy.Distribution}
        seed_rv --> {chaospy.Distribution}
        n_samples --> {int}
        '''
        self.gamma = gamma
        self.rho = rho
        self.theta = theta
        self.seed_rv = seed_rv
        self.n_kl = len(self.seed_rv)
        self.n_samples = n_samples

        self.seed_rv_samples = self.seed_rv.sample(n_samples) #(3,self.n_pc)
        self.gamma_samples = self.seed_rv_samples[0]*cp.Std(self.gamma) + cp.E(self.gamma) #(self.n_pc)
        self.rho_samples = self.seed_rv_samples[1]*cp.Std(self.rho) + cp.E(self.rho) #(self.n_pc)
        self.theta_samples = self.seed_rv_samples[2]*cp.Std(self.theta) + cp.E(self.theta) #(self.n_pc)

    def f(self, c):
        '''
        compute F(c)
        '''
        c = c.reshape(3,self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) #(self.n_pc,self.n_samples)
        x, y, z = c.T @ phi #(3,self.n_samples)
        x_dot = ((self.gamma_samples * (y-x))@phi.T)/self.n_samples #(self.n_pc)
        y_dot = ((self.rho_samples * x-y-x*z)@phi.T)/self.n_samples #(self.n_pc)
        z_dot = ((x*y-self.theta_samples*z)@phi.T)/self.n_samples #(self.n_pc)
        return np.concatenate([x_dot, y_dot, z_dot])

    def jacobian(self, c):
        c = c.reshape(3, self.n_pc).T
        phi = self.phi(*self.seed_rv_samples) #(self.n_pc,self.n_samples)
        x, y, z = c.T @ phi

        Jxx = (-(self.gamma_samples*phi) @ phi.T) / self.n_samples
        Jxy = ( (self.gamma_samples*phi) @ phi.T) / self.n_samples
        Jxz = np.zeros((self.n_pc,self.n_pc))

        Jyx = (((self.rho_samples - z)*phi) @ phi.T) / self.n_samples
        Jyy = -phi@phi.T / self.n_samples
        Jyz = (-(x*phi) @ phi.T) / self.n_samples

        Jzx = ((y*phi) @ phi.T) / self.n_samples
        Jzy = ((x*phi) @ phi.T) / self.n_samples
        Jzz = ((-self.theta_samples*phi) @ phi.T) / self.n_samples
        J = np.block([
                [Jxx, Jxy, Jxz],
                [Jyx, Jyy, Jyz],
                [Jzx, Jzy, Jzz]
            ])
        return J


    def run(self, degree_pc, n_init):
        '''
        solve F(c)=0 for n_init different initializations of the nonlinear solver

        degree_pc --> {int}
        n_init --> {int}
        '''
        # assemble matrices
        self.phi, self.phi_norm = cp.generate_expansion(degree_pc, self.seed_rv, retall=True)
        self.n_pc = len(self.phi_norm)
        self.solution = np.zeros((n_init,self.n_pc,3))
        self.coeff_init = []
        count = 0
        while count<n_init:
            # generate random initializations
            self.coeff_init.append(5.*np.random.randn(self.n_pc, 3))
            # solve F(c)=0
            sol = root(self.f, self.coeff_init[-1].ravel(), method='lm', tol=1e-8, jac=self.jacobian)
            loss = np.sum(np.abs(self.f(sol.x)))
            if (loss<1e-6) and (not np.isclose(self.solution[:count],sol.x.reshape(3,self.n_pc).T).all(2).all(1).any()):
                self.solution[count] = sol.x.reshape(3,self.n_pc).T
                #print(f"{self.solution[count]}")
                count+=1
                print(f"{count=}")

        self.samples_solution = (self.solution.transpose(0,2,1) @ self.phi(*self.seed_rv_samples))

    def continuation(self, degree_pc, n_branch):
        self.solution = [[] for _ in range(n_branch)]
        self.samples_solution = [[] for _ in range(n_branch)]
        for i in range(n_branch):
            degree_pc_iter = 0
            counter = 0 # Track attempts for this branch
            self.phi, self.phi_norm = cp.generate_expansion(degree_pc_iter, self.seed_rv, retall=True)
            self.n_pc = len(self.phi_norm)
            while degree_pc_iter <= degree_pc:
                counter += 1 # Increment every try
                if degree_pc_iter == 0:
                    coeff_init = 50.*np.random.randn(1, 3)
                else:
                    tmp = np.zeros((self.n_pc, 3))
                    tmp[:len(coeff_init)] = coeff_init
                    coeff_init = tmp.T
                
                sol = root(self.f, coeff_init.ravel(), method='lm', tol=1e-8, jac=self.jacobian)
                loss = np.sum(np.abs(self.f(sol.x)))
                c = sol.x.reshape(3, self.n_pc).T
                
                # Check convergence/uniqueness
                control = not any(np.isclose(c, self.solution[j][0][0]).all() for j in range(i) if self.solution[j]) if (degree_pc_iter == 0 and i > 0) else True
                if (loss < 1e-6) and control:
                    self.solution[i].append((c, degree_pc_iter)) 
                    self.samples_solution[i].append((c.T @ self.phi(*self.seed_rv_samples)))
                    degree_pc_iter += 1
                    counter = 0 
                    coeff_init = c.copy()
                    self.phi, self.phi_norm = cp.generate_expansion(degree_pc_iter, self.seed_rv, retall=True)
                    self.n_pc = len(self.phi_norm)
                
                if counter > 50: # Break if no solution found after 50 attempts
                    print(f"Stopping branch {i}: reached max attempts.")
                    break
            print(f"count_branch={i}")

    def plot_poly(self, transparent=True):
        fig, ax = plt.subplots(1,3, figsize=(18,5))
        ax[0].set_xlabel(r'$x$')
        ax[1].set_xlabel(r'$y$')
        ax[2].set_xlabel(r'$z$')
        # plot local extrema
        for i in range(3):
            vals = self.samples_solution[:,i].ravel()
            vals = vals[np.isfinite(vals)]
            ax[i].set_xlim([vals.min()-1,vals.max()+1])
            if np.allclose(vals, vals[0]):   # tutti uguali
                ax[i].axvline(vals[0], linewidth=3)
            else:
                ax[i].hist(vals, bins="fd", density=True, alpha=0.5, edgecolor="black", linewidth=0.7)
        fig.suptitle(fr"$\gamma=\mathcal{{U}}({self.gamma.lower.item():.3g},{self.gamma.upper.item():.3g}), \rho=\mathcal{{U}}({self.rho.lower.item():.3g},{self.rho.upper.item():.3g}), \theta=\mathcal{{U}}({self.theta.lower.item():.3g},{self.theta.upper.item():.3g})$")
        # put labels and save 
        fig.savefig(f"plots/lorenz_poly_({self.gamma.lower.item():.3g},{self.gamma.upper.item():.3g})_({self.rho.lower.item():.3g},{self.rho.upper.item():.3g})_({self.theta.lower.item():.3g},{self.theta.upper.item():.3g})_{degree_pc=}_{n_init=}.pdf", bbox_inches="tight", transparent=transparent)
        plt.close(fig)

    def plot_xyz_rho(self, n_branch):
        fig, ax = plt.subplots(1,3,figsize=(20,4))
        ax[0].set_xlabel(r"$\rho$")
        ax[1].set_xlabel(r"$\rho$")
        ax[2].set_xlabel(r"$\rho$")
        ax[0].set_ylabel(r"$x$")
        ax[1].set_ylabel(r"$y$")
        ax[2].set_ylabel(r"$z$")
        x = self.seed_rv_samples.copy()
        x[0], x[2] = 0,0
        samples_solution = [
            [
                (coeffs.T @ cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*x)) 
                for (coeffs, deg) in self.solution[i]
            ] 
            for i in range(len(self.solution))
        ]
        cmap = plt.get_cmap('tab10') 
        colors = [cmap(i) for i in range(len(samples_solution[0]))]
        for i in range(len(samples_solution)):
            degree=0
            for j in range(len(samples_solution[i])):
                label = f"{degree=}" if i==0 else None
                color = "black" if i>=n_branch else colors[j]
                ax[0].plot(self.rho_samples,samples_solution[i][j][0],'o',markersize=2.5,label=label, color=color)
                ax[1].plot(self.rho_samples,samples_solution[i][j][1],'o',markersize=2.5, color=color)
                ax[2].plot(self.rho_samples,samples_solution[i][j][2],'o',markersize=2.5, color=color)
                degree+=1
        fig.legend()
        fig.savefig(f"plots/lorenz_BA_poly_({self.gamma.lower.item():.3g},{self.gamma.upper.item():.3g})_({self.rho.lower.item():.3g},{self.rho.upper.item():.3g})_({self.theta.lower.item():.3g},{self.theta.upper.item():.3g})_{degree_pc=}_{n_init=}.pdf", bbox_inches="tight")
        plt.close(fig)
        
    def plot_xyz_rho_2(self, n_branch):
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        ax[0].set_xlabel(r"$\rho$")
        ax[1].set_xlabel(r"$\rho$")
        ax[2].set_xlabel(r"$\rho$")
        ax[0].set_ylabel(r"$x$")
        ax[1].set_ylabel(r"$y$")
        ax[2].set_ylabel(r"$z$")
        
        xi_grid = np.linspace(-np.sqrt(3), np.sqrt(3), 500)
        grid_eval = np.vstack([np.zeros_like(xi_grid), xi_grid, np.zeros_like(xi_grid)])
        rho_grid = xi_grid * cp.Std(self.rho) + cp.E(self.rho)
        
        theta_mean = cp.E(self.theta)
        
        for i in range(3):
            ax[i].plot(rho_grid, np.zeros_like(rho_grid), 'k', linewidth=4.0, zorder=1, label=r'$\bar{u}$' if i==0 else None)
            
        rho_valid = rho_grid[rho_grid >= 1.0]
        x_branch = np.sqrt(theta_mean * (rho_valid - 1.0))
        z_branch = rho_valid - 1.0
        
        ax[0].plot(rho_valid, x_branch, 'k', linewidth=4.0, zorder=1)
        ax[0].plot(rho_valid, -x_branch, 'k', linewidth=4.0, zorder=1)
        ax[1].plot(rho_valid, x_branch, 'k', linewidth=4.0, zorder=1) # y = x for exact branches
        ax[1].plot(rho_valid, -x_branch, 'k', linewidth=4.0, zorder=1)
        ax[2].plot(rho_valid, z_branch, 'k', linewidth=4.0, zorder=1)
        
        max_deg = max([deg for branch in self.solution for (_, deg) in branch]) if self.solution[0] else 0
        colors = plt.cm.coolwarm(np.linspace(0, 1, max_deg + 1))
        
        for i in range(min(n_branch, len(self.solution))):
            for j in range(len(self.solution[i])):
                coeffs, deg = self.solution[i][j]
                
                phi_eval = cp.generate_expansion(deg, self.seed_rv, retall=True)[0](*grid_eval)
                approx = coeffs.T @ phi_eval
                
                if deg == max_deg:
                    label = rf'$u_{{{deg}}}$' if i == 0 else None
                    ax[0].plot(rho_grid, approx[0], color=colors[deg], linewidth=2.0, zorder=5, linestyle='--',
                     marker='o', markersize=6, markevery=30, label=label)
                    ax[1].plot(rho_grid, approx[1], color=colors[deg], linewidth=2.0, zorder=5, linestyle='--',
                     marker='o', markersize=6, markevery=30)
                    ax[2].plot(rho_grid, approx[2], color=colors[deg], linewidth=2.0, zorder=5, linestyle='--',
                     marker='o', markersize=6, markevery=30)
                else:
                    ax[0].plot(rho_grid, approx[0], color=colors[deg], linewidth=1, linestyle='--', alpha=0.6, zorder=2)
                    ax[1].plot(rho_grid, approx[1], color=colors[deg], linewidth=1, linestyle='--', alpha=0.6, zorder=2)
                    ax[2].plot(rho_grid, approx[2], color=colors[deg], linewidth=1, linestyle='--', alpha=0.6, zorder=2)

        for i in range(3):
            ax[i].grid(True, alpha=0.3)
            
        ax[0].legend()
        fig.tight_layout()
        fig.savefig(f"plots/lorenz_poly_({self.gamma.lower.item():.3g},{self.gamma.upper.item():.3g})_({self.rho.lower.item():.3g},{self.rho.upper.item():.3g})_({self.theta.lower.item():.3g},{self.theta.upper.item():.3g})_{degree_pc=}_{n_init=}.pdf", bbox_inches="tight")
        plt.close(fig)

degree_pc=6
n_init=1
n_branch_to_approximate = 3

model = Lorenz(gamma=cp.Uniform(10.,10.*1.00001), 
                rho=cp.Uniform(1,2),
                theta=cp.Uniform(2.66666, 2.66667),
                seed_rv=cp.J(cp.Uniform(-np.sqrt(3),np.sqrt(3)),cp.Uniform(-np.sqrt(3),np.sqrt(3)),cp.Uniform(-np.sqrt(3),np.sqrt(3))), 
                n_samples=1000)


RUN_RANDOM_INIT = False 
RUN_CONTINUATION = True

if RUN_RANDOM_INIT:
    print("\n=== Executing Random Initialization ===")
    model.run(degree_pc=degree_pc, n_init=n_init)
    run_results_formatted = [[(sol, degree_pc)] for sol in model.solution]
    model.solution = run_results_formatted
    model.plot_xyz_rho_2(n_branch=n_init) 
    # model.plot_poly(transparent=False)

if RUN_CONTINUATION:
    print("\n=== Executing Degree Continuation ===")
    model.continuation(degree_pc=degree_pc, n_branch=n_branch_to_approximate)
    model.plot_xyz_rho_2(n_branch=n_branch_to_approximate)