import chaospy as cp
import numpy as np
from scipy.optimize import root
from scipy.stats import gaussian_kde, wasserstein_distance
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as patches

#np.random.seed(241)

# matplotlib.pyplot options
plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'lmodern'

#-------------------------------------------------------------------------------------------------------------------------
#Code solving u(u^2-mu)=0 (*)

#mu --> parameter --(KL)--> mu = \mathbb{E}[\mu] + \sigma\xi
#\sigma --> standard deviation of \mu
#\xi --(centered and rescaled \mu)--> \mu centered, i.e. \mathbb{E}[\xi]=0, and rescaled, i.e. \mathbb{E}[\xi^2]=1

#u --> solution --(PC)--> u=\sum_{i=1}^{N_{PC}} c_i \phi_i(\xi) 

#We will refer to (*) as F(c)=0
#-------------------------------------------------------------------------------------------------------------------------

class Saddle_Node():
    def __init__(self, mu, base_mu, n_samples=1000):
        '''
        mu --> {chaospy.Distribution}
        base_mu --> {chaospy.Distribution}
        n_samples --> {int}
        '''
        self.mu = mu
        self.base_mu = base_mu
        self.n_samples = n_samples

        # collect samples from mu and base_mu
        self.mu_base_samples = np.sort(self.base_mu.sample(n_samples))
        self.mu_samples = self.mu_base_samples*cp.Std(self.mu) + cp.E(self.mu)

    def f(self, c):
        '''
        compute F(c)
        '''
        c = c.reshape((self.n_pc,1))
        assert c.shape==(self.n_pc,1), f"c must be of shape {(self.n_pc,1)}, instead it is of shape {c.shape}"
        u = c.T @ self.phi(self.mu_base_samples)
        I = (u**2-self.mu_samples)
        II = (I @ self.phi(self.mu_base_samples).T)/self.n_samples
        return II.flatten()

    def run(self, degree_pc, n_init):
        '''
        solve F(c)=0 for n_init different initializations of the nonlinear solver

        degree_pc --> {int}
        n_init --> {int}
        '''
        # assemble matrices
        self.phi, self.phi_norm = cp.generate_expansion(degree_pc, self.base_mu, retall=True)
        self.n_pc = len(self.phi_norm)
        self.M = {0:(self.phi(self.mu_base_samples) @ self.phi(self.mu_base_samples).T)/self.n_samples,
                  1:self.phi(self.mu_base_samples),
                  2: (self.mu_samples @ self.phi(self.mu_base_samples).T)/self.n_samples}

        self.solution = np.zeros((n_init,self.n_pc))
        self.coeff_init = []
        for i in tqdm(range(n_init)):
            # generate random initializations
            self.coeff_init.append(0.5*np.random.randn(self.n_pc, 1))

            # solve F(c)=0
            sol = root(self.f, self.coeff_init[-1], method='hybr', tol=1e-8)

            # save the solution
            self.solution[i] = sol.x

        self.solution_samples = (self.solution @ self.phi(self.mu_base_samples))

    def plot_poly(self, x_mu, idx_poly, init_poly=False,transparent=True,index=1):
        fig, ax = plt.subplots(1,1, figsize=(10.5,5))
        # plot the bifurcation diagram
        idx_0 = np.argmax(x_mu>=0)
        ax.plot(x_mu[idx_0:], np.sqrt(x_mu[idx_0:]),color="red",linestyle=":",label="solution branches")
        ax.plot(x_mu[idx_0:], -np.sqrt(x_mu[idx_0:]),color="red",linestyle=":")
        
        # plot the support of \xi
        tmp = max([max(self.mu_samples)-cp.E(self.mu), cp.E(self.mu)- min(self.mu_samples)])
        rect = patches.Rectangle((cp.E(self.mu)-tmp, -2), 2*tmp, 4, color="#1f77b4", alpha=0.2, label=r"$supp_{\xi}$")
        ax.add_patch(rect)
        
        # plot polynomials
        for s in idx_poly:
            ax.plot(self.mu_samples, self.solution[s] @ self.phi(self.mu_base_samples))
        
        # put labels and save 
        ax.set_xlabel(rf"$\mu$")
        ax.set_ylabel(rf"$u$")
        ax.grid(True)
        ax.set_ylim([-2,2])
        ax.legend(loc="upper left", ncol=1)
        fig.savefig(f"Prova{index}poly_{idx_poly}.pdf", bbox_inches="tight", transparent=transparent)
        plt.close(fig)

for i in range(0,1):
    saddle_node = Saddle_Node( mu=cp.Uniform(0.8,1.2), 
                        base_mu=cp.Uniform(-np.sqrt(3),np.sqrt(3)), 
                        n_samples=1000)

    saddle_node.run(degree_pc=5, 
                    n_init=15)


    saddle_node.plot_poly(x_mu = np.linspace(-1,2,2000), 
                        idx_poly=[0,1,2,3,4,5], 
                        init_poly=False, 
                        transparent=False,index=i)