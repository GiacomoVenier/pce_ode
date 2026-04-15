import chaospy as cp
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(42)

plt.rcParams.update({
    'font.size': 16,
    'text.usetex': True,
    'font.family': 'lmodern'
})

# ==========================================
# Impostazioni del Sistema
# ==========================================
degree = 2
z_dist = cp.Uniform(-1, 1)
phi, phi_norm = cp.generate_expansion(degree, z_dist, retall=True)
n_pc = len(phi_norm)
n_dim = 2
R_rot = 5.0  # Fattore di rotazione (viola la coercività)

z_samples = z_dist.sample(2000)
phi_eval = phi(z_samples)

def F_galerkin(c):
    """Calcola il residuo standard di Galerkin P_N[F(u_N)]"""
    c_mat = c.reshape(n_dim, n_pc).T
    x, y = c_mat.T @ phi_eval
    
    # Campo vettoriale patologico (unica radice reale, ma fortemente rotante)
    Fx = x + R_rot * y**3 - z_samples
    Fy = y - R_rot * x**3
    
    x_dot = (Fx @ phi_eval.T) / len(z_samples)
    y_dot = (Fy @ phi_eval.T) / len(z_samples)
    return np.concatenate([x_dot, y_dot])

# ==========================================
# Implementazione della Deflazione
# ==========================================
found_roots = []

def deflated_objective(c):
    """
    Funzione obiettivo deflazionata: ||F(c)|| * Penalità
    Se il solutore si avvicina a una radice già nota, la penalità esplode,
    costringendolo a cercare altrove.
    """
    res = F_galerkin(c)
    norm_res = np.linalg.norm(res)
    
    penalty = 1.0
    for root in found_roots:
        dist = np.linalg.norm(c - root)
        if dist < 1e-4:
            return 1e6  # Barriera infinita
        # Formula di deflazione standard usata in letteratura
        penalty *= (1.0 + 1.0 / (dist**2))
        
    return norm_res * penalty

print("Avvio ricerca radici con Deflated Newton Method...")

# Effettuiamo diversi tentativi costringendo il solutore a "saltare" le radici note
for attempt in range(15):
    c_init = 2.0 * np.random.randn(n_pc * n_dim)
    
    # Usiamo minimize (BFGS) sull'obiettivo deflazionato invece del root-finding standard
    sol = minimize(deflated_objective, c_init, method='BFGS', 
                   options={'gtol': 1e-6, 'maxiter': 1000})
    
    # Verifichiamo se il residuo originale (non deflazionato) è davvero zero
    true_res = np.linalg.norm(F_galerkin(sol.x))
    
    if true_res < 1e-4:
        # Controlliamo che non sia numericamente identica a una già trovata
        is_new = True
        for r in found_roots:
            if np.linalg.norm(sol.x - r) < 1e-2:
                is_new = False
                break
        
        if is_new:
            found_roots.append(sol.x)
            print(f"Tentativo {attempt+1}: Trovata NUOVA radice spuria! Residuo: {true_res:.2e}")

print(f"\nRisultato finale: Trovate {len(found_roots)} radici polinomiali distinte!")

# ==========================================
# Plot dei risultati
# ==========================================
z_plot = np.linspace(-1, 1, 200)
phi_plot = phi(z_plot)

fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

for idx, coeffs in enumerate(found_roots):
    c_mat = coeffs.reshape(n_dim, n_pc).T
    u_eval = c_mat.T @ phi_plot
    
    # La prima radice trovata (generalmente la più stabile) in blu, le spurie in rosso
    if idx == 0:
        ax1.plot(z_plot, u_eval[0], 'b-', linewidth=3, label="Ramo Principale (Fisico)")
    else:
        ax1.plot(z_plot, u_eval[0], 'r--', alpha=0.7, label="Ramo Spuria (Bézout)" if idx==1 else "")

ax1.set_title("Stato $x(z)$ svelato dalla Deflazione")
ax1.set_xlabel("Parametro stocastico $z$")
ax1.set_ylabel("Coefficiente $x$")
ax1.grid(True, alpha=0.3)
ax1.legend()
plt.tight_layout()
plt.show()