import sympy as sp
import numpy as np
from sympy.physics.wigner import wigner_3j
from scipy.optimize import root
from numpy.polynomial.legendre import Legendre
import matplotlib.pyplot as plt
import math
import copy
from tqdm import tqdm


# plt.rcParams['text.latex.preamble']=r"\usepackage{lmodern}"
# plt.rcParams['text.usetex'] = False
# plt.rcParams['font.size'] = 20
# plt.rcParams['font.family'] = 'lmodern'


def solve_groebner_triangular_system(groebner_basis, variables, real_only=True):
    """
    Risolve un sistema triangolare di equazioni polinomiali usando sostituzione ricorsiva.
    
    Args:
        groebner_basis: Lista di polinomi in forma triangolare (da Triangular_Groebner)
        variables: Lista ordinata delle variabili
        real_only: Se True, filtra solo soluzioni reali
        
    Returns:
        Lista di dizionari {variabile: valore}
    """
    if not groebner_basis:
        return [{}]  # Nessuna equazione => soluzione vuota (valida)

    current_poly = groebner_basis[0]
    vars_in_poly = current_poly.free_symbols

    if not vars_in_poly:
        # Polinomio costante: se non nullo, il sistema è inconsistente
        return [] if current_poly != 0 else solve_groebner_triangular_system(groebner_basis[1:], variables)

    # Trova la variabile da risolvere (ultima tra le variabili presenti)
    current_var = [v for v in variables if v in vars_in_poly][-1]

    # Trova tutte le radici (simboliche se possibile)
    poly = sp.Poly(current_poly, current_var)
    roots = poly.nroots(maxsteps=500)

    # Filtra soluzioni reali se richiesto
    solutions = []
    for r in roots:
        r_val = r.evalf() if hasattr(r, 'evalf') else r
        if not real_only or sp.im(r_val) == 0:
            solutions.append(sp.re(r_val))

    if not solutions:
        return []

    remaining_basis = groebner_basis[1:]
    all_solutions = []

    for sol_val in solutions:
        substituted_basis = [poly.subs(current_var, sol_val) for poly in remaining_basis]

        # Verifica se c'è una contraddizione immediata (es. 1 == 0)
        inconsistent = any(p.is_number and not sp.simplify(p).is_zero for p in substituted_basis)
        if inconsistent:
            continue

        partial_solutions = solve_groebner_triangular_system(substituted_basis, variables, real_only)
        for psol in partial_solutions:
            sol_dict = {current_var: sol_val}
            sol_dict.update(psol)
            all_solutions.append(sol_dict)

    return all_solutions


def Triangular_Groebner(F, variables, order='grlex'):
    """
    Calcola una base di Gröbner e la trasforma in forma triangolare (tipo "a scalini").
    
    Args:
        F: Lista di polinomi
        variables: Lista ordinata delle variabili
        order: Ordinamento monomiale iniziale (default: 'grlex')
    
    Returns:
        Lista di polinomi in forma triangolare e monica
    """
    G = sp.groebner(F, *variables, order=order)
    triangular = list(G.fglm('lex'))[::-1]  # Reverse: variabili "alte" per prime

    # Rende ciascun polinomio monico
    monic_triangular = []
    for poly in triangular:
        p = sp.Poly(poly, *variables)
        monic = p / p.LC()
        monic_triangular.append(monic.as_expr())

    return monic_triangular


def Build_matrices(N_PC):
    """Uses Wigner 3-j symbols to build the system's structure, does not include mu and sigma"""
    N = N_PC + 1

    Variables = [sp.symbols(f'c{i}') for i in range(N)]
    Variables = np.array(Variables)

    # Construct matrices using Wigner 3j symbols
    matrices = []
    for i in range(N):
        A = [[0 for _ in range(N)] for _ in range(N)]
        for j in range(N):
            for k in range(N):
                symbol = sp.physics.wigner.wigner_3j(i, j, k, 0, 0, 0)
                A[j][k] = sp.Rational(2*(symbol**2))
        A = np.matrix(A)
        matrices.append(A)

    # Build polynomial system F = V^T M V for each matrix
    F = [(Variables @ matrices[i] @ Variables.T).item() for i in range(N)]

    return matrices, F, Variables


def Build_system(H, mu, s, Variables):
    """Crea il sistema quadratico associato alla biforcazione transcritica con espansione Uniform PC.
    mu è il valore atteso e s = sqrt(3)*sigma, dove sigma^2 è la varianza."""
    N = len(Variables)-1
    G = copy.deepcopy(H)

    # Add constraints
    
    G[0] -= 2*mu*Variables[0]+sp.Rational(2,3)*s*Variables[1]      # Constraint for c0
    G[0] = G[0].expand()
    for i in range(1,N):
        G[i] -= sp.Rational(2,2*i+1)*mu*Variables[i]+sp.Rational(2*s*(i+1),(2*i+1)*(2*i+3))*Variables[i+1]+sp.Rational(2*s*i,(2*i-1)*(2*i+1))*Variables[i-1]
        G[i] = G[i].expand()
    G[N] -= sp.Rational(2,2*N+1)*mu*Variables[N]+sp.Rational(2*s*N,(2*N-1)*(2*N+1))*Variables[N-1] # Constraint for c1
    G[N] = G[N].expand()
    return G


def Matrices_Legendre(N):
    """Calculates A^{(k)}, when N_PC = N, in the case of Uniform distribution, for every k, without using wigner-3j symbols"""

    A = [[[0 for i in range(2*N+2)] for j in range(2*N+2)] for k in range(N+1)]
    
    if N>=0:
        for i in range(0,2*N+2):                    # It is necessary to calculate them up to an higher horder, since if i = N, the recurrence relation written below needs i+1 and so on
            A[0][i][i] = sp.Rational(2,2*i+1)

    if N>=1:
        for i in range(1,2*N+2):
            A[1][i-1][i] = sp.Rational(2*i,4*i**2-1)
            A[1][i][i-1] = sp.Rational(2*i,4*i**2-1)

    for k in range(2,N+1):
        for i in range (0, 2*(N+1)-k):
            for j in range(i, 2*(N+1)-k):
                if i==N:
                    temp = sp.Rational((2*k-1)*(i+1),k*(2*i+1))*A[k-1][i+1][j]+sp.Rational((2*k-1)*i,k*(2*i+1))*A[k-1][i-1][j]-sp.Rational(k-1,k)*A[k-2][i][j]
                else:
                    temp = sp.Rational((2*k-1)*(i+1),k*(2*i+1))*A[k-1][i+1][j]+sp.Rational((2*k-1)*i,k*(2*i+1))*A[k-1][i-1][j]-sp.Rational(k-1,k)*A[k-2][i][j]
                A[k][i][j] = temp
                A[k][j][i] = temp
    B = [[[A[k][i][j] for i in range(0,N+1)]for j in range(0,N+1)]for k in range(0,N+1)]
    for k in range(0,N+1):
        B[k]=(np.array(B[k]))
    return B


def System_solver(F, Variables, N):
    """Solves the system associated to the bifurcation, by expoliting Groebner basis method. Returns both solutions as dictionary and as vector"""
    
    B = Triangular_Groebner(F, Variables)

    # Solve the triangular system
    solutions = solve_groebner_triangular_system(B, Variables)

    # Reorder solutions by variable index (x0, x1, x2, x3)
    ordered_symbols = [sp.symbols(f'c{i}') for i in range(N+1)]
    ordered_solutions = []

    for sol in solutions:
        ordered_sol = {f'c{i}': sol[symbol] for i, symbol in enumerate(ordered_symbols)}
        ordered_solutions.append(ordered_sol)
    
    solutions_vector = []

    for i, sol in enumerate(ordered_solutions):
        vec = [complex(sol[f'c{i}'].evalf()) for i in range(N+1)]
        for l in range(len(vec)):
            if vec[l].imag == 0:
                vec[l] = vec[l].real
        solutions_vector.append(vec)

    return ordered_solutions, solutions_vector

def Print_solutions(solutions):
    """Takes the dictionary of solutions as an input and prints it"""
    
    for i, sol in enumerate(solutions):
        print(f"Solution {i+1}:")
        print(sol)
        print("-"*50)
    return

def count_real_solutions(solutions_vector):
    """Counts the number of real solutions of the system"""
    # Inizializziamo i contatori
    number_of_solutions = len(solutions_vector)
    number_of_real_solutions = 0
    real_solutions = []
    # Cicliamo attraverso tutte le soluzioni
    for i in range(number_of_solutions):
        real = True
        for j in range(len(solutions_vector[i])):
            solution = solutions_vector[i][j]
            if solution.imag != 0:
                real = False
                break
        if real:
            number_of_real_solutions += 1
            real_solutions.append(solutions_vector[i])
                
    return number_of_real_solutions, real_solutions


def Numerical_solutions(Variables, F, solutions_vector):
    """ Numerical verification of the solutions obtained by groebner basis, using scipy.optimize.root"""

    variables_tuple = tuple(Variables)
    G = sp.lambdify(variables_tuple, F, "numpy")

    def func(vals):
        """Wrapper function for numerical root finding"""
        return G(*vals)
    
    updated_solutions = []
    residuals = []
    
    for groebner_sol in solutions_vector:
        sol = dict(zip(Variables, groebner_sol))
        res = 0
        for eq in F:
            res += abs(eq.subs(sol).evalf())
        if res<=10**2:
            numerical_sol = root(func, groebner_sol)
            
            updated_solutions.append(numerical_sol.x)
            residuals.append(np.linalg.norm(numerical_sol.fun))
    
    return updated_solutions, residuals


def multiple_systems_solver(mu, sigma, N_PC, number_of_solutions = False):
    """Solves the system for multiple values of mu, sigma, and returns the solutions"""
    
    solutions = []

    matrices, F, Variables = Build_matrices(N_PC)

    for i in range(0, len(mu)):
        for j in range(0, len(sigma)):
            
            H = copy.deepcopy(F) 
            G = Build_system(H, mu[i], sigma[j], Variables)
            
            variables = Variables[::-1]
            ordered_solutions, solutions_vector = System_solver(G, variables, N_PC)
            number_real_solutions, real_solutions_vector = count_real_solutions(solutions_vector)
            updated_solutions, residuals = Numerical_solutions(Variables, G, real_solutions_vector)
            
            if number_of_solutions is True:
                print(f"mu = {mu[i]}, sigma = {sigma[j]*math.sqrt(3)} \nNumber of real solutions: {number_real_solutions}")
                print("-"*50)

            solutions.append(updated_solutions)
    
    return solutions


def Plot_of_coefficients(N, mu, sigma, solutions):
    """Plots the values of the coefficients with respect to the change of the value of mu and sigma"""
    
    mu_values = []
    
    C = [[]for i in range (0,N+1)]

    for i in range (len(mu)):
        for j in range(len(solutions[i])):
            if solutions[i]!=[]:
                mu_values.append(mu[i])
                for k in range(0,N+1):
                    C[k].append(solutions[i][j][k])
    

    for k in range(0,N+1):
        fig, ax = plt.subplots(1,1, figsize=(10.5,5))
        ax.scatter(mu_values, C[k], marker=".",color=plt.cm.tab10(k % 10), label=f"$c_{k}$")
        ax.set_xlabel(rf"$\mu$")
        ax.set_ylabel(rf"$c$")
        ax.grid(True)
        ax.legend(loc="upper left", ncol=1)
        fig.savefig(f"Plots/Plot_of_coefficient_N={N}_c_{k}.png")


def Legendre_polynomials(n):
    x = sp.symbols('x')
    """Generates Legendre polynomials up to degree n using recurrence relation"""
    if n == 0:
        return [1]
    if n == 1:
        return [1,x]
    basis = [1,x]
    for i in range(2,n+1):
        temp = sp.expand(sp.Rational(2*i-1,i)*basis[i-1]*x - sp.Rational(i-1,i)*basis[i-2])
        basis.append(temp)
    return basis


def Calculate_polynomials_optimized(N, mu, sigma, coefficients, residuals):
    """
    Calcola i polinomi combinati come oggetti Legendre di NumPy.
    Converte esplicitamente i coefficienti simbolici in float.
    """
    a = float(mu - sigma)
    b = float(mu + sigma)
    domain = [a, b]
    solutions = []
    
    for coef, res in zip(coefficients, residuals):
        if res <= 1e-6:
            # Converti i coefficienti SymPy in float
            coef_numeric = [float(c.evalf()) if isinstance(c, sp.Expr) else c for c in coef]
            poly = Legendre(coef_numeric, domain=domain)
            solutions.append(poly)
    return solutions
       
        

def Plot_polynomials_optimized(polynomials, mu, sigma):
    """
    Plotta i polinomi usando valutazione vettorizzata.
    """
    a = float(mu - sigma)
    b = float(mu + sigma)
    x_vals = np.linspace(a, b, 400)
    
    plt.figure()
    for poly in polynomials:
        y_vals = poly(x_vals)
        plt.plot(x_vals, y_vals)
    
    plt.plot(x_vals, x_vals, color='black', linestyle=':')
    plt.plot(x_vals, np.zeros_like(x_vals), color='black', linestyle=':')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()


def find_extrema(polynomials, mu, sigma):
    """Given a list of polynomials, finds all maximum and minimum points in the support of the distribution.
    The output is a list of the values of the polynomials evaluated at the extrema points"""
    y = sp.symbols('y')
    y_values = []
    x_values = []

    for pol in polynomials:
        derivative = sp.diff(pol, y)
        critical_points = sp.Poly(derivative, y).nroots()
        
        for val in critical_points:
            if sp.im(val) == 0:
                if val<= mu+sigma and val>=mu-sigma:
                    x_values.append(val)
                    y_values.append(pol.subs(y, val))
        
    return x_values, y_values


def multiple_systems_solver(mu, sigma, N_PC, number_of_solutions = False):
    """Solves the system for multiple values of mu, sigma, and returns the solutions"""
    
    solutions = []

    matrices, F, Variables = Build_matrices(N_PC)
    for i in range(0, len(mu)):
        for j in range(0, len(sigma)):
            H = copy.deepcopy(F) 
            G = Build_system(H, mu[i], sigma[j])
            
            ordered_solutions, solutions_vector = System_solver(G, Variables, N_PC)
            number_real_solutions, real_solutions_vector = count_real_solutions(solutions_vector)
            updated_solutions, residuals = Numerical_solutions(Variables, G, real_solutions_vector)
            
            if number_of_solutions is True:
                print(f"mu = {mu[i]}, sigma = {sigma[j]*math.sqrt(3)} \nNumber of real solutions: {number_real_solutions}")
                print("-"*50)

            solutions.append(updated_solutions)
    
    return solutions



def Numerical_system_solver(G, Variables, n_init, tol=1e-4):
    """Solve symbolic system G=0 numerically from random starts. Return unique real solutions within a given tolerance."""

    assert n_init > 0, "n_init must be positive"

    N = len(G)
    G_numeric = sp.lambdify(Variables, G, modules='numpy')

    def wrapped_func(c_array):
        try:
            result = np.array(G_numeric(*c_array), dtype=np.float64).flatten()
        except Exception:
            result = np.full(N, np.nan)  # Ensure solver fails if evaluation fails
        return result

    all_solutions = []
    unique_solutions = []
    for _ in tqdm(range(n_init), desc="Solving"):
        init_guess = 0.5 * np.random.randn(N)
        sol = root(wrapped_func, init_guess, method='hybr', tol=1e-15)
        residual = np.linalg.norm(wrapped_func(sol.x))
        if np.all(np.isfinite(sol.x)) and residual<=tol:
            if all(np.linalg.norm(sol.x - np.array(uniq)) > tol for uniq in unique_solutions):
                unique_solutions.append(sol.x)


    return unique_solutions

def Legendre_polynomials(n):
    x = sp.symbols('x')
    """Generates Legendre polynomials up to degree n using recurrence relation"""
    if n == 0:
        return [1]
    if n == 1:
        return [1,x]
    basis = [1,x]
    for i in range(2,n+1):
        temp = sp.expand(sp.Rational(2*i-1,i)*basis[i-1]*x - sp.Rational(i-1,i)*basis[i-2])
        basis.append(temp)
    return basis

def Calculate_polynomials(N,mu,sigma, coefficients, residuals):
    """Calculates all the polynomials that are a solution to the system for specific values of mu and sigma"""

    x, y = sp.symbols('x y')
    pols = np.array(Legendre_polynomials(N))
    poly=[1]
    l = sigma
    a = mu - l      
    b = mu + l

    for i in range (1,len(pols)):
        poly.append(pols[i].subs({x: y/math.sqrt(3)}))          # rescales the polynomials in [-sqrt(3),sqrt(3)] 

    # Shift the ploynomials in the support of the real distribution, and not on the basis one

    poly = np.array(poly)

    solutions = []

    for i in range(len(coefficients)):
        if residuals[i]<=10**(-6):
            sol = coefficients[i] @ poly.T
            sol = sol.subs({y: math.sqrt(3)*(y-mu)/(l)})
            solutions.append(sol.expand())

    return solutions

def Plot_polynomials(polynomials, mu, sigma):
    """Plots all the polynomials that are a solution to the system for specific values of mu and sigma"""

    x, y = sp.symbols("x y")

    l = float(sigma)
    a = float(mu - l)      
    b = float(mu + l)

    for i in range(len(polynomials)):
        x_vals = np.linspace(a, b, 400)
        f = sp.lambdify(y, polynomials[i], 'numpy')
        y_vals = f(x_vals)
        if np.isscalar(y_vals):
            y_vals = np.full_like(x_vals, y_vals)
        plt.plot(x_vals, y_vals)
        plt.xlabel('x')
        plt.ylabel('f(x)')

    plt.plot(x_vals, x_vals, color = "black", linestyle=":")
    plt.plot(x_vals, [0 for i in range (len(x_vals))], color = "black", linestyle=":", label = "Solution branches")

    plt.grid(True)
    plt.legend()
    plt.show()


def find_extrema_num(polynomials, mu, sigma):
    """Given a list of polynomials, finds all maximum and minimum points in the support of the distribution.
    The output is a list of the values of the polynomials evaluated at the extrema points"""
    y = sp.symbols('y')
    x_values = []
    pol_index = []

    for i, pol in enumerate(polynomials):
        derivative = sp.diff(pol, y)
        critical_points = sp.Poly(derivative, y).nroots(maxsteps=200)
        points = []
        b = False

        for val in critical_points:
            if sp.im(val) == 0:
                b = True
                if val<= mu+sigma and val>=mu-sigma:
                    points.append(val)
        
        if b is True:
            x_values.append(points)
            pol_index.append(i)
        
    return x_values, pol_index