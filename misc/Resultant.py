import sympy as sp

# Define symbolic variables
x, y, z, m, s = sp.symbols('x y z m s')

# Define the system of polynomial equations
F1 = 30*x**2 + 10*y**2 + 6*z**2 - 15*m
F2 = 20*x*y + 8*y*z - 10*s          # here, s=sqrt(3)sigma, since it is better not to deal with irrational coefficients
F3 = 7*y**2 + 3*z**2 + 21*x*z

# Step 1: Compute resultants to eliminate variables
# Compute resultant of F1 and F2 with respect to z
R1 = sp.resultant(F1, F2, z)
# Simplify by removing common content (primitive part)
_, R1_reduced = R1.primitive()
print(f"R1 (reduced resultant of F1 and F2 wrt z):\n{R1_reduced}\n")

# Compute resultant of F1 and F3 with respect to z
R2 = sp.resultant(F1, F3, z)
_, R2_reduced = R2.primitive()
print(f"R2 (reduced resultant of F1 and F3 wrt z):\n{R2_reduced}\n")

# Step 2: Further eliminate variables
# Compute resultant of R1_reduced and R2_reduced with respect to y
R3 = sp.resultant(R1_reduced, R2_reduced, y)
_, R3_reduced = R3.primitive()
# Convert to polynomial and make monic
R3_poly = sp.Poly(R3_reduced, x, m, s)
R3_monic = R3_poly / R3_poly.LC()
print(f"R3 (monic resultant in x):\n{R3_monic}\n")

# Step 3: Solve for specific parameter values
sig = 0.2
mu = 1
# Substitute m=mu and s=sig into the polynomial
R3_evaluated = R3_monic.subs({m: mu, s: sig})
print(f"R3 evaluated at m={mu}, s={sig}:\n{R3_evaluated}\n")

# Find all roots (both real and complex)
roots = sp.solve(R3_evaluated, x)
# Extract only real roots with numerical evaluation
real_roots = [sol.evalf() for sol in roots if sol.is_real]
print(f"Real roots in x: {real_roots}\n")

# Step 4: Back-substitute to find corresponding y values
x_solutions = []
y_solutions = []

for x_root in real_roots:
    # Substitute x value and parameters into R2_reduced
    poly_evaluated = R2_reduced.subs({m: mu, s: sig, x: x_root})
    y_roots = sp.solve(poly_evaluated, y)
    
    # Filter real y roots
    real_y_roots = [root for root in y_roots if root.is_real]
    
    if real_y_roots:
        x_solutions.append(x_root)
        y_solutions.append(real_y_roots)

print(f"Corresponding real y solutions:\n{y_solutions}\n")

# Step 5: Find z values for each (x,y) pair
final_x = []
final_y = []
final_z = []

for i in range(len(x_solutions)):
    temp_z_solutions = []
    temp_y_solutions = []
    
    for y_root in y_solutions[i]:
        # Substitute into original F1 equation
        poly_evaluated = F1.subs({m: mu, s: sig, x: x_solutions[i], y: y_root})
        z_roots = sp.solve(poly_evaluated, z)
        
        # Filter real z roots
        real_z_roots = [root for root in z_roots if root.is_real]
        
        if real_z_roots:
            temp_z_solutions.append(real_z_roots)
            temp_y_solutions.append(y_root)
    
    if temp_z_solutions:
        final_x.append(x_solutions[i])
        final_y.append(temp_y_solutions)
        final_z.append(temp_z_solutions)

print(f"Corresponding real z solutions:\n{final_z}\n")

# Step 6: Verify solutions in all original equations
epsilon = 1e-6  # Numerical tolerance

print("Valid solutions that satisfy all equations:")
for i in range(len(final_x)):
    for j in range(len(final_y[i])):
        for k in range(len(final_z[i][j])):
            x_val = final_x[i]
            y_val = final_y[i][j]
            z_val = final_z[i][j][k]
            
            # Evaluate all original equations
            f1_error = abs(F1.subs({m: mu, s: sig, x: x_val, y: y_val, z: z_val}).evalf())
            f2_error = abs(F2.subs({m: mu, s: sig, x: x_val, y: y_val, z: z_val}).evalf())
            f3_error = abs(F3.subs({m: mu, s: sig, x: x_val, y: y_val, z: z_val}).evalf())
            
            # Check if all errors are within tolerance
            if all(error < epsilon for error in [f1_error, f2_error, f3_error]):
                print(f"Solution found: (x={x_val}, y={y_val}, z={z_val})")
                print(f"Equation errors: F1={f1_error:.2e}, F2={f2_error:.2e}, F3={f3_error:.2e}\n")