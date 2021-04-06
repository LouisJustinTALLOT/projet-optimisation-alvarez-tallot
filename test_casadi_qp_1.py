import casadi

# nos données
L = 10
N = 60
K = 10
gamma = 3

# on en déduit alors
dx = L/N

# on lance une optimisation avec CasADi
opti = casadi.Opti()
# nombre de variables : 
n = 2 * N + 2
# notre variable d'optimisation dans CasADi, avec nos notations : 
z = opti.variable(n)
# notre fonction à minimiser : 
f = 0
# on met les y d'abord : 
for i in range(N+1):
    f += dx**2 *z[i]**2

f += K * (z[-1] - L/2)**2

# on veut minimiser
opti.minimize(f)
# on rajoute les contraintes
opti.subject_to(z[0] = 0)

