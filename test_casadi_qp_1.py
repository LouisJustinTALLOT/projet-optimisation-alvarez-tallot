import casadi
import numpy as np
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
opti.subject_to(z[0] == 0) # y_0
opti.subject_to(z[N] == 0) # y_N
opti.subject_to(z[N+1] == 0) # x_0

for i in range(N):
    # on rajoute les contraintes inégalités
    pass



# on résoud
opti.solver('ipopt')
sol = opti.solve()
print(sol.value(z))