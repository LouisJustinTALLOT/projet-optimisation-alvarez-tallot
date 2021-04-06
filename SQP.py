import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt



L = 10
K = 10
N = 60
g = 3
dx_carre = L/N



G = np.zeros((2*(N+1),2*(N+1)))
for i in range(N):
    G[i][i] = 2*dx_carre
G[-1][-1] = 2*K


d = np.zeros(2*(N+1))
d[-1] = -K*L

def f(z):
    a = 0.5*np.matmul(z.T, np.matmul(G, z))
    b = np.matmul(z.T, d)
    c = K*(L/2)**2
    return a + b + c

#print(f(np.zeros((2*(N+1)))))

def c_eq(z):
    L = [z[0], z[N], z[N+1]]
    for i in range(N):
        L.append((z[N+2+i] - z[N+1+i])**2 + (z[1+i] - z[i])**2 - dx_carre)
    return L


#print(c_eq(np.ones((2*(N+1)))))

def c_ineq(z):
    L = [z[N+1] - z[N+2]]
    for i in range(1, N):
        L.append(z[N+1+i] - z[N+2+i])
        L.append((z[N+2+i] - 2*z[N+1+i] + z[N+i])**2 + (z[1+i] - 2*z[i] + z[i-1])**2 - (dx_carre**2)*(g**2))
    L = np.array(L)
    return (-1)*L     #contraintes doivent etre positives

#print(c_ineq(np.ones((2*(N+1)))))

cons = [{'type':'eq', 'fun':c_eq}, {'type':'ineq', 'fun':c_ineq}]
result = optimize.minimize(f, np.zeros((2*(N+1))), method='SLSQP', constraints=cons).x
print(result)

X = [result[i] for i in range(N+1, 2*N+2)]
Y = [result[i] for i in range(0, N+1)]
plt.plot(X, Y)
plt.show()
#method='SLSQP', constraints=[dict('type' : 'eq',  'fun': c_eq, ), dict('type' : 'ineq', 'fun' : c_ineq)])