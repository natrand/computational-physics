import numpy as np
import numba as nb
import matplotlib.pyplot as plt

N = 101  
dx = 0.01  
dt = 0.005 
c = 1.0  
u0 = np.exp(-100 * (np.linspace(0, 1, N) - 0.5) ** 2)
v0 = np.zeros(N)
beta_wartosci = [0.5, 2, 4]  


@nb.jit
def drgania_tlumione(u0, v0, N, dx, dt, c, beta, kroki_czasowe):
    u = np.zeros((kroki_czasowe, N))
    u[0] = u0
    u[1] = u0 + v0 * dt  
    v = np.zeros(N)
    
    for t in range(1, kroki_czasowe - 1):
        for x in range(1, N - 1):
            v[x] = (u[t, x] - u[t-1, x]) / dt
            a = (u[t, x+1] + u[t, x-1] - 2*u[t, x]) / dx**2 - 2 * beta * v[x]
            u[t+1, x] = 2*u[t, x] - u[t-1, x] + dt**2 * a
        
        u[t+1, 0] = u[t+1, N-1] = 0
    
    return u

u_wartosci = []
for beta in beta_wartosci:
    u_wartosci.append(drgania_tlumione(u0, v0, N, dx, dt, c, beta, 1000))

plt.figure(figsize=(15, 5))

for i, beta in enumerate(beta_wartosci):
    plt.subplot(1, len(beta_wartosci), i+1)
    plt.imshow(u_wartosci[i], extent=[0, 1, 0, 1000*dt], aspect='auto', cmap='jet', origin='lower')
    plt.colorbar(label='u(x, t)')
    plt.title(f'beta = {beta}')
    plt.xlabel('x [m]')
    plt.ylabel('t [s]')

plt.tight_layout()
plt.show()
