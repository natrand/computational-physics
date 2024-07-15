import numpy as np
import numba as nb
import matplotlib.pyplot as plt

N = 101  
dx = 0.01  
dt = 0.005  
c = 1.0  
u0 = np.exp(-100 * (np.linspace(0, 1, N) - 0.5) ** 2)
v0 = np.zeros(N)

@nb.jit
def funkcja_falowa(u0, v0, N, dx, dt, c, kroki_czasowe, warunki_brzegowe='sztywne'):
    u = np.zeros((kroki_czasowe, N))
    u[0] = u0
    u[1] = u0 + v0 * dt 
    v = np.zeros(N)
    
    for t in range(1, kroki_czasowe - 1):
        for x in range(1, N - 1):
            a = (u[t, x+1] + u[t, x-1] - 2*u[t, x]) / dx**2
            u[t+1, x] = 2*u[t, x] - u[t-1, x] + dt**2 * a
        
        if warunki_brzegowe == 'sztywne':
            u[t+1, 0] = u[t+1, N-1] = 0
        elif warunki_brzegowe == 'luzne':
            u[t+1, 0] = u[t+1, 1]
            u[t+1, N-1] = u[t+1, N-2]
    
    return u

u_sztywne = funkcja_falowa(u0, v0, N, dx, dt, c, 1000, warunki_brzegowe='sztywne')
u_luzne = funkcja_falowa(u0, v0, N, dx, dt, c, 1000, warunki_brzegowe='luzne')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(u_sztywne, extent=[0, 1, 0, 1000*dt], aspect='auto', cmap='jet', origin='lower')
plt.colorbar(label='u(x, t)')
plt.xlabel('x [m]')
plt.ylabel('t [s]')

plt.subplot(1, 2, 2)
plt.imshow(u_luzne, extent=[0, 1, 0, 1000*dt], aspect='auto', cmap='jet', origin='lower')
plt.colorbar(label='u(x, t)')
plt.xlabel('x [m]')
plt.ylabel('t [s]')

plt.tight_layout()
plt.show()
