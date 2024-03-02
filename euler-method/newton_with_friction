import numpy as np
import matplotlib.pyplot as plt
import numba as nb

# Definicja potencjału V(x)
@nb.jit
def V(x):
    return -np.exp(-x**2) - 1.2 * np.exp(-(x - 2)**2)

@nb.jit
def derivative_V(x):
    delta_x = 0.001
    return (V(x + delta_x) - V(x - delta_x)) / (2 * delta_x)

@nb.jit
def integrate_euler_with_friction(x0, v0, dt, T, alpha):
    num_steps = int(T / dt)
    x = np.zeros(num_steps)
    v = np.zeros(num_steps)
    Ek = np.zeros(num_steps)
    Vx = np.zeros(num_steps)
    
    x[0] = x0
    v[0] = v0
    Ek[0] = 0.5 * v0**2
    Vx[0] = V(x0)
    # Stałe
    m = 1  # Masa
    
    for i in range(1, num_steps):
        a = -1 * derivative_V(x[i-1])/m  
        v[i] = v[i-1] + a * dt - alpha * v[i-1] * dt
        x[i] = x[i-1] + v[i] * dt
        
        Ek[i] = 0.5 * m * v[i]**2
        Vx[i] = V(x[i])
        
    return x, v, Ek, Vx

x0 = 2.8  
v0 = 0.0  
dt = 0.01  
T = 30  
alphas = [0.5, 5, 201]  


for alpha in alphas:
    x, v, Ek, Vx = integrate_euler_with_friction(x0, v0, dt, T, alpha)
    t = np.arange(0, T, dt)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(t,x)
    plt.title('x(t)')
    plt.xlabel('t [s]')
    plt.ylabel('x[m]')

    plt.subplot(2, 2, 2)
    plt.plot(t, v)
    plt.title('v(t)')
    plt.xlabel('t [s]')
    plt.ylabel('v [m/s]')

    plt.subplot(2, 2, 3)
    plt.plot(t, Ek, label='Ek')
    plt.plot(t, Vx, label='Ep')
    plt.title('Ek and Ep')
    plt.xlabel('t [s]')
    plt.ylabel('E [J]')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(t, Ek + Vx)
    plt.title('E total')
    plt.xlabel('t [s]')
    plt.ylabel('E total')

    plt.tight_layout()
    plt.suptitle(f'Alpha parameter = {alpha}', fontsize=16)
    plt.show()

T_values_portret = [100, 1000]

for alpha in alphas:
    for T in T_values_portret:
        t_portret = np.arange(0, T, dt)
        x_portret, v_portret, _, _ = integrate_euler_with_friction(x0, v0, dt, T, alpha)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x_portret, v_portret)
        plt.title(f'Phase portrait for alpha = {alpha}, T = {T} s and $\Delta t$ = {dt} s')
        plt.xlabel('x(t)')
        plt.ylabel('v(t)')
        plt.grid(True)
        plt.show()

