import numpy as np
import numba as nb

@nb.jit
def V(x):
    return -np.exp(-x**2) - 1.2 * np.exp(-(x - 2)**2)

@nb.jit
def pochodna_V(x):
    delta_x = 0.001
    return (V(x + delta_x) - V(x - delta_x)) / (2 * delta_x)

@nb.jit
def F1(x, v, dt):
    return x[1] - x[0] - dt / 2 * (v[1] + v[0])

@nb.jit
def F2(x, v, dt, alfa):
    
    return x[1] - x[0] - dt / 2 * (-1 * pochodna_V(x[1]) - alfa * v[1]) - dt / 2 * (pochodna_V(x[0]) - alfa * v[0])

@nb.jit
def iteracja(x, v, dt, alfa):
    for i in range(1, len(x)):
        v[i] = v[i - 1] + dt / 2 * (-1 / m * pochodna_V(x[i]) - alfa * v[i] - 1 / m * pochodna_V(x[i - 1]) - alfa * v[i - 1])
        x[i] = x[i - 1] + dt / 2 * (v[i] + v[i - 1])

m = 1
dt = 0.01
alfa = 0
dokladnosc = 0.001
max_iteracji = 10000
n = 1000
x = np.zeros(n)
v = np.zeros(n)
x[0] = 2.8
v[0] = 0

for i in range(max_iteracji):
    iteracja(x, v, dt, alfa)
    f1 = F1(x, v, dt)
    f2 = F2(x, v, dt, alfa)
    if abs(f1) < dokladnosc and abs(f2) < dokladnosc:
        break

#x i v po 1 kroku czasowym
print("x po pierwszym kroku czasowym:", x[1])
print("v po pierwszym kroku czasowym:", v[1])
print("F1 = ",f1)
print("F2 = ",f2)
