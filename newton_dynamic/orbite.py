import numpy as np
import numba as nb
import matplotlib.pyplot as plt


G = 6.6741e-11  
M = 1.989e30  
au = 149597870700  

# Warunki początkowe
x0 = 0  
y0 = 0.586 * au  
vx0 = 54600  
vy0 = 0  

# Czas symulacji
t_1 = 0  
t_2 = 10 * 365.25 * 24 * 3600  # 10 lat w sekundach
dt = 3600  # krok czasowy - 1 godzina



@nb.jit
def orbita(x0, y0, vx0, vy0, t_1, t_2, dt):
    # Inicjalizacja tablic do przechowywania wyników
    n = int((t_2 - t_1) / dt) + 1
    t_wartosci = np.linspace(t_1, t_2, n)
    x_wartosci = np.zeros(n)
    y_wartosci = np.zeros(n)
    vx_wartosci = np.zeros(n)
    vy_wartosci = np.zeros(n)

    # Warunki początkowe
    x = x0
    y = y0
    vx = vx0
    vy = vy0

    # Obliczenia numeryczne z wykorzystaniem równań ruchu
    for i, t in enumerate(t_wartosci):
        r = np.sqrt(x**2 + y**2)
        ax = -G * M * x / r**3
        ay = -G * M * y / r**3

        # Aktualizacja położenia i prędkości zgodnie z równaniami ruchu
        x += vx * dt
        y += vy * dt
        vx += ax * dt
        vy += ay * dt

        # Zapisanie wyników
        x_wartosci[i] = x
        y_wartosci[i] = y
        vx_wartosci[i] = vx
        vy_wartosci[i] = vy

    return t_wartosci, x_wartosci, y_wartosci, vx_wartosci, vy_wartosci

# Wywołanie funkcji i uzyskanie wyników
t_wartosci, x_wartosci, y_wartosci, vx_wartosci, vy_wartosci = orbita(x0, y0, vx0, vy0, t_1, t_2, dt)

print(t_wartosci)

# Rysowanie trajektorii orbity
plt.figure(figsize=(10, 5))
plt.plot(x_wartosci, y_wartosci)
plt.title('Trajektoria orbity komety Halleya')
plt.xlabel('Pozycja x [m]')
plt.ylabel('Pozycja y [m]')
plt.grid(True)
plt.show()

# Rysowanie zależności pozycji od czasu
plt.figure(figsize=(10, 5))
plt.plot(t_wartosci, x_wartosci, label='Pozycja x')
plt.plot(t_wartosci, y_wartosci, label='Pozycja y')
plt.title('Zależność pozycji od czasu')
plt.xlabel('Czas [s]')
plt.ylabel('Pozycja [m]')
plt.legend()
plt.grid(True)
plt.show()
