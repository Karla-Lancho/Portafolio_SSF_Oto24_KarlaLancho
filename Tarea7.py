def mean(t):
    return(float(sum(t)) / len(t))

import numpy as np
print(np.mean([1,4,3,2,6,4,4,3,2,6]))  # 3.5

def var(t, mu):
    dev2 = [(x - mu)**2 for x in t]
    var = mean(dev2)
    return(var)

print(np.var([1,3,3,6,3,2,7,5,9,1]))  # 6.4

t = [1,2,2,3,1,2,3,2,1,3,3,3,3]
hist = {}
for x in t:
    hist[x] = hist.get(x,0) + 1
    n = float(len(t))

pmf = {}
for x, freq in hist.items():
    pmf[x] = freq / n

print(pmf)


from pylab import *
# Eliminar la línea de importación de tools
# from tools import euler

N = 1000
xo = 0.0
vo = 0.0
tau = 3.0
dt = tau / float(N - 1)
k = 3.5
m = 0.2
gravedad = 9.8

time = linspace(0, tau, N)

y = zeros([N, 2])
y[0, 0] = xo
y[0, 1] = vo

def euler(state, time, dt, derivs):
    """
    Implementación básica del método de Euler para resolver ecuaciones diferenciales ordinarias.
    """
    return state + derivs(state, time) * dt

def SHO(state, time):
    g0 = state[1]
    g1 = -k/m * state[0] - gravedad
    return array([g0, g1])

for j in range(N - 1):
    y[j + 1] = euler(y[j], time[j], dt, SHO)

xdata = [y[j, 0] for j in range(N)]
vdata = [y[j, 1] for j in range(N)]

plot(time, xdata, label="Posición")
plot(time, vdata, label="Velocidad")
xlabel("Tiempo")
ylabel("Posición, Velocidad")
legend()
show()

from pylab import *
import numpy as np

def euler(state, time, dt, derivs):
    """
    Implementación básica del método de Euler para resolver ecuaciones diferenciales ordinarias.
    """
    return state + derivs(state, time) * dt

def rk2(y, time, dt, derivs):
    """
    Esta función avanza el valor de 'y' un paso hacia adelante con un solo
    paso de tamaño 'dt' utilizando el método de Runge-Kutta de segundo orden.
    """
    k0 = dt * derivs(y, time)
    k1 = dt * derivs(y + 0.5 * k0, time + dt)
    y_next = y + k1
    return y_next

def SHO(state, time, k):
    """
    Ecuaciones diferenciales para el oscilador armónico simple.
    """
    g0 = state[1]
    g1 = -k * state[0]
    return np.array([g0, g1])

def run_simulation(k_values):
    """
    Ejecuta la simulación para diferentes valores de k y compara los resultados.
    """
    N = 1000
    xo = 0.0
    vo = 1.0
    tau = 10.0
    dt = tau / float(N - 1)
    gravedad = 9.8
    time = np.linspace(0, tau, N)

    plt.figure(figsize=(12, 8))

    for k in k_values:
        y_euler = np.zeros([N, 2])
        y_rk2 = np.zeros([N, 2])

        y_euler[0, 0] = xo
        y_euler[0, 1] = vo

        y_rk2[0, 0] = xo
        y_rk2[0, 1] = vo

        for j in range(N - 1):
            y_euler[j + 1] = euler(y_euler[j], time[j], dt, lambda y, t: SHO(y, t, k))
            y_rk2[j + 1] = rk2(y_rk2[j], time[j], dt, lambda y, t: SHO(y, t, k))

        plt.plot(time, y_euler[:, 0], label=f'Euler (k={k})')
        plt.plot(time, y_rk2[:, 0], '--', label=f'RK2 (k={k})')

    plt.title('Simulación del Oscilador Armónico Simple para diferentes valores de k')
    plt.xlabel('Tiempo')
    plt.ylabel('Posición')
    plt.legend()
    plt.grid(True)
    plt.show()

# Valores de k a probar
k_values = [0.1, 1, 10, 50, 100, 500, 1000]

# Ejecutar la simulación y comparación
run_simulation(k_values)