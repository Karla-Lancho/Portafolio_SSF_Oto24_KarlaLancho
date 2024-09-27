# -*- coding: utf-8 -*-
"""

@author: Karla Edith

1. Implementar y ejecutar el código de Example 4.1.3, y generar la gráfica 4.1 (del Ayars)
"""
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
m = 1.0  # masa en kg
g = 9.8  # gravedad en m/s^2
k = 10.0  # constante del resorte en N/m
x0 = 0.5  # posición inicial en metros
v0 = 0.0  # velocidad inicial en m/s
t0 = 0.0  # tiempo inicial
tf = 10.0  # tiempo final
dt = 0.01  # paso de tiempo

# Definir las funciones para el método RK4
def acceleration(x):
    return (k/m)*x - g

def rk4(x, v, dt):
    k1x = v
    k1v = acceleration(x)
    
    k2x = v + 0.5*dt*k1v
    k2v = acceleration(x + 0.5*dt*k1x)
    
    k3x = v + 0.5*dt*k2v
    k3v = acceleration(x + 0.5*dt*k2x)
    
    k4x = v + dt*k3v
    k4v = acceleration(x + dt*k3x)
    
    x_new = x + (dt/6)*(k1x + 2*k2x + 2*k3x + k4x)
    v_new = v + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v)
    
    return x_new, v_new

# Listas para almacenar los resultados
t_values = np.arange(t0, tf, dt)
x_values = []
v_values = []

# Condiciones iniciales
x = x0
v = v0

# Bucle del método RK4
for t in t_values:
    x_values.append(x)
    v_values.append(v)
    x, v = rk4(x, v, dt)

# Graficar la posición
plt.plot(t_values, x_values)
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición (m)')
plt.title('Movimiento de una masa oscilando en un resorte (Ejemplo 4.1.3)')
plt.grid(True)
plt.show()

"""
 2. Implementar y ejecutar el código del Example 4.4.1 (Ayars), modificar k (3 valores cercanos a 0, 3 del orden de 1-10, 3 mayores a 50) y comparar los resultados.
 """
import numpy as np
import matplotlib.pyplot as plt

# Definir la ecuación diferencial
def derivs(y, t, k):
    return -k * y

# Implementación del método Runge-Kutta de segundo orden
def rk2(y, t, dt, k):
    k0 = dt * derivs(y, t, k)
    k1 = dt * derivs(y + k0, t + dt, k)
    y_next = y + 0.5 * (k0 + k1)
    return y_next

# Parámetros iniciales
y0 = 1.0  # valor inicial
t0 = 0.0  # tiempo inicial
tf = 10.0  # tiempo final
dt = 0.01  # paso de tiempo
t_values = np.arange(t0, tf, dt)

# Tres valores de k: cercanos a 0, en el rango de 1-10 y mayores a 50
k_values = [0.01, 0.05, 0.1, 1, 5, 10, 50, 100, 200]

# Graficar resultados para cada valor de k
plt.figure(figsize=(10, 6))
for k in k_values:
    y_values = []
    y = y0
    for t in t_values:
        y_values.append(y)
        y = rk2(y, t, dt, k)
    
    plt.plot(t_values, y_values, label=f'k = {k}')

# Etiquetas y leyenda
plt.xlabel('Tiempo (s)')
plt.ylabel('Valor de y')
plt.title('Solución Numérica con Runge-Kutta de Segundo Orden para Diferentes k')
plt.legend()
plt.grid(True)
plt.show()

"""
 3. Implementar y ejecutar el código de Example 4.5.2 (Ayars), modificar k como el punto 2. 
 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Definir la ecuación diferencial
def pendulum_derivs(state, time, g, L, b, beta, omega):
    theta, omega_theta = state
    g1 = omega_theta
    g2 = -g/L * np.sin(theta) - b * omega_theta + beta * np.cos(omega * time)
    return [g1, g2]

# Parámetros del sistema
g = 9.81  # gravedad
L = 1.0   # longitud del péndulo
beta = 0.5  # amplitud de la fuerza externa
omega_ext = 1.0  # frecuencia angular de la fuerza externa
theta0 = [np.pi/4, 0.0]  # condición inicial (ángulo, velocidad angular)
t = np.linspace(0, 10, 1000)  # tiempo

# Tres valores de b (amortiguamiento): cercano a 0, en el rango de 1-10, y mayores a 50
b_values = [0.01, 0.5, 1, 5, 10, 50]

# Graficar los resultados
plt.figure(figsize=(10, 6))
for b in b_values:
    sol = odeint(pendulum_derivs, theta0, t, args=(g, L, b, beta, omega_ext))
    plt.plot(t, sol[:, 0], label=f'b = {b}')  # Graficar theta en función del tiempo

# Etiquetas y leyenda
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (rad)')
plt.title('Péndulo Amortiguado y Forzado con Diferentes Valores de Amortiguamiento (b)')
plt.legend()
plt.grid(True)
plt.show()


