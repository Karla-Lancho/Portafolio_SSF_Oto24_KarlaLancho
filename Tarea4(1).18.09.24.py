# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:34:52 2024

@author: Karla Edith
"""


"""
Tarea 1: Resolver la versión no lineal (x³) del oscilador armónico, usando su propio código.

Formalismo
En este trabajo se estudia el oscilador armónico no lineal con una fuerza restauradora de tipo cúbico. La ecuación diferencial que modela este sistema es:
d**2/dt**2=-k*(x**3)
Esta ecuación puede reescribirse como un sistema de ecuaciones de primer orden introduciendo v=dx/dt, quedando dv/dt=-k*(x**3
Este sistema describe el movimiento de una partícula sometida a una fuerza no lineal, cuya constante k afecta la rigidez del sistema. Se investigan diferentes valores de 
k para observar su influencia en el comportamiento dinámico del sistema.

Algoritmos
Se utilizó el método de Euler explícito para integrar el sistema de ecuaciones diferenciales. El método de Euler aproxima la solución en cada paso temporal utilizando las derivadas locales del sistema. Dado un estado (i,vi) en el tiempo ti.
Se implementó una función que toma como entrada el método numérico y los parámetros del sistema, y genera las trayectorias para la posición y velocidad a lo largo del tiempo. Se evalúan tres intervalos para k: [0, 1], [1, 25] y [30, 200], con tres diferentes tamaños de partición y número de segmentos para cada intervalo de 
k.
"""
#Código
import numpy as np
import matplotlib.pyplot as plt

# Definir la ecuación diferencial del oscilador no lineal
def nonlinear_harmonic_eq(x, t, k):
    return x[1], -k * x[0]**3

# Implementación del método de Euler para dos variables (posición y velocidad)
def euler_2var(x, func, t, k, dt):
    y = func(x, t, k)
    return x[0] + dt * y[0], x[1] + dt * y[1]

# Función para realizar las simulaciones y gráficar los resultados
def calc_plot2var(method, equation, k_values, dt, n_steps):
    for k in k_values:
        t = np.arange(0, n_steps * dt, dt)
        x = np.zeros((n_steps, 2))  # Inicializar arreglo de posición y velocidad
        x[0][0] = 2.0  # Posición inicial
        x[0][1] = 0.0  # Velocidad inicial

        # Resolver la ecuación diferencial usando Euler
        for i in range(n_steps - 1):
            x[i + 1] = method(x[i], nonlinear_harmonic_eq, t[i], k, dt)

        # Graficar posición y velocidad
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(t, x[:, 0], 'r', label="$x(t)$ (Posición)")
        ax1.plot(t, x[:, 1], 'b', label="$v(t)$ (Velocidad)")
        ax1.set_xlabel("Tiempo ($t$)")
        ax1.set_title(f"Oscilador no lineal con $k = {k}$")
        ax1.legend(loc="upper right")

        # Graficar la trayectoria en el espacio fase
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(x[:, 0], x[:, 1], '#ff8800')
        ax2.set_xlabel("Posición ($x(t)$)")
        ax2.set_ylabel("Velocidad ($v(t)$)")
        ax2.set_title("Espacio fase")

        plt.tight_layout()
        plt.show()

# Parámetros de simulación
dt_values = [0.01, 0.001, 0.0001]  # Diferentes tamaños de paso
n_steps_values = [5000, 10000, 20000]  # Diferentes números de pasos

# Definir los valores de k para explorar
k_values_1 = np.linspace(0.01, 1, 10)  # Entre 0 y 1
k_values_2 = np.linspace(1, 25, 10)    # Entre 1 y 25
k_values_3 = np.linspace(30, 200, 10)  # Entre 30 y 200

# Ejecutar simulaciones para los tres conjuntos de valores de k
for dt, n_steps in zip(dt_values, n_steps_values):
    print(f"Simulando para dt = {dt}, n_steps = {n_steps}")
    calc_plot2var(euler_2var, nonlinear_harmonic_eq, k_values_1, dt, n_steps)
    calc_plot2var(euler_2var, nonlinear_harmonic_eq, k_values_2, dt, n_steps)
    calc_plot2var(euler_2var, nonlinear_harmonic_eq, k_values_3, dt, n_steps)
    
"""
1.-Trayectorias para k∈[0,1]:Para valores pequeños de k, el sistema presenta oscilaciones armónicas regulares con pequeñas desviaciones de la linealidad.
2.-Trayectorias para k∈[1,25]:A medida que k aumenta, las oscilaciones se vuelven más no lineales y se observan trayectorias más complejas en el espacio fase.
3.-Trayectorias para k∈[30,200]:Para valores grandes de k, el sistema exhibe oscilaciones rápidas y amplitudes menores debido a la fuerza restauradora más intensa.

Análisis crítico
Este experimento numérico permitió observar cómo la constante k modifica el comportamiento dinámico del oscilador armónico no lineal. Los valores pequeños de k producen oscilaciones cercanas al comportamiento armónico, mientras que para valores más grandes, las oscilaciones se vuelven más no lineales y rápidas.
Se observó que el método de Euler explícito, aunque simple, es suficiente para capturar el comportamiento cualitativo del sistema. Sin embargo, para valores muy grandes de k, el método podría requerir pasos temporales más pequeños para garantizar la estabilidad y precisión de la solución. Una posible mejora sería la implementación de métodos numéricos más robustos, como el método de Runge-Kutta de cuarto orden, que ofrecería mayor precisión sin la necesidad de disminuir el tamaño del paso temporal.
"""

R"esultados
Se obtuvieron gráficos que muestran la evolución temporal de la posición x(t) y la velocidad v(t) del sistema para diferentes valores de k.
Trayectorias para k∈[0,1]:Para valores pequeños de 

""