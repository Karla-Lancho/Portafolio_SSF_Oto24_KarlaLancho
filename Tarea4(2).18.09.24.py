# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:09:08 2024

@author: Karla Edith
"""

"""
Tarea 2: Usar el programa para tomar diferentes valores de k y explorar cómo afecta el comportamiento del sistema.
Formalismo
En esta tarea, se estudia el comportamiento del oscilador armónico no lineal con una fuerza restauradora de tipo cúbico descrita por la ecuación:
d**2x/dt**2=-k*(x**3)
Esta ecuación describe un sistema donde la rigidez del oscilador es no lineal, lo que introduce complejidad en su dinámica. La constante k juega un rol clave al determinar la fuerza restauradora.
El objetivo de esta tarea es explorar cómo varía el comportamiento del oscilador para diferentes valores de
Algoritmos
El programa utiliza el método de Euler explícito para resolver el sistema de ecuaciones diferenciales de primer orden derivado de la ecuación original. Se descompone la ecuación de segundo orden en dos ecuaciones de primer orden introduciendo la variable auxiliar v=dx/dt, obteniendo el sistema:
El programa simula el comportamiento del sistema para diferentes valores de k, ajustando tanto el tamaño del paso temporal como el número de segmentos de la simulación para explorar cómo estos parámetros afectan la precisión de la solución.
Código
"""
import numpy as np
import matplotlib.pyplot as plt

# Función que describe el sistema de ecuaciones diferenciales
def oscilador_no_lineal(t, x, v, k):
    dxdt = v
    dvdt = -k * x**3
    return dxdt, dvdt

# Función que resuelve el sistema usando el método de Euler explícito
def resolver_oscilador(k, x0=1, v0=0, t_max=100, dt=0.01):
    # Inicialización
    n_steps = int(t_max / dt)
    t_values = np.linspace(0, t_max, n_steps)
    x_values = np.zeros(n_steps)
    v_values = np.zeros(n_steps)

    # Condiciones iniciales
    x_values[0] = x0
    v_values[0] = v0

    # Integración usando Euler
    for i in range(1, n_steps):
        dxdt, dvdt = oscilador_no_lineal(t_values[i-1], x_values[i-1], v_values[i-1], k)
        x_values[i] = x_values[i-1] + dxdt * dt
        v_values[i] = v_values[i-1] + dvdt * dt

    return t_values, x_values, v_values

# Función para ejecutar las simulaciones con diferentes valores de k y dt
def simulaciones(k_values, dt_values, t_max=100):
    for k in k_values:
        for dt in dt_values:
            t, x, v = resolver_oscilador(k=k, t_max=t_max, dt=dt)
            plt.plot(t, x, label=f'k={k}, dt={dt}')
    
    plt.title('Posición x(t) para diferentes valores de k y dt')
    plt.xlabel('Tiempo t')
    plt.ylabel('Posición x')
    plt.legend()
    plt.show()

# Valores de k en los diferentes rangos especificados
k_values_1 = np.linspace(0.01, 1, 10)  # Rango de k entre 0 y 1
k_values_2 = np.linspace(1, 25, 10)    # Rango de k entre 1 y 25
k_values_3 = np.linspace(30, 200, 10)  # Rango de k entre 30 y 200

# Valores de dt y número de segmentos (adaptado como número de pasos temporales)
dt_values = [0.01, 0.05, 0.1]  # Diferentes tamaños de partición

# Ejecutar las simulaciones para cada rango de k
print("Simulaciones para k entre 0 y 1:")
simulaciones(k_values_1, dt_values)

print("Simulaciones para k entre 1 y 25:")
simulaciones(k_values_2, dt_values)

print("Simulaciones para k entre 30 y 200:")
simulaciones(k_values_3, dt_values)

"""
Resultados 
Se realizaron simulaciones para los tres rangos de valores de k solicitados, con tres configuraciones diferentes para el tamaño del paso temporal y el número de pasos.
1.-Valores de k∈[0,1]:En este intervalo, el sistema presenta oscilaciones armónicas relativamente simples con pequeñas desviaciones no lineales. El comportamiento es cercano al de un oscilador lineal.
2.-Valores de k∈[1,25]:A medida que k aumenta en este intervalo, las oscilaciones se vuelven notablemente no lineales, y se observan trayectorias más complejas en el espacio fase. Las soluciones son más sensibles al valor de k y las variaciones en el paso temporal.
3.-Valores de k∈[30,200]:En este rango, el oscilador experimenta oscilaciones rápidas y con amplitudes menores debido a la mayor fuerza restauradora. La dinámica se vuelve aún más no lineal y la resolución numérica requiere pasos temporales más pequeños para mantener la precisión.
Análisis crítico
Al usar el programa para explorar diferentes valores de k, se pudo comprobar que a medida que k aumenta, el sistema exhibe un comportamiento cada vez más no lineal y sensible a las variaciones en el paso temporal. Esto confirma que, para valores grandes de k, el sistema se vuelve más rígido, lo que requiere métodos numéricos más avanzados (o pasos temporales más pequeños) para mantener la estabilidad y la precisión en la simulación.
El uso del método de Euler funciona bien para valores pequeños de k, pero para k grandes, el método presenta limitaciones en cuanto a precisión. Una mejora posible sería la implementación de un método numérico de mayor orden, como Runge-Kutta, o la incorporación de un esquema adaptativo de paso temporal para gestionar mejor la dinámica rápida en sistemas con k elevados.
"""