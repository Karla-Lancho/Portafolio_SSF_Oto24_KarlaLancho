# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:51:05 2024

@author: Karla Edith
"""
#Ejercicico A.2
"""
Formalismo
La función a graficar es 𝑓(𝑥)=sin(1/𝑥+𝜀)
Usamos n+1 puntos en la gráfica para definir el número de nodos.
Variamos 𝑛 y ε para observar cómo cambia la gráfica de la función.
Algoritmos
Utilizaremos un bucle while para calcular el valor mínimo de 𝑛 tal que la diferencia entre dos gráficas sea menor que 0.1.
La función max nos ayudará a obtener la diferencia máxima entre las funciones evaluadas para diferentes valores de n.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_function(n, eps):
    # Definir los puntos de x
    x = np.linspace(0, 1, n+1)
    # Calcular la función f(x)
    f_x = np.sin(1 / (x + eps))
    return x, f_x

def find_min_n(eps, tol=0.1):
    n = 10
    while True:
        # Obtener las funciones con n y n + 10
        x_n, f_x_n = plot_function(n, eps)
        x_n10, f_x_n10 = plot_function(n + 10, eps)
        
        # Calcular la diferencia máxima entre las dos funciones
        diff = np.max(np.abs(f_x_n - f_x_n10))
        
        if diff < tol:
            break
        n += 1
    
    return n

# (a) Probar con n = 10 y ε = 1/5
n = 10
epsilon = 1/5
x, f_x = plot_function(n, epsilon)

plt.figure()
plt.plot(x, f_x, label=f'n={n}, ε={epsilon}')
plt.title('Plot of sin(1/(x + ε))')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# (b) Refinar el programa para dos valores de n
n1 = 10
n2 = n1 + 10
x1, f_x1 = plot_function(n1, epsilon)
x2, f_x2 = plot_function(n2, epsilon)

plt.figure()
plt.plot(x1, f_x1, label=f'n={n1}, ε={epsilon}')
plt.plot(x2, f_x2, label=f'n={n2}, ε={epsilon}')
plt.title('Comparison for Different n Values')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# (c) Encontrar el valor mínimo de n para que la diferencia sea menor a 0.1
min_n = find_min_n(epsilon)
print(f"Minimum n for ε={epsilon} such that the difference is less than 0.1: {min_n}")

# (d) y (e) Repetir para diferentes valores de ε
epsilons = [1/10, 1/20]
for eps in epsilons:
    min_n = find_min_n(eps)
    print(f"Minimum n for ε={eps} such that the difference is less than 0.1: {min_n}")
"""
Resultados
El código genera gráficos de la función 
𝑓(𝑥) para diferentes valores de n y ε.
Además, encuentra el valor mínimo de 𝑛 para que la diferencia entre las dos funciones sea menor que 0.1.
Análisis Crítico
Aprendizaje: Este ejercicio ayuda a entender cómo los parámetros afectan la convergencia de una función y la importancia de ajustar n para obtener una precisión deseada.
Mejoras posibles: Se podrían explorar diferentes métodos de interpolación para mejorar la precisión. También se podría usar técnicas de paralelización para mejorar la eficiencia del código cuando se trabaja con valores grandes de n.
"""
#EJERCICIO A.3
"""
Formalismo 
Función f(x)= sen (1/x+E)
Derivada exacta = f´(x)=-(cos (1/x+E))/(x+E)**2
Utilizamos una aproximacion de diferencias finitas para calcular la derivada de f(x) con n nodos computacionales en el intervalo [0,1].
Algoritmos
Aproximación de diferencias finitas: La derivada de f(x) se puede aproximar numéricamente usando diferencias finitas de la siguiente forma:
f´(x)=f(x+h)-f(x)/h donde h es un pequeño incremento.
Comparación con la derivada exacta: Compararemos la derivada aproximada con la derivada exacta para encontrar el valor mínimo de 
𝑛
n tal que la diferencia máxima sea menor que 0.1.
"""
import numpy as np
import matplotlib.pyplot as plt

def function_f(x, eps):
    return np.sin(1 / (x + eps))

def exact_derivative_f(x, eps):
    return -np.cos(1 / (x + eps)) / (x + eps)**2

def finite_difference_derivative(x, eps, h=1e-5):
    # Aproximación de diferencias finitas
    return (function_f(x + h, eps) - function_f(x, eps)) / h

def plot_derivatives(n, eps):
    # Definir los puntos de x
    x = np.linspace(0, 1, n+1)
    
    # Calcular la derivada exacta y la aproximación de diferencias finitas
    exact_derivative = exact_derivative_f(x, eps)
    approx_derivative = finite_difference_derivative(x, eps)

    # Graficar las derivadas
    plt.figure()
    plt.plot(x, exact_derivative, label='Exact derivative', color='r')
    plt.plot(x, approx_derivative, label='Finite difference approximation', linestyle='--', color='b')
    plt.title(f'Derivatives of f(x) for n={n}, ε={eps}')
    plt.xlabel('x')
    plt.ylabel('f\'(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def find_min_n(eps, tol=0.1):
    n = 10
    while True:
        # Calcular derivadas para n y n+10
        x_n = np.linspace(0, 1, n+1)
        exact_derivative_n = exact_derivative_f(x_n, eps)
        approx_derivative_n = finite_difference_derivative(x_n, eps)
        
        x_n10 = np.linspace(0, 1, n+11)
        exact_derivative_n10 = exact_derivative_f(x_n10, eps)
        approx_derivative_n10 = finite_difference_derivative(x_n10, eps)
        
        # Comparar la diferencia máxima
        max_diff = np.max(np.abs(exact_derivative_n - approx_derivative_n))
        
        if max_diff < tol:
            break
        n += 1
    
    return n

# (a) Probar con n = 10 y ε = 1/5
n = 10
epsilon = 1/5
plot_derivatives(n, epsilon)

# (b) Refinar para dos valores de n
n1 = 10
n2 = n1 + 10
plot_derivatives(n1, epsilon)
plot_derivatives(n2, epsilon)

# (c) Encontrar el valor mínimo de n para que la diferencia sea menor a 0.1
min_n = find_min_n(epsilon)
print(f"Minimum n for ε={epsilon} such that the difference is less than 0.1: {min_n}")

# (d) y (e) Repetir para diferentes valores de ε
epsilons = [1/10, 1/20]
for eps in epsilons:
    min_n = find_min_n(eps)
    print(f"Minimum n for ε={eps} such that the difference is less than 0.1: {min_n}")

"""
Resultados
Visualización: Se generan gráficos que comparan la derivada exacta y la derivada aproximada usando diferencias finitas.
Resultados de Convergencia: Se determina el valor mínimo de n para cada ε de forma que la diferencia sea menor a 0.1.
Análisis Crítico
Aprendizaje: Este ejercicio muestra cómo implementar y utilizar métodos de diferencias finitas para aproximar derivadas y la importancia de ajustar el número de nodos computacionales para obtener resultados precisos.
Mejoras posibles: Se podrían probar diferentes métodos de diferencias finitas (como diferencias centradas) o incrementar la precisión ajustando adaptativamente el valor de h.
"""
#Ejercicio A.4
"""
Formalismo
Integral a calcular a=de 0 a 1, e**4x dx = 1/4 e**4 -1/4
Queremos aproximar la integral usando el método del trapecio y comparar el resultado con el valor exacto a.

El objetivo es determinar el valor mínimo de n (número de subdivisiones).
Algoritmo
Método del trapecio: La fórmula para aproximar una integral usando el método del trapecio con n subdivisiones.
Comparar el valor exacto con el aproximado: Encontrar el mínimo valor de n para el cual la diferencia sea menor o igual a ε.
"""
import numpy as np

def exact_integral():
    # Cálculo exacto de la integral
    return (1/4) * np.exp(4) - (1/4)

def trapezoidal_method(n):
    # Definición de la función f(x) = e^(4x)
    def f(x):
        return np.exp(4 * x)
    
    # Limites de la integral
    a, b = 0, 1
    h = (b - a) / n  # tamaño del paso

    # Aplicación del método del trapecio
    integral_approx = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        integral_approx += f(a + i * h)
    
    integral_approx *= h
    return integral_approx

def find_min_n(epsilon):
    exact_value = exact_integral()
    n = 1
    while True:
        approx_value = trapezoidal_method(n)
        error = abs(exact_value - approx_value)
        if error <= epsilon:
            break
        n += 1
    return n

# (a) Calcular para ε = 1/100
epsilon1 = 1/100
n1 = find_min_n(epsilon1)
print(f"Minimum n for ε={epsilon1}: {n1}")

# (b) Repetir para ε = 1/1000
epsilon2 = 1/1000
n2 = find_min_n(epsilon2)
print(f"Minimum n for ε={epsilon2}: {n2}")

# (c) Repetir para ε = 1/10000
epsilon3 = 1/10000
n3 = find_min_n(epsilon3)
print(f"Minimum n for ε={epsilon3}: {n3}")

# (d) Determinar experimentalmente cuán grande debe ser n
# para un valor general de ε
epsilons = [epsilon1, epsilon2, epsilon3]
for eps in epsilons:
    min_n = find_min_n(eps)
    print(f"Minimum n for ε={eps}: {min_n}")
"""
Resultados
Visualización: El código no incluye visualización, pero podemos imprimir los valores de n necesarios para cada valor de ε.
Resultados de Convergencia: Determinamos el valor mínimo de n para cada ε de forma que la diferencia sea menor o igual al umbral dado.
Análisis Crítico
Aprendizaje: Este ejercicio demuestra cómo usar el método del trapecio para calcular integrales numéricas y encontrar el valor óptimo de subdivisiones para alcanzar una precisión deseada.
Mejoras posibles: Se podría comparar este método con otras técnicas de integración numérica (como la regla de Simpson) para ver cuál es más eficiente en términos de convergencia.
"""