# -*- coding: utf-8 -*-
"""
Created on Tue Sep 5 18:28:22 2024

@author: Karla Edith
Formalismo
La integración numérica es una técnica utilizada para calcular aproximaciones al valor de una integral definida cuando no es posible obtener una solución exacta analíticamente. Tres métodos comunes para la integración numérica son:
Regla del Trapecio: Aproxima el área bajo la curva de la función utilizando trapezoides. 
Regla de Simpson: Utiliza polinomios de segundo grado para aproximar la función.
Cuadratura de Gauss-Legendre: Utiliza puntos y pesos específicos para evaluar la integral, proporcionando una alta precisión con menos puntos. 
Algoritmo
El código implementa los tres métodos de integración:

#Regla del Trapecio:
"""
def trapecio_compuesto(f, a, b, n):
    h = (b - a) / n
    suma = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        suma += f(a + i * h)
    return h * suma
#Regla de Simpson:

def simpson_compuesto(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    suma = f(a) + f(b)
    for i in range(1, n, 2):
        suma += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        suma += 2 * f(a + i * h)
    return h * suma / 3
#Cuadratura de Gauss-Legendre:

def gauss_legendre(f, a, b, n):
    [x, w] = roots_legendre(n)
    x_mapped = 0.5 * (x * (b - a) + (b + a))
    w_mapped = 0.5 * (b - a) * w
    return sum(w_mapped * f(x_mapped))
#Código
import numpy as np
from scipy.special import roots_legendre
from scipy.integrate import quad

#Función a integrar
def f(x):
    return (np.exp(x) * np.sin(x)) / (1 + x**2)

valor_exacto, _ = quad(f, 0, 3)

# Regla del Trapecio
def trapecio_compuesto(f, a, b, n):
    h = (b - a) / n
    suma = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        suma += f(a + i * h)
    return h * suma

# Regla de Simpson
def simpson_compuesto(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    suma = f(a) + f(b)
    for i in range(1, n, 2):
        suma += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        suma += 2 * f(a + i * h)
    return h * suma / 3

# Cuadratura de Gauss-Legendre
def gauss_legendre(f, a, b, n):
    [x, w] = roots_legendre(n)
    x_mapped = 0.5 * (x * (b - a) + (b + a))
    w_mapped = 0.5 * (b - a) * w
    return sum(w_mapped * f(x_mapped))

# Cálculo de las integrales numéricas y errores
a, b = 0, 3
n_vals = [6, 15, 20]

for n in n_vals:
    resultado_trapecio = trapecio_compuesto(f, a, b, n)
    resultado_simpson = simpson_compuesto(f, a, b, n)
    resultado_gauss = gauss_legendre(f, a, b, n)

    error_trapecio = abs(valor_exacto - resultado_trapecio)
    error_simpson = abs(valor_exacto - resultado_simpson)
    error_gauss = abs(valor_exacto - resultado_gauss)

    print(f"\nPara n = {n}:")
    print(f"Regla del Trapecio: {resultado_trapecio:.8f}, Error: {error_trapecio:.8e}")
    print(f"Regla de Simpson: {resultado_simpson:.8f}, Error: {error_simpson:.8e}")
    print(f"Gauss-Legendre: {resultado_gauss:.8f}, Error: {error_gauss:.8e}")
#Repetir el problema 2. pero considerando los límites en (-2, 0)
import numpy as np
from scipy.special import roots_legendre
from scipy.integrate import quad

# Función a integrar
def f(x):
    return (np.exp(x) * np.sin(x)) / (1 + x**2)

#Límites de integración
valor_exacto, _ = quad(f, -2, 0)

# Regla del Trapecio
def trapecio_compuesto(f, a, b, n):
    h = (b - a) / n
    suma = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        suma += f(a + i * h)
    return h * suma

# Regla de Simpson
def simpson_compuesto(f, a, b, n):
    if n % 2 == 1:  # Simpson requiere un número par de subintervalos
        n += 1
    h = (b - a) / n
    suma = f(a) + f(b)
    for i in range(1, n, 2):
        suma += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        suma += 2 * f(a + i * h)
    return h * suma / 3

# Cuadratura de Gauss-Legendre
def gauss_legendre(f, a, b, n):
    [x, w] = roots_legendre(n)
    x_mapped = 0.5 * (x * (b - a) + (b + a))
    w_mapped = 0.5 * (b - a) * w
    return sum(w_mapped * f(x_mapped))

# Cálculo de las integrales numéricas y errores con los nuevos límites
a, b = -2, 0
n_vals = [6, 15, 20]

for n in n_vals:
    resultado_trapecio = trapecio_compuesto(f, a, b, n)
    resultado_simpson = simpson_compuesto(f, a, b, n)
    resultado_gauss = gauss_legendre(f, a, b, n)

    error_trapecio = abs(valor_exacto - resultado_trapecio)
    error_simpson = abs(valor_exacto - resultado_simpson)
    error_gauss = abs(valor_exacto - resultado_gauss)

    print(f"\nPara n = {n}:")
    print(f"Regla del Trapecio: {resultado_trapecio:.8f}, Error: {error_trapecio:.8e}")
    print(f"Regla de Simpson: {resultado_simpson:.8f}, Error: {error_simpson:.8e}")
    print(f"Gauss-Legendre: {resultado_gauss:.8f}, Error: {error_gauss:.8e}")
#4. Integrar numéricamente f(x) = e^x/x, g(x) = (1-e^x)/x en el intervalo (0,5).
#Elijan el método y número de particiones.
import numpy as np
from scipy.integrate import quad

# Funciones a integrar
def f(x):
    return np.exp(x) / x

def g(x):
    return (1 - np.exp(x)) / x

#
def f_approximated(x):
    return np.exp(x) / x if x > 1e-10 else 1.0
def g_approximated(x):
    return (1 - np.exp(x)) / x if x > 1e-10 else -1.0

# Regla de Simpson
def simpson_compuesto(f, a, b, n):
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    suma = f(a) + f(b)
    for i in range(1, n, 2):
        suma += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        suma += 2 * f(a + i * h)
    return h * suma / 3

# Cálculo de las integrales numéricas
a, b = 0, 5
n = 100

resultado_f = simpson_compuesto(f_approximated, a, b, n)
resultado_g = simpson_compuesto(g_approximated, a, b, n)

# Comparación con el valor exacto usando scipy
valor_exacto_f, _ = quad(f_approximated, a, b)
valor_exacto_g, _ = quad(g_approximated, a, b)

error_f = abs(valor_exacto_f - resultado_f)
error_g = abs(valor_exacto_g - resultado_g)

print(f"Integración de f(x) = e^x/x en [0, 5]:")
print(f"Resultado Simpson: {resultado_f:.8f}, Error: {error_f:.8e}")

print(f"Integración de g(x) = (1 - e^x)/x en [0, 5]:")
print(f"Resultado Simpson: {resultado_g:.8f}, Error: {error_g:.8e}")


"""
Análisis Crítico
Regla del Trapecio: A medida que aumentamos el número de subintervalos, la precisión mejora. Sin embargo, el método puede ser menos eficiente comparado con Simpson y Gauss-Legendre para el mismo número de particiones.
Regla de Simpson: Generalmente ofrece una mayor precisión con menos particiones en comparación con la regla del trapecio, dado que usa aproximaciones cuadráticas en lugar de lineales.
Cuadratura de Gauss-Legendre: Proporciona la mayor precisión con el menor número de puntos, pero la complejidad aumenta con el número de puntos. Es especialmente útil para integrales en intervalos finitos y funciones que tienen características complejas.
