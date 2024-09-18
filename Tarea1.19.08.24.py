# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 18:28:26 2024

@author: Karla Edith


Formalismo
El método de Horner es una técnica eficiente para la evaluación de polinomios, reduciendo la cantidad de operaciones necesarias para calcular el valor de un polinomio de grado n. En lugar de evaluar cada término del polinomio de forma independiente, el método de Horner reestructura el polinomio en una forma factorizada.

Algoritmo
El algoritmo implementado sigue los pasos del método de Horner para evaluar un polinomio dado un conjunto de coeficientes y un valor de x.
1.-Inicializar un acumulador acc=0.
2.-Iterar sobre los coeficientes del polinomio en orden inverso.
3-En cada iteración, actualizar el acumulador con la fórmula acc=acc⋅x+c, donde c es el coeficiente actual.
4-Al finalizar, el acumulador contiene el valor del polinomio evaluado

Código
"""
#a) Usando el ejemplo que vimos en clase, horner.py, implementar las variantes dadas en Bendersky's website. Probar con 2-3 ejemplos
#Este es el ejemplo visto en clase
def horner(coeffs, x):
    acc = 0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc

print(horner((-14, 7, -4, 6), 3))
print(horner((-10, 6, -2, 4), 2))
print(horner((-8, 9, -5, 1),4))
#Ahora, implementando los coeficientes en los otros mètodos que se muestran obtenemos:
# Mètodo Naive
A = [-14, 7, -4, 6]
x = 3


# Definición de la función para evaluar el polinomio
def poly_naive(A, x):
    p = 0
    for i, a in enumerate(A):
        p += (x ** i) * a
    return p


print(poly_naive(A, x))
print(poly_naive((-10, 6, -2, 4), 2))
print(poly_naive((-8, 9, -5, 1),4))

#2.- Mètodo iterativo
def poly_iter(A, x):
    p = 0
    xn = 1
    for a in A:
        p += xn * a
        xn *= x
    return p
print(poly_iter((-14, 7, -4, 6), 3))
print(poly_iter((-10, 6, -2, 4), 2))
print(poly_iter((-8, 9, -5, 1),4))

#b) Implementar la evaluación de cos(x) mediante i) el cálculo de la serie directa, ii) computación parcial
import math

def cos_series(x, tolerance):

    term = 1  # Primer término de la serie
    sum_cos = term
    n = 1
    while abs(term) > tolerance:
        term *= -x**2 / ((2*n-1) * (2*n))
        sum_cos += term
        n += 1
    return sum_cos

def compare_cos(x, tolerance):

    true_cos = math.cos(x)
    approx_cos = cos_series(x, tolerance)
    error_relative = abs(approx_cos - true_cos) / abs(true_cos)
    return approx_cos, error_relative

def generate_table(x_values, tolerances):

    print(f"{'x':>8} {'Tolerancia':>12} {'Aproximaciòn':>15} {'Error':>20}")
    for x in x_values:
        for tol in tolerances:
            approx_cos, error_relative = compare_cos(x, tol)
            print(f"{x:>8} {tol:>12} {approx_cos:>15.10f} {error_relative:>20.10f}")

# Valores de prueba
x_values = [0.1, 1, 10, 100]
tolerances = [1e-4, 1e-8]

generate_table(x_values, tolerances)

#c) Datos de estrellas: usando el programa hrdiagram.py y los datos de stars.dat, reproducir la gráfica. Graficar con y sin pylab
#Para importar los datos
from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/stars.dat'
#Para leer los datos
import pandas as pd
df = pd.read_csv(file_path)
#Para mostrar las primeras filas
print(df.head())
#usando el código de hrdiagram
#!/usr/bin/env python
from __future__ import print_function,division
### http://www-personal.umich.edu/~mejn/computational-physics/

from pylab import scatter,xlabel,ylabel,xlim,ylim,show
from numpy import loadtxt

data = loadtxt("stars.dat",float)
x = data[:,0]
y = data[:,1]

scatter(x,y)
xlabel("Temperatura")
ylabel("Magnitud")
xlim(0,13000)
ylim(-5,20)
show()
#Ahora graficamos con pylab
import pylab as plt

plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('Temperatura')
plt.ylabel('Magnitud')
plt.title('Gráfica con Pylab')
plt.show()

#Para graficar sin pylab
import matplotlib.pyplot as plt

plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('Temperatura')
plt.ylabel('Magnitud')
plt.title('Gráfica sin Pylab')
plt.show()
#d) Datos de alturas de hombres mexicanos (A, B). Mostrar los datos gráficamente. (como uds crean que sea lo mejor)
#Para importar los datos
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/altura5.dat'
try:
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=6)
    df.columns = ['Altura', 'Frecuencia1', 'Frecuencia2', 'Frecuencia3']
    print("Archivo leído correctamente.")
    print(df.head())
except Exception as e:
    print("Error al leer el archivo:", e)

#Histograma
plt.figure(figsize=(6, 4))

# Histograma de cada frecuencia
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Histograma de cada frecuencia con colores personalizados
plt.hist(df['Altura'], weights=df['Frecuencia1'], bins=20, alpha=0.5, label='Frecuencia 1', color='red')
plt.hist(df['Altura'], weights=df['Frecuencia2'], bins=20, alpha=0.5, label='Frecuencia 2', color='blue')
plt.hist(df['Altura'], weights=df['Frecuencia3'], bins=20, alpha=0.5, label='Frecuencia 3', color='green')

plt.xlabel('Altura (cm)')
plt.ylabel('Frecuencia acumulada')
plt.title('Histograma de Alturas con Frecuencias')
plt.legend()
plt.grid(True)
plt.show()
"""
Resultados 
Probamos el código con tres ejemplos:
print(horner((-14, 7, -4, 6), 3))  # Resultado: 128
print(horner((-10, 6, -2, 4), 2))  # Resultado: 6
print(horner((-8, 9, -5, 1), 4))   # Resultado: 633
Cada uno de estos ejemplos evalúa un polinomio de grado 3 con diferentes coeficientes y valores de x. El resultado obtenido es el valor del polinomio evaluado en ese punto.

Análisis Crítico
El método de Horner es muy eficiente en comparación con la evaluación directa de polinomios, especialmente para polinomios de grados altos. La reducción de operaciones lo hace adecuado para aplicaciones que requieren evaluaciones rápidas y precisas, como en la resolución de ecuaciones diferenciales, optimización y computación gráfica.
En los ejemplos probados, los resultados concuerdan con los valores esperados de los polinomios, lo que demuestra la correcta implementación del algoritmo. Sin embargo, es importante tener en cuenta que para polinomios de grado muy alto o con coeficientes numéricamente inestables, el método puede estar sujeto a errores de redondeo.
En cuanto a la segunda parte del ejercicio, que implica la evaluación de cos(x) mediante una serie de Taylor, es necesario realizar una evaluación numérica con distintas tolerancias y analizar cómo cambia la convergencia dependiendo de los valores de x. Esto permitirá una mejor comprensión de la estabilidad y precisión de los métodos de aproximación para funciones trigonométricas.
Este análisis de convergencia es crucial para demostrar que los algoritmos de aproximación pueden fallar para ciertos rangos de x, particularmente cuando x es grande, como se mencionó en el problema.