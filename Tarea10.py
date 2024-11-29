#Ejemplo 6.1.1
from random import random
from math import sqrt

# Dado que el radio es 1, no necesitamos incluirlo explícitamente en los cálculos.
N = 1000000  # Número de puntos aleatorios en el cubo unitario
count = 0  # Número de puntos dentro de la esfera

# Generamos puntos aleatorios y contamos los que caen dentro de la esfera
for _ in range(N):
    point = (random(), random(), random())
    if point[0]**2 + point[1]**2 + point[2]**2 < 1:
        count += 1

# Calculamos la proporción de puntos dentro de la esfera
Answer = float(count) / float(N)

# Multiplicamos por 4 para obtener el volumen total de la semiesfera
Answer *= 4

# Calculamos la incertidumbre estadística
uncertainty = 4 * sqrt(N) / float(N)

# Mostramos el resultado
print(f"La volumen de una semiesfera de radio 1 es {Answer:.4f} ± {uncertainty:.4f}.")

#Problema 6.0
from random import uniform
from math import sin, sqrt, pi

# Número de puntos aleatorios
N = 1000000

# Suma acumulada de los valores de sin(x)
sum_sin = 0

# Generar puntos aleatorios y evaluar sin(x)
for _ in range(N):
    x = uniform(0, pi)
    sum_sin += sin(x)

# Estimar la integral
integral = (pi / N) * sum_sin

# Calcular la incertidumbre
uncertainty = pi / sqrt(N)

# Imprimir resultados
print(f"Resultado de la integral estimada: {integral:.6f} ± {uncertainty:.6f}")
print("Resultado conocido: 2.000000")

#Problema 6.2
from random import uniform
from math import pi

# Número de puntos aleatorios
N = 1000000

# Inicializar contador de puntos dentro de la 4-esfera
count_inside = 0

# Generar puntos aleatorios en el hipercubo y verificar si están dentro de la 4-esfera
for _ in range(N):
    x = uniform(-1, 1)
    y = uniform(-1, 1)
    z = uniform(-1, 1)
    w = uniform(-1, 1)
    if x**2 + y**2 + z**2 + w**2 <= 1:
        count_inside += 1

# Calcular volumen de la 4-esfera
volume_4sphere = (count_inside / N) * 16

# Calcular valor de alpha
alpha = volume_4sphere / pi**4

# Imprimir resultados
print(f"Volumen estimado de la 4-esfera: {volume_4sphere:.6f}")
print(f"Valor estimado de alpha: {alpha:.6f}")



