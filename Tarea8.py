#Tarea 8:A. Calcular la probabilidad de que al tomar un conjunto de 3 alumnos de un grupo de 40, los 3 sean mujeres, si sólo hay 9 mujeres en el grupo.

from math import comb

# Datos
total_alumnos = 40
total_mujeres = 9
alumnos_seleccionados = 3

# Cálculo de combinaciones
comb_mujeres = comb(total_mujeres, alumnos_seleccionados)
comb_total = comb(total_alumnos, alumnos_seleccionados)

# Cálculo de la probabilidad
probabilidad = comb_mujeres / comb_total
print("Probabilidad de seleccionar 3 mujeres:", probabilidad)

#b)B. Considere una clase de 100 alumnos. Asigne, de manera aleatoria, sus calificaciones entre 5 y 10. Suponga que su resultado sea típico. a. calcule el promedio y la varianza del experimento; b. repita el ejercicio 10 veces (10 cursos)
import numpy as np

# Inicialización de variables
num_estudiantes = 100
num_cursos = 10
promedios = []
varianzas = []

# Generación de calificaciones aleatorias y cálculo de estadísticas
for _ in range(num_cursos):
    calificaciones = np.random.uniform(5, 10, num_estudiantes)
    promedio = np.mean(calificaciones)
    varianza = np.var(calificaciones)
    promedios.append(promedio)
    varianzas.append(varianza)

print("Promedios de cada curso:", promedios)
print("Varianzas de cada curso:", varianzas)

#c) C. De A2, calcular promedio y desviación estándar
# Cálculo del promedio y desviación estándar de los promedios de los 10 cursos
promedio_general = np.mean(promedios)
desviacion_estandar = np.std(promedios)

print("Promedio de los promedios de los 10 cursos:", promedio_general)
print("Desviación estándar de los promedios de los 10 cursos:", desviacion_estandar)


