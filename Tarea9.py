import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import seaborn as sns
print(sns.__version__)


# Carga el archivo Excel. Cambia la ruta según corresponda.
ruta = "C:\\Users\\Karla\\Downloads\\S1_Dataset.xlsx"
datos = pd.read_excel(ruta)

# Visualización de las primeras filas
print(datos.head())

# Verifica los nombres de las columnas
print("Nombres de columnas:", datos.columns)

# Selección de variables (reemplaza 'Variable1' y 'Variable2' con nombres reales de columnas)
# Asegúrate de que los nombres de las columnas sean correctos
variable1 = datos['Variable1'].dropna()  # Reemplazar con nombres de columnas reales
variable2 = datos['Variable2'].dropna()

# Asegúrate de que ambas variables tengan la misma longitud después de eliminar NaN
min_length = min(len(variable1), len(variable2))
variable1 = variable1[:min_length]
variable2 = variable2[:min_length]

# Análisis descriptivo
print("Media de Variable1:", np.mean(variable1))
print("Desviación estándar de Variable1:", np.std(variable1))
print("Media de Variable2:", np.mean(variable2))
print("Desviación estándar de Variable2:", np.std(variable2))

# Gráfico de dispersión
plt.figure(figsize=(8, 6))
sns.scatterplot(x=variable1, y=variable2)
plt.title("Gráfico de dispersión entre Variable1 y Variable2")
plt.xlabel("Variable1")
plt.ylabel("Variable2")
plt.show()

# Prueba de correlación
correlacion, p_valor = stats.pearsonr(variable1, variable2)
print("Coeficiente de correlación:", correlacion)
print("Valor p:", p_valor)
