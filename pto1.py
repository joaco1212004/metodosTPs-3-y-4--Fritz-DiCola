#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Cargar dataset
file_path = 'dataset03.csv'
y_path = 'y3.txt'
dataset = pd.read_csv(file_path)
y = np.loadtxt(y_path)

# Función para calcular la matriz de similaridad
def calcular_matriz_similaridad(X, sigma=1.0):
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    K = np.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    return K

# Dataset original sin la primera columna (índice)
X_original = dataset.iloc[:, 1:].values

# Normalizar los datos
scaler = StandardScaler()
X_original_normalized = scaler.fit_transform(X_original)

# Calcular la matriz de similaridad original
similaridad_original = calcular_matriz_similaridad(X_original_normalized)

# Realizar la reducción de dimensionalidad usando PCA para diferentes valores de d
d_values = [2, 6, 10]
similaridades_reducidas = {}
X_reducidos = {}
componentes_principales = {}

for d in d_values:
    pca = PCA(n_components=d)
    X_reducido = pca.fit_transform(X_original_normalized)
    X_reducidos[d] = X_reducido
    similaridades_reducidas[d] = calcular_matriz_similaridad(X_reducido)
    componentes_principales[d] = pca.components_

# Mostrar las matrices de similaridad
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Matriz de similaridad original
sns.heatmap(similaridad_original, ax=axes[0], cmap='viridis')
axes[0].set_title('Similaridad Original')

# Matrices de similaridad reducidas
for i, d in enumerate(d_values):
    sns.heatmap(similaridades_reducidas[d], ax=axes[i+1], cmap='viridis')
    axes[i+1].set_title(f'Similaridad Reducida (d={d})')

plt.show()

# Clustering y visualización de clusters para d=2, 6, 10
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for i, d in enumerate(d_values):
    X_reducido = X_reducidos[d]
    kmeans = KMeans(n_clusters=2)  # Suponemos que hay 2 clusters
    clusters = kmeans.fit_predict(X_reducido)
    
    axes[i].scatter(X_reducido[:, 0], X_reducido[:, 1], c='blue', marker='o')
    axes[i].set_xlabel('Componente Principal 1')
    axes[i].set_ylabel('Componente Principal 2')
    axes[i].set_title(f'Clusters en el Espacio Reducido (d={d})')
    axes[i].grid(True)

plt.show()

# Realizar SVD
U, S, VT = np.linalg.svd(X_original_normalized, full_matrices=False)

# Graficar los autovectores (columnas de V^T)
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

for i, d in enumerate(d_values):
    V_d = VT[:d, :]
    for j in range(d):
        axes[i].plot(V_d[j], label=f'Autovector {j+1}')
    axes[i].set_title(f'Autovectores para d={d}')
    axes[i].legend()
    axes[i].grid(True)

plt.show()

# Identificar las dimensiones originales más importantes para cada d
importances = {}
for d in d_values:
    V_d = VT[:d, :]
    importancia = np.sum(np.abs(V_d), axis=0)  # Sumar las contribuciones absolutas de cada dimensión original
    dimensiones_mas_importantes = np.argsort(importancia)[-3:][::-1]  # Obtener las 3 dimensiones más importantes
    importances[d] = dimensiones_mas_importantes
    print(f"\nPara d={d}, las dimensiones originales más importantes son: {dimensiones_mas_importantes + 1}")

# Método para determinar las dimensiones más importantes
print("\nMétodo utilizado: Análisis de los autovectores de la matriz V^T obtenidos por SVD y la magnitud de las contribuciones absolutas.")

# Indicar las dimensiones originales más importantes
print("\nResumen de dimensiones originales más importantes para cada d:")
for d in d_values:
    dimensiones = importances[d] + 1
    print(f"Para d={d}, las dimensiones originales más importantes son: {dimensiones}")

# Ejercicio 3: Predicción con regresión lineal

# Entrenar y evaluar el modelo para diferentes dimensiones d
errors = []

for d in d_values:
    # Reducir dimensionalidad con PCA
    pca = PCA(n_components=d)
    X_reducido = pca.fit_transform(X_original_normalized)
    
    # Ajustar el modelo de regresión lineal
    reg = LinearRegression()
    reg.fit(X_reducido, y)
    
    # Predicciones
    y_pred = reg.predict(X_reducido)
    
    # Calcular error
    error = mean_squared_error(y, y_pred)
    errors.append(error)
    
    print(f'Error para d={d}: {error}')

# Encontrar la mejor dimensión d
best_d_index = np.argmin(errors)
best_d = d_values[best_d_index]

print(f'\nLa mejor dimensión d que minimiza el error es: {best_d}')

# Volver a ajustar el modelo con la mejor dimensión d
pca_best = PCA(n_components=best_d)
X_reducido_best = pca_best.fit_transform(X_original_normalized)
reg_best = LinearRegression()
reg_best.fit(X_reducido_best, y)
y_pred_best = reg_best.predict(X_reducido_best)

# Identificar las muestras con mejor predicción
errors_best = np.abs(y - y_pred_best)
best_predictions_indices = np.argsort(errors_best)[:5]  # Top 5 mejores predicciones
print(f'\nÍndices de las muestras con mejor predicción: {best_predictions_indices}')
print(f'Valores reales: {y[best_predictions_indices]}')
print(f'Valores predichos: {y_pred_best[best_predictions_indices]}')

# Ajustar el modelo en el espacio original
reg_original = LinearRegression()
reg_original.fit(X_original_normalized, y)
beta = reg_original.coef_
print(f'\nPesos asignados a cada dimensión original (vector β):\n{beta}')

# Gráfica del error en función de d
plt.figure(figsize=(8, 6))
plt.plot(d_values, errors, marker='o')
plt.xlabel('Dimensión d')
plt.ylabel('Error de predicción (MSE)')
plt.title('Error de predicción en función de la dimensión d')
plt.grid(True)
plt.show()

# Gráfica de los pesos β para cada dimensión original
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(beta) + 1), beta)
plt.xlabel('Dimensiones originales')
plt.ylabel('Peso (β)')
plt.title('Pesos asignados a cada dimensión original (vector β)')
plt.grid(True)
plt.show()


# %%
