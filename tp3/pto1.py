#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar los datos
X = pd.read_csv('dataset03.csv').values
y = pd.read_csv('y3.txt', header=None).values.flatten()

print("Dimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

# Función de similitud no lineal (kernel gaussiano)
def gaussian_kernel(x1, x2, sigma=1.0):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * sigma ** 2))

# Matriz de similitud en el espacio original
sigma = 1.0
similarity_matrix = squareform(pdist(X, metric=lambda u, v: gaussian_kernel(u, v, sigma)))

# Verificar los valores de la matriz de similitud
print("Valores de la matriz de similitud original:")
print(similarity_matrix)

# Normalizar la matriz de similitud
similarity_matrix_normalized = (similarity_matrix - np.min(similarity_matrix)) / (np.max(similarity_matrix) - np.min(similarity_matrix))

# Visualización de la matriz de similitud original
plt.figure(figsize=(10, 8))
plt.title('Matriz de Similitud en el Espacio Original')
plt.imshow(similarity_matrix_normalized, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

# Función para graficar matriz de similitud en el espacio reducido
def plot_similarity_matrix(X_reduced, d):
    similarity_matrix_reduced = squareform(pdist(X_reduced, metric=lambda u, v: gaussian_kernel(u, v, sigma)))
    similarity_matrix_reduced_normalized = (similarity_matrix_reduced - np.min(similarity_matrix_reduced)) / (np.max(similarity_matrix_reduced) - np.min(similarity_matrix_reduced))
    plt.figure(figsize=(10, 8))
    plt.title(f'Matriz de Similitud en el Espacio Reducido (d={d})')
    plt.imshow(similarity_matrix_reduced_normalized, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

# Reducción de dimensionalidad y visualización de matrices de similitud
dimensions = [2, 6, 10]
for d in dimensions:
    pca = PCA(n_components=d)
    X_reduced = pca.fit_transform(X)
    plot_similarity_matrix(X_reduced, d)

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=10)
pca.fit(X)

# Componentes principales
components = pca.components_

# Importancia de cada dimensión original
importance = np.sum(np.abs(components), axis=0)

# Índices de las dimensiones más importantes
important_dims = np.argsort(importance)[::-1]

print("Dimensiones más importantes:", important_dims[:10])
print("Importancia:", importance[important_dims[:10]])

# Función para análisis de regresión
def regression_analysis(X, y, d):
    pca = PCA(n_components=d)
    X_reduced = pca.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_reduced, y)
    y_pred = model.predict(X_reduced)
    
    mse = mean_squared_error(y, y_pred)
    print(f'MSE para d={d}:', mse)
    return mse

# Evaluar el MSE para diferentes dimensiones
mse_values = {}
for d in dimensions:
    mse_values[d] = regression_analysis(X, y, d)

# Mejor dimensión para predicción
best_d = min(mse_values, key=mse_values.get)
print("Mejor dimensión para predicción:", best_d)

# %%