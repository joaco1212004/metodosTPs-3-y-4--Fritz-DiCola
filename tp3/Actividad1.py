import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Cargar el dataset
df = pd.read_csv('dataset03.csv', index_col=0)  # index_col=0 para usar la primera columna como índice si aplica
X = df.values

# Paso 2: Descomponer la matriz X utilizando SVD
def reduce_dimensionality_svd(X, d):
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    U_d = U[:, :d]
    S_d = np.diag(S[:d])
    VT_d = VT[:d, :]
    X_reduced = np.dot(U_d, np.dot(S_d, VT_d))
    return X_reduced

# Paso 3: Determinar la similaridad par-a-par en el espacio de dimensión X usando operaciones vectorizadas
def gaussian_similarity(X, sigma=1.0):
    sq_dists = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2)
    similarity_matrix = np.exp(-sq_dists / (2 * sigma ** 2))
    return similarity_matrix

# Parámetro sigma
sigma = 1.0
similarity_matrix_X = gaussian_similarity(X, sigma)

# Paso 4: Reducir la dimensionalidad y calcular matrices de similaridad en los espacios reducidos
d_values = [2, 6, 10]
reduced_datasets = {}
similarity_matrices_reduced = {}

for d in d_values:
    Z = reduce_dimensionality_svd(X, d)
    reduced_datasets[d] = Z
    similarity_matrix_Z = gaussian_similarity(Z, sigma)
    similarity_matrices_reduced[d] = similarity_matrix_Z

# Paso 5: Comparar las medidas de similaridad
def plot_similarity_matrix(matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap='viridis')
    plt.title(title)
    plt.show()

# Visualizar la matriz de similaridad en el espacio original
plot_similarity_matrix(similarity_matrix_X, 'Matriz de Similaridad en el Espacio Original')

# Visualizar las matrices de similaridad en los espacios reducidos
for d in d_values:
    plot_similarity_matrix(similarity_matrices_reduced[d], f'Matriz de Similaridad en el Espacio Reducido a d={d}')

# Imprimir una parte de las matrices de similaridad para verificar
print("Matriz de Similaridad en el Espacio Original:")
print(similarity_matrix_X[:5, :5])  # Imprime las primeras 5 filas y columnas

for d in d_values:
    print(f"Matriz de Similaridad en el Espacio Reducido a d={d}:")
    print(similarity_matrices_reduced[d][:5, :5])  # Imprime las primeras 5 filas y columnas
