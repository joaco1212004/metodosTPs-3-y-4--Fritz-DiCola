import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def load_and_preprocess_images(folder, image_shape):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img = imread(os.path.join(folder, filename))
            if img.ndim == 3:
                img = rgb2gray(img)
            img_resized = resize(img, image_shape, anti_aliasing=True)
            images.append(img_resized.flatten())
    return np.array(images)


image_shape = (128, 128)
X_images_1 = load_and_preprocess_images('datasets_imgs', image_shape)
X_images_2 = load_and_preprocess_images('datasets_imgs_02', image_shape)

def compress_images(X, d):
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    S_d = np.diag(S[:d])
    X_compressed = np.dot(np.dot(U[:, :d], S_d), VT[:d, :])
    return X_compressed



def calculate_similarity_matrix(X_compressed, metric='cosine'):
    if metric == 'cosine':
        similarity_matrix = cosine_similarity(X_compressed)
    elif metric == 'euclidean':
        similarity_matrix = euclidean_distances(X_compressed)
        similarity_matrix = 1 / (1 + similarity_matrix)  
    else:
        raise ValueError("Métrica no soportada. Use 'cosine' o 'euclidean'.")
    return similarity_matrix


def plot_similarity_matrix(similarity_matrix, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()

dimensions = [5, 10, 20, 50]


similarity_matrices = {}
for d in dimensions:
    X_compressed_1 = compress_images(X_images_1, d)
    similarity_matrix = calculate_similarity_matrix(X_compressed_1, metric='cosine')
    similarity_matrices[d] = similarity_matrix
    plot_similarity_matrix(similarity_matrix, f'Similaridad entre Imágenes (d={d})')


plt.figure(figsize=(15, 10))
for i, d in enumerate(dimensions[:2], 1):
    X_compressed_1 = compress_images(X_images_1, d)
    similarity_matrix = calculate_similarity_matrix(X_compressed_1, metric='cosine')
    similarity_matrices[d] = similarity_matrix
    plt.subplot(1, 2, i)
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(f'Similaridad entre Imágenes (d={d})')
plt.show()


plt.figure(figsize=(15, 10))
for i, d in enumerate(dimensions[2:], 1):
    X_compressed_1 = compress_images(X_images_1, d)
    similarity_matrix = calculate_similarity_matrix(X_compressed_1, metric='cosine')
    similarity_matrices[d] = similarity_matrix
    plt.subplot(1, 2, i)
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar()
    plt.title(f'Similaridad entre Imágenes (d={d})')
plt.show()


for d in dimensions:
    print(f"Análisis para d={d}:")
    sim_matrix = similarity_matrices[d]
    for i in range(sim_matrix.shape[0]):
        for j in range(i + 1, sim_matrix.shape[1]):
            if sim_matrix[i, j] > 0.9:  
                print(f"Imágenes {i} y {j} son muy similares (similaridad={sim_matrix[i, j]:.2f})")

    print(f"Matriz de similaridad para d={d}:")
    print(sim_matrix)
