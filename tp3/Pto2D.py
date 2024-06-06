import os
import numpy as np
from PIL import Image
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import norm
import matplotlib.pyplot as plt


def load_images_from_folder(folder_path):
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.png', '.jpg', '.jpeg'))]
    images = [Image.open(image_file) for image_file in image_files]
    images = [np.array(image.convert('L')).flatten() for image in images]
    return np.array(images)


folder_path1 = 'datasets_imgs'
folder_path2 = 'datasets_imgs_02'

X2 = load_images_from_folder(folder_path2)
n2, p_squared2 = X2.shape
p2 = int(np.sqrt(p_squared2))

error_threshold = 0.10
min_d = None


for d in range(1, p_squared2):
    svd = TruncatedSVD(n_components=d)
    X2_reduced = svd.fit_transform(X2)
    X2_reconstructed = svd.inverse_transform(X2_reduced)
    error = norm(X2 - X2_reconstructed, 'fro') / norm(X2, 'fro')
    if error <= error_threshold:
        min_d = d
        break

print(f"El valor mínimo de d para que el error no exceda el 10% es: {min_d}")


X1 = load_images_from_folder(folder_path1)
X1_reduced = svd.transform(X1)
X1_reconstructed = svd.inverse_transform(X1_reduced)

reconstruction_error_1 = norm(X1 - X1_reconstructed, 'fro') / norm(X1, 'fro')
print(f"El error de reconstrucción para las imágenes del dataset 1 usando d = {min_d} es: {reconstruction_error_1:.4f}")


X2 = load_images_from_folder(folder_path2)
X2_reduced = svd.transform(X2)
X2_reconstructed = svd.inverse_transform(X2_reduced)

reconstruction_error_2 = norm(X2 - X2_reconstructed, 'fro') / norm(X2, 'fro')
print(f"El error de reconstrucción para las imágenes del dataset 2 usando d = {min_d} es: {reconstruction_error_2:.4f}")


fig, axes = plt.subplots(1, min(X1.shape[0], 5), figsize=(15, 5))
fig.suptitle(f'Reconstrucción del dataset 1 con d = {min_d}')
for i, ax in enumerate(axes):
    ax.imshow(X1_reconstructed[i].reshape((p2, p2)), cmap='gray')
    ax.axis('off')
plt.show()


fig, axes = plt.subplots(1, min(X2.shape[0], 5), figsize=(15, 5))
fig.suptitle(f'Reconstrucción del dataset 2 con d = {min_d}')
for i, ax in enumerate(axes):
    ax.imshow(X2_reconstructed[i].reshape((p2, p2)), cmap='gray')
    ax.axis('off')
plt.show()
