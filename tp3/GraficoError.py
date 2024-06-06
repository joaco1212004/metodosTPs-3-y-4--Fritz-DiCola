import os
import numpy as np
from PIL import Image
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import norm
import matplotlib.pyplot as plt


def load_images_from_folder(folder_path):
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = [np.array(Image.open(image_file).convert('L')).flatten() for image_file in image_files]
    return np.array(images)


folder_path = 'datasets_imgs'
X = load_images_from_folder(folder_path)

n, p_squared = X.shape
p = int(np.sqrt(p_squared))


d_values = range(1, 21)
errors = []


for d in d_values:
    svd = TruncatedSVD(n_components=d)
    X_reduced = svd.fit_transform(X)
    X_reconstructed = svd.inverse_transform(X_reduced)
    error = norm(X - X_reconstructed, 'fro') / norm(X, 'fro')
    errors.append(error)


plt.plot(d_values, errors, marker='o')
plt.title('Average Reconstruction Error vs d')
plt.xlabel('d')
plt.ylabel('Average Reconstruction Error')
plt.grid(True)
plt.show()

