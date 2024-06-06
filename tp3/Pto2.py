import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize


image_folder = 'datasets_imgs'
images = []
image_shape = (28, 28)  

for filename in os.listdir(image_folder):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        img = imread(os.path.join(image_folder, filename))
        if img.ndim == 3:
            img = rgb2gray(img)
        img_resized = resize(img, image_shape, anti_aliasing=True)
        images.append(img_resized.flatten())


X_images = np.array(images)


def compress_images(X, d):
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    S_d = np.diag(S[:d])
    X_compressed = np.dot(np.dot(U[:, :d], S_d), VT[:d, :])
    return X_compressed


dimensions = [5, 10, 20, 50]


for d in dimensions:
    X_compressed = compress_images(X_images, d)
    plt.figure(figsize=(10, 10))
    for i in range(min(10, X_compressed.shape[0])):  
        plt.subplot(1, 10, i + 1)
        plt.imshow(X_compressed[i].reshape(image_shape), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Imágenes Reconstruidas (d={d})')
    plt.show()
