"""
==========
Kernel PCA
==========

This example shows that Kernel PCA is able to find a projection of the data
that makes data linearly separable.
"""
from PIL import Image, ImageDraw

print(__doc__)

# Authors: Mathieu Blondel
#          Andreas Mueller
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles


im = Image.open("inB/small1.bmp")
(width, height) = im.size
px = im.load()

im_copy = im.copy()

X_base = []
X_colored = []


def get_color(r, g, b):
    if (r, g, b) == (255, 216, 0):
        return "yellow"
    if (r, g, b) == (0, 255, 0):
        return "green"
    if (r, g, b) == (0, 38, 255):
        return "blue"
    return "red"

def load_data():
    for col in range(1, width):
        for row in range(1, height):
            print(px[row, col])
            if px[row,col] == (255, 255, 255):
                continue
            else:
                X_base.append([row, col])
                (r, g, b ) = px[row, col]
                X_colored.append(get_color(r, g, b))

load_data()


np.random.seed(0)
X_new = np.array(X_base)
X, y = make_circles(n_samples=400, factor=.3, noise=.05)
print("X size: {}, X_new size: {}".format(len(X), len(X_new)))
print("X\n {} ".format(X))
print("X_new\n {} ".format(X_new))
print("y\n {} ".format(y))
print("X_colored \n {} ".format(X_colored))

X = X_new
k_type = "rbf"
gamma_val=20

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=20)
X_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(X)

# Plot results


#plt.subplots(2)

plt.subplot(1, 2, 1, aspect='equal')
plt.title("PCA")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=X_colored,
            s=20, edgecolor='k')

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.subplot(1, 2, 2, aspect='equal')
plt.title("Kernel pca {} {}".format(k_type, gamma_val))
plt.scatter(X_kpca[:,0], X_kpca[:, 1], c=X_colored,
            s=20, edgecolor='k')
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")



'''
X1, X2 = np.meshgrid(np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50))
X_grid = np.array([np.ravel(X1), np.ravel(X2)]).T
# projection on the first principal component (in the phi space)
Z_grid = kpca.transform(X_grid)[:, 0].reshape(X1.shape)
plt.contour(X1, X2, Z_grid, colors='grey', linewidths=1, origin='lower')

plt.subplot(2, 2, 2, aspect='equal')
plt.scatter(X_pca[reds, 0], X_pca[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_pca[blues, 0], X_pca[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd component")

plt.subplot(2, 2, 3, aspect='equal')
plt.scatter(X_kpca[reds, 0], X_kpca[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_kpca[blues, 0], X_kpca[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Projection by KPCA")
plt.xlabel("1st principal component in space induced by $\phi$")
plt.ylabel("2nd component")

plt.subplot(2, 2, 4, aspect='equal')
plt.scatter(X_back[reds, 0], X_back[reds, 1], c="red",
            s=20, edgecolor='k')
plt.scatter(X_back[blues, 0], X_back[blues, 1], c="blue",
            s=20, edgecolor='k')
plt.title("Original space after inverse transform")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
'''
#plt.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.04, 0.35)

plt.show()
