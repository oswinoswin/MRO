import numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA


im = Image.open("inB/test1.bmp")
(width, height) = im.size
px = im.load()

im_copy = im.copy()

X_base = []
X_colored = []

im_out = Image.new("RGB", im.size, "#fff")
px2 = im_out.load()

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
            (r, g, b) = px[row, col]

            if (r, g, b) == (255, 255, 255):
                continue
            else:
                X_base.append([row - height, col - width])
                X_colored.append([px[row, col]])


if __name__ == '__main__':
    load_data()
    X = np.array(X_base)
    #print(X)
    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    X_kpca = kpca.fit_transform(X)
    X_back = kpca.inverse_transform(X_kpca)
    pca = PCA()
    X_pca = pca.fit_transform(X)

    x_transformed = X_pca.tolist()

    eigenvectors = pca.components_
    eigen_val = pca.explained_variance_ratio_

    print(eigenvectors[0]*eigen_val[0])
    print(eigenvectors[1]*eigen_val[1])

    draw = ImageDraw.Draw(im_copy)
    draw.line((0, 0) + im.size, fill=255)
    del draw


    for i, point in enumerate(x_transformed):
        row = point[0]
        col = point[1]
        if row >=0 and row <= height and col >= 0 and col <= width:
            px2[row, col] = X_colored[i][0]

    im_copy.save('outB/test1_vectors.bmp')
    im_out.save('outB/new2.bmp')
