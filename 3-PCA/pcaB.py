import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

im = Image.open("inB/test1.bmp")
(width, height) = im.size
px = im.load()

im_copy = im.copy()

X_base = []
X_colored = []

im_out = Image.new("RGB", im.size, "#fff")
px2 = im_out.load()


def load_data():
    for col in range(1, width):
        for row in range(1, height):
            (r, g, b) = px[row, col]
            print(px[row, col])
            if (r, g, b) == (255, 255, 255):
                continue
            else:
                X_base.append([row, col])
                X_colored.append([px[row, col]])


if __name__ == '__main__':
    load_data()
    X = np.array(X_base)
    print(X)
    pca = PCA(n_components=2)

    x_transformed = pca.fit_transform(X).tolist()

    print('X_base.len = {}, X_colored.len = {}, x_transformed.len = {}'.format(len(X_base), len(X_colored), len(x_transformed)))

    for i, point in enumerate(x_transformed):
        row = point[0]
        col = point[1]
        if row >=0 and row <= height and col >= 0 and col <= width:
            px2[row, col] = X_colored[i][0]

    im_copy.save('outB/test1.bmp')
    im_out.save('outB/new.bmp')
