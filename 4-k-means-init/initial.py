import random

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

im = Image.open("in/image1.bmp")
(width, height) = im.size
px = im.load()

im_copy = im.copy()

X_base = []
X_colored = []

data = []
colors = []


def get_color(r, g, b):
    if (r, g, b) == (255, 216, 0):
        return "yellow"
    if (r, g, b) == (0, 255, 0):
        return "green"
    if (r, g, b) == (255, 0, 0):
        return "red"
    return "blue"


def load_data():
    for row in range(1, height):
        for col in range(1, width):
            if px[col, row] == (255, 255, 255):
                continue
            else:
                data.append([col, row])
                (r, g, b) = px[col, row]
                colors.append(get_color(r, g, b))


def forgy_init(dane, k):
    elements = random.sample(list(dane), k)
    return np.array(elements)


def chunk(xs, n):
    ys = list(xs)
    random.shuffle(ys)
    ylen = len(ys)
    size = int(ylen / n)
    chunks = [ys[0 + size * i: size * (i + 1)] for i in range(n)]
    leftover = ylen - size * n
    edge = size * n
    for i in range(leftover):
        chunks[i % n].append(ys[edge + i])
    return chunks


def random_partition_init(dane, k):
    tmp = chunk(dane, k)

    result = []

    for row in tmp:
        x = np.array(row)[:, 0]
        y = np.array(row)[:, 1]
        result.append([int(np.mean(x)), int(np.mean(y))])

    return np.array(result)


load_data()

data_np = np.array(data)
max_it_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 200, 300, 400, 600, 800, 1000, 1200]
n_runs = 5
inertia_k_means_random = np.zeros((len(max_it_range), n_runs))
inertia_k_means_plus = np.zeros((len(max_it_range), n_runs))

for j, max_it in enumerate(max_it_range):
    print(inertia_k_means_random[j])
    for i in range(n_runs):
        print("i = {}, j = {}".format(i, j))
        k_means_random = KMeans(n_clusters=9, random_state=i, init='random', n_init=1, max_iter=max_it).fit(data_np)
        k_means_plus = KMeans(n_clusters=9, random_state=i, init='k-means++', n_init=1, max_iter=max_it).fit(data_np)
        inertia_k_means_random[j, i] = k_means_random.inertia_
        inertia_k_means_plus[j, i] = k_means_plus.inertia_
    print(inertia_k_means_random[j])

print(inertia_k_means_random)
print(inertia_k_means_plus)

k_means_random_partition = KMeans(n_clusters=9, random_state=8, init=random_partition_init(data_np, 9), n_init=1,
                                  max_iter=800).fit(data_np)
k_means_forgy = KMeans(n_clusters=9, random_state=8, init=forgy_init(data_np, 9), n_init=1, max_iter=800).fit(data_np)
k_means_plus = KMeans(n_clusters=9, random_state=8, init='k-means++', n_init=5, max_iter=800).fit(data_np)

lol = np.array(k_means_plus.cluster_centers_)
print(lol)
print(lol.max())

plt.title("k-means")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(data_np[:, 0], data_np[:, 1], c=k_means_forgy.labels_)
#plt.show()


'''
plt.subplot(1, 1, 1, aspect='equal')
plt.title("kpca {}".format(k_type))
plt.scatter(X_kpca[:,0], X_kpca[:, 1], c=X_colored)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")'''
