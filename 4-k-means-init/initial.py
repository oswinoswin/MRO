import random
import itertools
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


im = Image.open("in/image1.bmp")
(width, height) = im.size
px = im.load()

data = []
colors = []

fig = plt.figure()
plots = []
legends = []
color_list = ["#7e1e9c", "#15b01a", "#0343df", "#95d0fc", "#c20078", "#c79fef", "#ff796c", "#ff028d", "#fd3c06"]

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


def euclid(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def si(a, b):
    return (a - b) / max(a, b)


def avarage_distance(elements, point):  # avarage distance from el to other el in cluster
    sum = 0
    for el in elements:
        sum += euclid(el, point)
    return sum / len(elements)


def min_distance(elements, point):  # assume point is not in elements
    distances = np.array([euclid(point, el) for el in elements])
    return distances.min()


def silhouette(points_and_labels, point):
    point_label =  [x for x in points_and_labels if x[0] == point ][0][1]
    cluster = [x[0] for x in points_and_labels if x[1] == point_label]
    other_clusters = [x[0] for x in points_and_labels if x[1] != point_label]
    ai = avarage_distance(cluster, point)
    bi = min_distance(other_clusters, point)
    return si(ai, bi)

def show_silhouette(labels, method, iterations):
    lp = list(zip(data, labels))
    plt.title("Silhouette index for {} after {} iterations".format(method, iterations))
    sil_k_means_plus = [silhouette(lp, p) for p in data]

    colored = [color_list[c] for c in list(labels)]

    plt.xlabel("point")
    plt.ylabel("silhouette index")

    plt.bar(range(len(sil_k_means_plus)), sil_k_means_plus, color=colored)
    plt.show()


load_data()

data_np = np.array(data)
max_it_range = range(1, 31)
n_runs = 5
inertia_k_means_random = np.zeros((len(max_it_range), n_runs))
inertia_k_means_partition = np.zeros((len(max_it_range), n_runs))
inertia_k_means_forgy = np.zeros((len(max_it_range), n_runs))
inertia_k_means_plus = np.zeros((len(max_it_range), n_runs))

k_means_random = []
k_means_partition = []
k_means_forgy = []
k_means_plus = []

for j, max_it in enumerate(max_it_range):
    for i in range(n_runs):
        k_means_random = KMeans(n_clusters=9, random_state=i, init='random', n_init=1, max_iter=max_it).fit(data_np)
        k_means_partition = KMeans(n_clusters=9, random_state=8, init=random_partition_init(data_np, 9), n_init=1,
                                   max_iter=800).fit(data_np)
        k_means_forgy = KMeans(n_clusters=9, random_state=8, init=forgy_init(data_np, 9), n_init=1, max_iter=800).fit(
            data_np)

        k_means_plus = KMeans(n_clusters=9, random_state=i, init='k-means++', n_init=1, max_iter=max_it).fit(data_np)

        inertia_k_means_random[j, i] = k_means_random.inertia_
        inertia_k_means_partition[j, i] = k_means_partition.inertia_
        inertia_k_means_forgy[j, i] = k_means_forgy.inertia_
        inertia_k_means_plus[j, i] = k_means_plus.inertia_
    if max_it == 30:
        show_silhouette(k_means_plus.labels_, "k-means++", max_it)
'''
ax = plt.gca()
ax.set_color_cycle(['red', 'green', 'blue', 'orange'])
plt.ylim([0, 18000])

p_random = plt.errorbar(max_it_range, inertia_k_means_random.mean(axis=1), inertia_k_means_random.std(axis=1))
plots.append(p_random[0])
legends.append("k-means with random init")

p_partition = plt.errorbar(max_it_range, inertia_k_means_partition.mean(axis=1), inertia_k_means_partition.std(axis=1))
plots.append(p_partition[0])
legends.append("k-means with random partition init")

p_forgy = plt.errorbar(max_it_range, inertia_k_means_forgy.mean(axis=1), inertia_k_means_forgy.std(axis=1))
plots.append(p_forgy[0])
legends.append("k-means with forgy init")

p_plus = plt.errorbar(max_it_range, inertia_k_means_plus.mean(axis=1), inertia_k_means_plus.std(axis=1))
plots.append(p_plus[0])
legends.append("k-means++")



plt.title("k-means")
plt.xlabel("max no of iterations")
plt.ylabel("inertia")
plt.legend(plots, legends)
plt.show()'''

