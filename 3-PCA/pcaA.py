import math
import random
import decimal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd

from sklearn.decomposition import PCA

def calculate_center(dimension):
    center = []
    for w in range(0, dimension):
        center.append(0)
    return center


def distance_from_center(point):
    dist = 0
    for a in point:
        dist = dist + a * a
    return math.sqrt(dist)


def is_in_sphere(point, R):
    return distance_from_center(point) <= R


def random_point(R, dimension):
    return [random.randint(0, 2 * R) - R for x in range(0, dimension)]


def generate_points(R, dimension, N):
    return [random_point(R, dimension) for x in range(0, N)]

def is_vertex(point, R):
    for x in point:
        if abs(x) is not R:
            return False
    return True

def get_color(point, R):
    if is_in_sphere(point, R):
        return 'green'
    if is_vertex(point, R):
        return 'red'
    return 'blue'


if __name__ == '__main__':
    R = 10
    N = 2
    M = 10000

    X_base = generate_points(R, N, M)
    X_colored = [ get_color(point, R) for point in X_base]
    X = np.array(X_base)
    print(X)
    pca = PCA(n_components=2)

    x_transformed = pca.fit_transform(X).tolist()

    x_cord = [point[0] for point in x_transformed]
    y_cord = [point[1] for point in x_transformed]

    plt.scatter(x_cord, y_cord, c=X_colored, marker='o')
    plt.suptitle("N = {}".format(N))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()



    '''with open('data/eggs5.csv', 'w', newline='') as csvfile:
        csvfile.writelines('{},{},{}\n'.format("points", "value", "error"))
        for n in range(3000):
            resulcik = experiment(n, 3, R)
            # print('{}, {}, {}'.format(n,resulcik, resulcik - expected ))
            csvfile.writelines('{0:.0f}, {1:.4f}, {2:.4f}\n'.format(n, resulcik, abs(resulcik - expected)))
            result.append([n, resulcik, abs(resulcik - expected)])'''
