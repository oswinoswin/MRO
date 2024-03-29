import math
import random
import decimal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd

def calculate_center(R, dimension):
    center = []
    for w in range(0, dimension):
        center.append(R)
    return center


def calculate_distance(vectorA, vectorB):
    #calculates distance from the 0-point
    dist = 0
    for a, b in zip(vectorA, vectorB):
        dist = dist + (a-b)*(a-b)
    return math.sqrt(dist)


def is_in_sphere(dist, R):
    return dist < R


def random_point(MaxSize, dimension):
    point = []
    for x in range(0, dimension):
        point.append(random.uniform(0, MaxSize))
    return point


def experiment(N, dimension, R):
    MaxSize = 2*R
    center = calculate_center(R, dimension)
    in_sphere = 0
    for i in range(1, N):
        point = random_point(MaxSize, dimension)
        distance = calculate_distance(point, center)
        #print('distance: {}, point: {}, in_sphere: {}'.format(distance, point, is_in_sphere(distance, R)))
        if is_in_sphere(distance, R):
            in_sphere += 1
    return in_sphere/N


if __name__ == '__main__':
    #checking points number, n = 1000
    decimal.getcontext().prec = 4
    expected = 0.5235
    R = 0.5
    MazSize = 2*R
    result = []
    with open('data/eggs5.csv', 'w', newline='') as csvfile:
        csvfile.writelines('{},{},{}\n'.format("points", "value", "error"))
        for n in range( 3000):
            resulcik = experiment(n, 3, R)
            #print('{}, {}, {}'.format(n,resulcik, resulcik - expected ))
            csvfile.writelines('{0:.0f}, {1:.4f}, {2:.4f}\n'.format(n, resulcik, abs(resulcik - expected)))
            result.append([n, resulcik, abs(resulcik - expected)])




