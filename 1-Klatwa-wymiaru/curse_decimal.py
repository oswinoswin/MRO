import math
import random
import decimal as d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import pandas as pd

def calculate_center(R, dimension):
    center = []
    for w in range(0, dimension):
        center.append(d.Decimal(R))
    return center


def calculate_distance(vectorA, vectorB):
    #calculates distance from the 0-point
    dist = d.Decimal(0)
    for a, b in zip(vectorA, vectorB):
        dist = dist + d.Decimal((a-b)*(a-b))
    return dist.sqrt()


def is_in_sphere(dist, R):
    return dist < R


def random_point(MaxSize, dimension):
    point = []
    for x in range(0, dimension):
        point.append(d.Decimal(str(random.uniform(0, 1))))
    return point


def experiment(N, dimension, R):
    MaxSize = d.Decimal(2)*d.Decimal(R)
    center = calculate_center(R, dimension)
    in_sphere = d.Decimal(0)
    for i in range(1, N):
        point = random_point(MaxSize, dimension)
        distance = calculate_distance(point, center)
        #print('distance: {}, point: {}, in_sphere: {}'.format(distance, point, is_in_sphere(distance, R)))
        if is_in_sphere(distance, R):
            in_sphere = in_sphere + d.Decimal(1)
    if d.Decimal(N)  != d.Decimal(0):
        return d.Decimal(in_sphere/d.Decimal(N))
    return d.Decimal(0)


if __name__ == '__main__':
    #checking points number, n = 1000
    d.getcontext().prec = 8
    expected = d.Decimal(0.5235)
    R = d.Decimal(0.5)
    MazSize = 1.0
    result = []
    with open('data/eggs6.csv', 'w', newline='') as csvfile:
        csvfile.writelines('{},{},{}\n'.format("points", "value", "error"))
        for n in range(1000, 1500):
            resulcik = d.Decimal(experiment(n, 3, R))
            #print('{}, {}, {}'.format(n,resulcik, resulcik - expected ))
            csvfile.writelines('{}, {}, {}\n'.format(d.Decimal(n), d.Decimal(resulcik), d.Decimal(resulcik) - d.Decimal(expected)))
            result.append([n, resulcik, abs(resulcik - expected)])




