import math
import random


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
        point.append(random.randint(0, MaxSize))
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
    #for n in range(10, 100000):
    print('{}'.format(experiment(1000, 5, 100000)))

