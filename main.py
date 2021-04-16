import pylab
import numpy as np
from scipy.stats import poisson
import math
import matplotlib.patches
import matplotlib.lines
import matplotlib.path
import matplotlib.pyplot as plt

def distr(x, y, p):
    return 1 / (2 * math.pi * math.sqrt(1 - p * p)) * math.exp(-1 / (2 * (1 - p * p)) * (x * x - 2 * p * x * y + y * y))

def average(x, n):
    result = 0.
    for i in range(n):
        result += x[i]
    result /= n
    return result

def quad_average(x, n):
    result = 0.
    for i in range(n):
        result += x[i] * x[i]
    result /= n
    return result

def despertion(x, size):
    y = average(x, size)
    result = 0.
    for i in x:
        result += (i - y) * (i - y)
    result /= size
    return result

def pirson_cov(arr, n):
    result = 0.
    x = 0.
    y = 0.
    for i in range(n):
        result += arr[i][0] * arr[i][1]
        x += arr[i][0] * arr[i][0]
        y += arr[i][1] * arr[i][1]
    result /= math.sqrt(x * y)
    return result

def mediana(x, size):
    if size % 2 == 1:
        num = (size - 1) / 2 + 1
        result = x.index(num)
    else:
        num = math.ceil(size / 2)
        result = (x[num] + x[num + 1]) / 2.
    return result

def quad_cov(arr, size):
    med_x = mediana(arr[:, 0], size)
    med_y = mediana(arr[:, 1], size)
    n1 = 0
    n2 = 0
    for i in range(size):
        if arr[i][0] > med_x:
            if arr[i][1] > med_y:
                n1 += 1
            else:
                n2 += 1
        else:
            if arr[i][1] > med_y:
                n2 += 1
            else:
                n1 += 1
    return ((n1 - n2) / size)

def ranks(arr, size):
    result = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if arr[j] <= arr[i]:
                result[i] += 1
    return result

def spirman_cov(arr, size):
    x = arr[:, 0]
    y = arr[:, 1]
    u = ranks(x, size)
    v = ranks(y, size)
    n = (size + 1) / 2
    result = 0.
    x = 0.
    y = 0.
    for i in range(size):
        result += (u[i] - n) * (v[i] - n)
        x += (u[i] - n) * (u[i] - n)
        y += (v[i] - n) * (v[i] - n)
    result /= math.sqrt(x * y)
    return result

p = {0., 0.5, 0.9}
arr = {20, 60, 100}
for i in p:
    for j in arr:
        pirson = np.zeros(1000)
        quad = np.zeros(1000)
        spirman = np.zeros(1000)
        for q in range(1000):
            x = np.random.multivariate_normal((0, 0), [[1, i], [i, 1]], j)
            pirson[q] = pirson_cov(x, j)
            quad[q] = quad_cov(x, j)
            spirman[q] = spirman_cov(x, j)
        print("Pirson", "Quadration", "Spirman", "Coefittient =", i, "Num points =", j)
        print(average(pirson, 1000), average(quad, 1000), average(spirman, 1000))
        print(quad_average(pirson, 1000), quad_average(quad, 1000), quad_average(spirman, 1000))
        print(despertion(pirson, 1000), despertion(quad, 1000), despertion(spirman, 1000))
        print()

        x = np.random.multivariate_normal((0, 0), [[1, i], [i, 1]], j)
        cov = [[1, i], [i, 1]]
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        from matplotlib.patches import Ellipse
        import matplotlib.pyplot as plt

        ax = plt.subplot(111, aspect='equal')
        ell = Ellipse(xy=(np.mean(x[:, 0]), np.mean(x[:, 1])),
                      width=lambda_[0] * 6, height=lambda_[1] * 6,
                     angle=np.rad2deg(np.arccos(v[0, 0])), color= 'black')
        ell.set_facecolor('none')
        ax.add_artist(ell)
        plt.scatter(x[:, 0], x[:, 1])
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.show()

for i in arr:
    pirson = np.zeros(1000)
    quad = np.zeros(1000)
    spirman = np.zeros(1000)
    for j in range(1000):
        x = 0.9 * np.random.multivariate_normal((0, 0), [[1, 0.9], [0.9, 1]], i) + 0.1 * np.random.multivariate_normal((0, 0), [[10, -9], [-9, 10]], i)
        pirson[j] = pirson_cov(x, i)
        quad[j] = quad_cov(x, i)
        spirman[j] = spirman_cov(x, i)
    print("Pirson", "Quadration", "Spirman", "Num points =", i)
    print(average(pirson, 1000), average(quad, 1000), average(spirman, 1000))
    print(quad_average(pirson, 1000), quad_average(quad, 1000), quad_average(spirman, 1000))
    print(despertion(pirson, 1000), despertion(quad, 1000), despertion(spirman, 1000))
    print()
