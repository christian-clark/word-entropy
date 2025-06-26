import math
import numpy as np
import random

VALUES = 100
XMIN = 1
XMAX = 10000
SAMPLES = 100000


def renyi_entropy(x, alpha=1/2):
    total = 0
    for xi in x:
        total += math.pow(xi, alpha)
    return math.log2(total) / (1-alpha)


x = [random.randint(XMIN, XMAX) for _ in range(VALUES)]
total = sum(x)
x = np.array([i/total for i in x])
print("renyi entropy of x:", renyi_entropy(x))

sample = np.random.choice(x, SAMPLES, replace=True, p=x)
sample_total = sum(1/math.sqrt(si) for si in sample)
estimated_ent = 2 * math.log2(sample_total/SAMPLES)
print("estimated renyi entropy of x:", estimated_ent)