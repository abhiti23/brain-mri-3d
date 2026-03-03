import numpy as np
import math

var = np.load("var_train.npy")
print(var.shape)

N = 3659572
factors = []

for x in range(1, 300):
    if N % x == 0:
        for y in range(1, 300):
            if (N // x) % y == 0:
                z = N // (x*y)
                if z < 300:
                    factors.append((x,y,z))

print(factors)