# negative Sum of log of probabilities = Entropy
# Low entropy = good mode, therefore our goal is to minimise the entropy


import numpy as np
import math


def cross_entropy(Y, p):
    sum = 0
    for i in range(len(Y)):
        sum -= Y[i] * math.log(p[i]) + (1-Y[i])*math.log(1-p[i])
    return sum


def main():
    Y = [1,1,0]
    p = [0.8, 0.7, 0.1]

    cross__entropy = cross_entropy(Y,p)
    print(f"Cross Entropy is: {cross__entropy}")


if __name__ == "__main__":
    main()
