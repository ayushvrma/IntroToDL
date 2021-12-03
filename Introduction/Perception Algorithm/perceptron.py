import numpy as np
import pandas as pd 
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i], W, b)
        if(y[i]- y_hat==1):
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate

        if(y[i] - y_hat == -1):
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    # print(X.shape)
    x_min, x_max = np.min(X.T[0]), np.max(X.T[0])
    y_min, y_max = np.min(X.T[1]), np.max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines


def main():
    data = np.genfromtxt('Introduction/Perception Algorithm/data.csv', delimiter=',')
    # print(data[:, -1])
    boundary_lines = trainPerceptronAlgorithm(data[:,:-1], data[:, -1])
    print(boundary_lines)


if __name__ == "__main__":
    main()
