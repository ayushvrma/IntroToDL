# Softmax converts scores obtained by linear regression to probabilities between 0 and 1
# Used for multi-class classification i.e more than 2 classes
# eg: duck, beaver, walrus



import numpy as np 
import math


def softmax(X):
    l1 = []
    total = 0
    for i in range(len(X)):
        total += math.exp(X[i])
    for i in range(len(X)):
        l1.append(math.exp(X[i])/total)
    return l1

def main():
    X = [4,3,2,1,0,-1,-2,-3]
    score_to_prob = softmax(X)
    print(score_to_prob)


if __name__ == "__main__":
    main()
