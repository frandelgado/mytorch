import numpy as np


def sigmoid(z):
    return 1/1+np.exp(-z)


def relu(z):
    return np.maximum(0, z)


def sigmoid_backward(dA, z):
    sig = sigmoid(z)
    return dA * sig * (1 - sig)


def relu_backward(dA, z):
    dZ = np.array(dA, copy=True)
    dZ[z <= 0] = 0
    return dZ


def stable_softmax(x):
    shifted_x = x - np.max(x)
    exps = np.exp(shifted_x)
    exps_sum = np.sum(exps)
    return exps/exps_sum



