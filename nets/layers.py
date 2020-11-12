import numpy as np

from abc import ABC

from nets.activations import relu, sigmoid, softmax, linear, relu_backward, sigmoid_backward, softmax_backward, \
    linear_backward


class Layer(ABC):

    def forward(self, A_prev, W_curr, b_curr):
        raise NotImplementedError

    def backward(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, action=None):
        raise NotImplementedError

    @staticmethod
    def calculate_grads(A_prev, W_curr, dZ_curr):
        m = A_prev.shape[1]
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)
        return dA_prev, dW_curr, db_curr


class ReLu(Layer):

    def forward(self, A_prev, W_curr, b_curr):
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        return relu(Z_curr), Z_curr

    def backward(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, action=None):
        dZ_curr = relu_backward(dA_curr, Z_curr, action=action)
        return self.calculate_grads(A_prev, W_curr, dZ_curr)


class Sigmoid(Layer):
    def forward(self, A_prev, W_curr, b_curr):
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        return sigmoid(Z_curr), Z_curr

    def backward(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, action=None):
        dZ_curr = sigmoid_backward(dA_curr, Z_curr, action=action)
        return self.calculate_grads(A_prev, W_curr, dZ_curr)


class Softmax(Layer):
    def forward(self, A_prev, W_curr, b_curr):
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        return softmax(Z_curr), Z_curr

    def backward(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, action=None):
        dZ_curr = softmax_backward(dA_curr, Z_curr, action=action)
        return self.calculate_grads(A_prev, W_curr, dZ_curr)


class Linear(Layer):
    def forward(self, A_prev, W_curr, b_curr):
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        return linear(Z_curr), Z_curr

    def backward(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, action=None):
        dZ_curr = linear_backward(dA_curr, Z_curr, action=action)
        return self.calculate_grads(A_prev, W_curr, dZ_curr)


class Net:
    def __init__(self):
        self.layers = []

    def forward(self):
        pass
