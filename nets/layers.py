import numpy as np

from abc import ABC

from nets.activations import relu, sigmoid, softmax, linear, relu_backward, sigmoid_backward, softmax_backward, \
    linear_backward


class Layer(ABC):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.store = {
            "prevdW": np.zeros(shape=(output_dim, input_dim)),
            "prevdb": np.zeros(shape=(output_dim, 1)),
            "prevVnW": np.zeros(shape=(output_dim, input_dim)),
            "prevVnb": np.zeros(shape=(output_dim, 1)),
            'W': np.random.randn(output_dim, input_dim) * (1 / np.sqrt(input_dim)),
            'b': np.random.randn(output_dim, 1) * (1 / np.sqrt(input_dim))
        }

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


