import numpy as np

from nets.activations import relu, sigmoid, relu_backward, sigmoid_backward, softmax, softmax_backward


class Net:
    def __init__(self, nn_architecture):
        self.cost_history = []
        self.nn_architecture = nn_architecture
        self.params_values = self.init_layers()

    def init_layers(self, seed=99):
        np.random.seed(seed)
        number_of_layers = len(self.nn_architecture)
        params_values = {}

        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            params_values["prevW" + str(layer_idx)] = np.zeros(shape=(layer_output_size, layer_input_size))
            params_values["prevb" + str(layer_idx)] = np.zeros(shape=(layer_output_size, 1))

            params_values['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * (1 / np.sqrt(layer_output_size) - 0.5)
            params_values['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * (1 / np.sqrt(layer_output_size) - 0.5)

        return params_values

    def single_layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="relu"):
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation == "relu":
            activation_func = relu
        elif activation == "sigmoid":
            activation_func = sigmoid
        elif activation == "softmax":
            activation_func = softmax
        else:
            raise Exception('Non-supported activation function')

        return activation_func(Z_curr), Z_curr

    def full_forward_propagation(self, X):
        memory = {}
        A_curr = X

        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            A_prev = A_curr

            activation_function_curr = layer["activation"]
            W_curr = self.params_values["W" + str(layer_idx)]
            b_curr = self.params_values["b" + str(layer_idx)]
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activation_function_curr)

            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr

        return A_curr, memory

    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu",
                                          action=None):
        m = A_prev.shape[1]
        # choose the appropriate activation function backward computation
        if activation == "relu":
            backward_activation_func = relu_backward
        elif activation == "sigmoid":
            backward_activation_func = sigmoid_backward
        elif activation == "softmax":
            backward_activation_func = softmax_backward
        else:
            raise Exception('Non-supported activation function')

        dZ_curr = backward_activation_func(dA_curr, Z_curr, action=action)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_backward_propagation(self, loss, cache, action=None):
        # Dictionary to accumulate the gradients
        grads_values = {}
        m = loss.shape[1]

        dA_prev = loss

        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]

            # Derivative of the activations with respect to the loss function for current layer
            dA_curr = dA_prev

            # Activation output values for the previous layer
            A_prev = cache["A" + str(layer_idx_prev)]
            # Z values for the current layer A_curr = activ(Z_curr) = activ((A_prev * W_curr) + b_curr)
            Z_curr = cache["Z" + str(layer_idx_curr)]
            # Weights of the current layer
            W_curr = self.params_values["W" + str(layer_idx_curr)]
            # biases of the current layer
            b_curr = self.params_values["b" + str(layer_idx_curr)]
            # Calculate dL/dA, dL/dW, dL/db
            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr, action=action
            )

            # Store the gradients for weights and biases (will be used for updates)
            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values

    def update(self, grads_values, learning_rate):
        for layer_idx, layer in enumerate(self.nn_architecture):
            layer_idx = layer_idx + 1
            self.params_values["W" + str(layer_idx)] -= learning_rate * (
                    grads_values["dW" + str(layer_idx)]
            )
            self.params_values["b" + str(layer_idx)] -= learning_rate * (
                    grads_values["db" + str(layer_idx)]
            )

    def train(self, X, loss, dLoss, epochs, learning_rate, actions):
        _, cache = self.full_forward_propagation(X)
        grads_values = self.full_backward_propagation(dLoss, cache, actions)
        self.update(grads_values, learning_rate)

