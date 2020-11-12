import numpy as np

from nets import layers
from nets.activations import relu, sigmoid, relu_backward, sigmoid_backward, softmax, softmax_backward, linear, \
    linear_backward


class Net:
    def __init__(self, nn_architecture, optimizer="momentum"):
        self.optimizer = optimizer
        self._step = 0
        if self.optimizer == "momentum" or self.optimizer == "adam":
            self._save_prev_grads = True
        else:
            self._save_prev_grads = False

        if self.optimizer == "rmsprop" or self.optimizer == "adam":
            self._save_second_order = True
        else:
            self._save_second_order = False

        self.cost_history = []
        self.nn_architecture = nn_architecture
        self.layers = []
        for architecture_layer in nn_architecture:
            layer = {
                "sigmoid": layers.Sigmoid(),
                "relu": layers.ReLu(),
                "linear": layers.Linear(),
                "softmax": layers.Softmax()
            }.get(architecture_layer["activation"])
            self.layers.append(layer)

        self.params_values = self.init_layers()

    def init_layers(self, seed=99):
        np.random.seed(seed)
        params_values = {}

        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            if self._save_prev_grads:
                params_values["prevdW" + str(layer_idx)] = np.zeros(shape=(layer_output_size, layer_input_size))
                params_values["prevdb" + str(layer_idx)] = np.zeros(shape=(layer_output_size, 1))

            if self._save_second_order:
                params_values["prevVnW" + str(layer_idx)] = np.zeros(shape=(layer_output_size, layer_input_size))
                params_values["prevVnb" + str(layer_idx)] = np.zeros(shape=(layer_output_size, 1))

            params_values['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * (1 / np.sqrt(layer_input_size))
            params_values['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * (1 / np.sqrt(layer_input_size))

        return params_values

    def single_layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="relu"):
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation == "relu":
            activation_func = relu
        elif activation == "sigmoid":
            activation_func = sigmoid
        elif activation == "softmax":
            activation_func = softmax
        elif activation == "linear":
            activation_func = linear
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
            A_curr, Z_curr = self.layers[idx].forward(A_prev, W_curr, b_curr)

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
        elif activation == "linear":
            backward_activation_func = linear_backward
        else:
            raise Exception('Non-supported activation function')

        dZ_curr = backward_activation_func(dA_curr, Z_curr, action=action)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_backward_propagation(self, dLoss, cache, action=None):
        # Dictionary to accumulate the gradients
        grads_values = {}
        dA_prev = dLoss

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
            dA_prev, dW_curr, db_curr = self.layers[layer_idx_prev].backward(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, action=action
            )

            # Store the gradients for weights and biases (will be used for updates)
            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values

    def update(self, grads_values, learning_rate):
        if self.optimizer == "momentum":
            self._update_momentum(grads_values, learning_rate)
        elif self.optimizer == "rmsprop":
            self._update_rmsprop(grads_values, learning_rate)
        elif self.optimizer == "adam":
            self._update_adam(grads_values, learning_rate)

    def _update_momentum(self, grads_values, learning_rate):
        for layer_idx, layer in enumerate(self.nn_architecture):
            layer_idx = layer_idx + 1

            dW = grads_values["dW" + str(layer_idx)] + (0.7 * self.params_values["prevdW" + str(layer_idx)])
            db = grads_values["db" + str(layer_idx)] + (0.7 * self.params_values["prevdb" + str(layer_idx)])

            self.params_values["W" + str(layer_idx)] += learning_rate * dW
            self.params_values["b" + str(layer_idx)] += learning_rate * db

            self.params_values["prevdW" + str(layer_idx)] = dW
            self.params_values["prevdb" + str(layer_idx)] = db

    def _update_rmsprop(self, grads_values, learning_rate):

        for layer_idx, layer in enumerate(self.nn_architecture):
            layer_idx = layer_idx + 1
            beta = 0.9

            dW = grads_values["dW" + str(layer_idx)]
            db = grads_values["db" + str(layer_idx)]

            VnW = beta * self.params_values["prevVnW" + str(layer_idx)] + (1 - beta) * np.square(dW)
            Vnb = beta * self.params_values["prevVnb" + str(layer_idx)] + (1 - beta) * np.square(db)

            self.params_values["prevVnW" + str(layer_idx)] = VnW
            self.params_values["prevVnb" + str(layer_idx)] = Vnb

            rmsprop_lrW = learning_rate / np.sqrt(VnW + 1e-8)
            rmsprop_lrb = learning_rate / np.sqrt(Vnb + 1e-8)

            self.params_values["W" + str(layer_idx)] += rmsprop_lrW * dW
            self.params_values["b" + str(layer_idx)] += rmsprop_lrb * db

    def _update_adam(self, grads_values, learning_rate):
        self._step += 1
        for layer_idx, layer in enumerate(self.nn_architecture):
            layer_idx = layer_idx + 1
            beta_2 = 0.9
            beta_1 = 0.9

            dW = grads_values["dW" + str(layer_idx)]
            db = grads_values["db" + str(layer_idx)]

            MnW = beta_1 * self.params_values["prevdW" + str(layer_idx)] + (1 - beta_2) * dW
            Mnb = beta_1 * self.params_values["prevdb" + str(layer_idx)] + (1 - beta_2) * db

            self.params_values["prevdW" + str(layer_idx)] = MnW
            self.params_values["prevdb" + str(layer_idx)] = Mnb

            VnW = beta_2 * self.params_values["prevVnW" + str(layer_idx)] + (1 - beta_2) * np.square(dW)
            Vnb = beta_2 * self.params_values["prevVnb" + str(layer_idx)] + (1 - beta_2) * np.square(db)

            self.params_values["prevVnW" + str(layer_idx)] = VnW
            self.params_values["prevVnb" + str(layer_idx)] = Vnb

            MnW_hat = MnW / (1 - np.power(beta_1, self._step))
            Mnb_hat = Mnb / (1 - np.power(beta_1, self._step))

            VnW_hat = VnW / (1 - np.power(beta_2, self._step))
            Vnb_hat = Vnb / (1 - np.power(beta_2, self._step))

            rmsprop_lrW = learning_rate / np.sqrt(VnW_hat + 1e-8)
            rmsprop_lrb = learning_rate / np.sqrt(Vnb_hat + 1e-8)

            self.params_values["W" + str(layer_idx)] += rmsprop_lrW * MnW_hat
            self.params_values["b" + str(layer_idx)] += rmsprop_lrb * Mnb_hat

    def mean_grads(self, grads_values_batch, batch_size):
        grads_values_sum = {}
        for layer_idx, layer in enumerate(self.nn_architecture):
            layer_idx = layer_idx + 1
            for grad_values in grads_values_batch:
                if not "dW" + str(layer_idx) in grads_values_sum:
                    grads_values_sum["dW" + str(layer_idx)] = grad_values["dW" + str(layer_idx)]
                    grads_values_sum["db" + str(layer_idx)] = grad_values["db" + str(layer_idx)]
                else:
                    grads_values_sum["dW" + str(layer_idx)] += grad_values["dW" + str(layer_idx)]
                    grads_values_sum["db" + str(layer_idx)] += grad_values["db" + str(layer_idx)]

            grads_values_sum["dW" + str(layer_idx)] /= batch_size
            grads_values_sum["db" + str(layer_idx)] /= batch_size

        return grads_values_sum
