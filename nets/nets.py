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
                "sigmoid": layers.Sigmoid(architecture_layer["input_dim"], architecture_layer["output_dim"]),
                "relu": layers.ReLu(architecture_layer["input_dim"], architecture_layer["output_dim"]),
                "linear": layers.Linear(architecture_layer["input_dim"], architecture_layer["output_dim"]),
                "softmax": layers.Softmax(architecture_layer["input_dim"], architecture_layer["output_dim"])
            }.get(architecture_layer["activation"])
            self.layers.append(layer)

    def full_forward_propagation(self, X):
        memory = {}
        A_curr = X

        for idx, _ in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            A_prev = A_curr

            layer = self.layers[idx]

            W_curr = layer.store["W"]
            b_curr = layer.store["b"]
            A_curr, Z_curr = layer.forward(A_prev, W_curr, b_curr)

            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr

        return A_curr, memory

    def full_backward_propagation(self, dLoss, cache, action=None):
        # Dictionary to accumulate the gradients
        grads_values = {}
        dA_prev = dLoss

        for layer_idx_prev, _ in reversed(list(enumerate(self.nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1

            layer = self.layers[layer_idx_prev]
            # Derivative of the activations with respect to the loss function for current layer
            dA_curr = dA_prev

            # Activation output values for the previous layer
            A_prev = cache["A" + str(layer_idx_prev)]
            # Z values for the current layer A_curr = activ(Z_curr) = activ((A_prev * W_curr) + b_curr)
            Z_curr = cache["Z" + str(layer_idx_curr)]
            # Weights of the current layer
            W_curr = layer.store["W"]
            # biases of the current layer
            b_curr = layer.store["b"]
            # Calculate dL/dA, dL/dW, dL/db
            dA_prev, dW_curr, db_curr = layer.backward(
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
        for layer_idx, _ in enumerate(self.nn_architecture):
            layer = self.layers[layer_idx]
            layer_idx = layer_idx + 1

            dW = grads_values["dW" + str(layer_idx)] + (0.7 * layer.store["prevdW"])
            db = grads_values["db" + str(layer_idx)] + (0.7 * layer.store["prevdb"])

            layer.store["W"] += learning_rate * dW
            layer.store["b"] += learning_rate * db

            layer.store["prevdW"] = dW
            layer.store["prevdb"] = db

    def _update_rmsprop(self, grads_values, learning_rate):

        for layer_idx, _ in enumerate(self.nn_architecture):
            layer = self.layers[layer_idx]
            layer_idx = layer_idx + 1
            beta = 0.9

            dW = grads_values["dW" + str(layer_idx)]
            db = grads_values["db" + str(layer_idx)]

            VnW = beta * layer.store["prevVnW"] + (1 - beta) * np.square(dW)
            Vnb = beta * layer.store["prevVnb"] + (1 - beta) * np.square(db)

            layer.store["prevVnW"] = VnW
            layer.store["prevVnb"] = Vnb

            rmsprop_lrW = learning_rate / np.sqrt(VnW + 1e-8)
            rmsprop_lrb = learning_rate / np.sqrt(Vnb + 1e-8)

            layer.store["W"] += rmsprop_lrW * dW
            layer.store["b"] += rmsprop_lrb * db

    def _update_adam(self, grads_values, learning_rate):
        self._step += 1
        for layer_idx, _ in enumerate(self.nn_architecture):
            layer = self.layers[layer_idx]
            layer_idx = layer_idx + 1
            beta_2 = 0.9
            beta_1 = 0.9

            dW = grads_values["dW" + str(layer_idx)]
            db = grads_values["db" + str(layer_idx)]

            MnW = beta_1 * layer.store["prevdW"] + (1 - beta_2) * dW
            Mnb = beta_1 * layer.store["prevdb"] + (1 - beta_2) * db

            layer.store["prevdW"] = MnW
            layer.store["prevdb"] = Mnb

            VnW = beta_2 * layer.store["prevVnW"] + (1 - beta_2) * np.square(dW)
            Vnb = beta_2 * layer.store["prevVnb"] + (1 - beta_2) * np.square(db)

            layer.store["prevVnW"] = VnW
            layer.store["prevVnb"] = Vnb

            MnW_hat = MnW / (1 - np.power(beta_1, self._step))
            Mnb_hat = Mnb / (1 - np.power(beta_1, self._step))

            VnW_hat = VnW / (1 - np.power(beta_2, self._step))
            Vnb_hat = Vnb / (1 - np.power(beta_2, self._step))

            rmsprop_lrW = learning_rate / np.sqrt(VnW_hat + 1e-8)
            rmsprop_lrb = learning_rate / np.sqrt(Vnb_hat + 1e-8)

            layer.store["W"] += rmsprop_lrW * MnW_hat
            layer.store["b"] += rmsprop_lrb * Mnb_hat

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
