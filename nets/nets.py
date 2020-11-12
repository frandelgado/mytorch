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

    def forward(self, X):
        A_curr = X
        
        for layer in self.layers:
            
            A_prev = A_curr
            A_curr, Z_curr = layer.forward(A_prev)
            layer.store["Z"] = Z_curr
            layer.store["A"] = A_prev
        
        return A_curr
    
    def backward(self, dLoss, action=None):
        dA_prev = dLoss
        
        for layer in reversed(self.layers):
            dA_curr = dA_prev
            # Activation output values for the previous layer
            A_prev = layer.store["A"]
            # Z values for the current layer A_curr = activ(Z_curr) = activ((A_prev * W_curr) + b_curr)
            Z_curr = layer.store["Z"]

            # Weights of the current layer
            W_curr = layer.store["W"]
            # biases of the current layer
            b_curr = layer.store["b"]
            # Calculate dL/dA, dL/dW, dL/db
            dA_prev, dW_curr, db_curr = layer.backward(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, action=action
            )

            # Store the gradients for weights and biases (will be used for updates)
            layer.store["dW"] += dW_curr
            layer.store["db"] += db_curr

    def update(self, learning_rate):
        if self.optimizer == "momentum":
            self._update_momentum(learning_rate)
        elif self.optimizer == "rmsprop":
            self._update_rmsprop(learning_rate)
        elif self.optimizer == "adam":
            self._update_adam(learning_rate)

    def _update_momentum(self, learning_rate):

        for layer in self.layers:
            dW = layer.store["dW"] + (0.7 * layer.store["prevdW"])
            db = layer.store["db"] + (0.7 * layer.store["prevdb"])

            layer.store["W"] += learning_rate * dW
            layer.store["b"] += learning_rate * db

            layer.store["prevdW"] = dW
            layer.store["prevdb"] = db

    def _update_rmsprop(self, learning_rate):

        for layer in self.layers:
            beta = 0.9

            dW = layer.store["dW"]
            db = layer.store["db"]

            VnW = beta * layer.store["prevVnW"] + (1 - beta) * np.square(dW)
            Vnb = beta * layer.store["prevVnb"] + (1 - beta) * np.square(db)

            layer.store["prevVnW"] = VnW
            layer.store["prevVnb"] = Vnb

            rmsprop_lrW = learning_rate / np.sqrt(VnW + 1e-8)
            rmsprop_lrb = learning_rate / np.sqrt(Vnb + 1e-8)

            layer.store["W"] += rmsprop_lrW * dW
            layer.store["b"] += rmsprop_lrb * db

    def _update_adam(self, learning_rate):
        self._step += 1
        for layer in self.layers:
            beta_2 = 0.9
            beta_1 = 0.9

            dW = layer.store["dW"]
            db = layer.store["db"]

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

    def mean_grads(self, batch_size):
        for layer in self.layers:
            layer.mean_grads(batch_size)
