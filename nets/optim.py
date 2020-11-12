from abc import ABC

import numpy as np


class Optimizer(ABC):
    def __init__(self):
        self._t = 0

    def update(self, layers, lr):
        self._t += 1


class Momentum(Optimizer):
    
    def update(self, layers, lr):
        super(Momentum, self).update(layers, lr)
        for layer in layers:
            dW = layer.store["dW"] + (0.7 * layer.store["prevdW"])
            db = layer.store["db"] + (0.7 * layer.store["prevdb"])

            layer.store["W"] += lr * dW
            layer.store["b"] += lr * db

            layer.store["prevdW"] = dW
            layer.store["prevdb"] = db


class RMSProp(Optimizer):
    
    def __init__(self, beta=0.9):
        super(RMSProp, self).__init__()
        self.beta = beta
    
    def update(self, layers, lr):
        super(RMSProp, self).update(layers, lr)
        for layer in layers:

            dW = layer.store["dW"]
            db = layer.store["db"]

            VnW = self.beta * layer.store["prevVnW"] + (1 - self.beta) * np.square(dW)
            Vnb = self.beta * layer.store["prevVnb"] + (1 - self.beta) * np.square(db)

            layer.store["prevVnW"] = VnW
            layer.store["prevVnb"] = Vnb

            rmsprop_lrW = lr / np.sqrt(VnW + 1e-8)
            rmsprop_lrb = lr / np.sqrt(Vnb + 1e-8)

            layer.store["W"] += rmsprop_lrW * dW
            layer.store["b"] += rmsprop_lrb * db


class Adam(Optimizer):

    def __init__(self, beta_1=0.9, beta_2=0.9):
        super(Adam, self).__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update(self, layers, lr):
        super(Adam, self).update(layers, lr)
        for layer in layers:

            dW = layer.store["dW"]
            db = layer.store["db"]

            MnW = self.beta_1 * layer.store["prevdW"] + (1 - self.beta_2) * dW
            Mnb = self.beta_1 * layer.store["prevdb"] + (1 - self.beta_2) * db

            layer.store["prevdW"] = MnW
            layer.store["prevdb"] = Mnb

            VnW = self.beta_2 * layer.store["prevVnW"] + (1 - self.beta_2) * np.square(dW)
            Vnb = self.beta_2 * layer.store["prevVnb"] + (1 - self.beta_2) * np.square(db)

            layer.store["prevVnW"] = VnW
            layer.store["prevVnb"] = Vnb

            MnW_hat = MnW / (1 - np.power(self.beta_1, self._t))
            Mnb_hat = Mnb / (1 - np.power(self.beta_1, self._t))

            VnW_hat = VnW / (1 - np.power(self.beta_2, self._t))
            Vnb_hat = Vnb / (1 - np.power(self.beta_2, self._t))

            rmsprop_lrW = lr / np.sqrt(VnW_hat + 1e-8)
            rmsprop_lrb = lr / np.sqrt(Vnb_hat + 1e-8)

            layer.store["W"] += rmsprop_lrW * MnW_hat
            layer.store["b"] += rmsprop_lrb * Mnb_hat
