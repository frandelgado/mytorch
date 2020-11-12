from nets.optim import Momentum


class Net:
    def __init__(self, layers, lr, optimizer=Momentum()):
        self.layers = layers
        self.lr = lr
        self.optimizer = optimizer
        self.cost_history = []

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

    def update(self):
        self.optimizer.update(self.layers, lr=self.lr)

    def mean_grads(self, batch_size):
        for layer in self.layers:
            layer.mean_grads(batch_size)
