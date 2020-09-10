import numpy as np

from nets import nets

nn_architecture = [
    {"input_dim": 2, "output_dim": 3, "activation": "sigmoid"},
    {"input_dim": 3, "output_dim": 2, "activation": "sigmoid"},
]

training_example = np.array([[1],
                             [2]])
target_value = np.array([[2],
                         [4]])

nets.train(training_example, target_value, nn_architecture, 1, 0.01)


training_example = np.array([[1, 4],
                             [2, 5]])
target_value = np.array([[2, 8],
                         [4, 10]])
