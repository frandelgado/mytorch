import numpy as np

from nets.activations import relu, sigmoid, relu_backward, sigmoid_backward


def init_layers(nn_architecture, seed=99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1

    return params_values


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_curr), Z_curr


def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    return A_curr, memory


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    m = A_prev.shape[1]
    # choose the appropriate activation function backward computation
    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    # Dictionary to accumulate the gradients
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)

    # Derivative of the loss function: dL/dA TODO consider abstracting this behaviour for different cost functions
    dA_prev = -(Y - Y_hat)

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]

        # Derivative of the activations with respect to the loss function for current layer
        dA_curr = dA_prev

        # Activation output values for the previous layer
        A_prev = memory["A" + str(layer_idx_prev)]
        # Z values for the current layer A_curr = activ(Z_curr) = activ((A_prev * W_curr) + b_curr)
        Z_curr = memory["Z" + str(layer_idx_curr)]
        # Weights of the current layer
        W_curr = params_values["W" + str(layer_idx_curr)]
        # biases of the current layer
        b_curr = params_values["b" + str(layer_idx_curr)]
        # Calculate dL/dA, dL/dW, dL/db
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        #Store the gradients for weights and biases (will be used for updates)
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values


def get_cost_value(Y_hat, Y):

    m = Y_hat.shape[1]
    cost = 0.5 * np.sum(np.subtract(Y_hat, Y)**2)
    return np.squeeze(cost)


def train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []

    for i in range(epochs):
        Y_hat, cache = full_forward_propagation(X, params_values, nn_architecture)
        # Do forward pass of stable softmax
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        # Do backwards pass of stable softmax
        grads_values = full_backward_propagation(Y_hat, Y, cache, params_values, nn_architecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

    return params_values, cost_history
