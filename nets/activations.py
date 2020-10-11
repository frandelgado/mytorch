import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0/(1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def sigmoid_backward(dA, z, action):
    """
    Calculates dL/dZ

    :param dA:
        ndarray representing the vector of partial derivatives dL/dA
    :param z:
        the values of z that were passed to the sigmoid on the forwards pass
    :return:
        ndarray representing the vector of partial derivatives dL/dZ
    """
    sig = sigmoid(z)
    dZ = dA * sig * (1 - sig)
    return dZ


def relu_backward(dA, z, action):
    """
    Calculates dL/dZ

    :param dA:
        ndarray representing the vector of partial derivatives dL/dA
    :param z:
        the values of z that were passed to the relu on the forwards pass
    :return:
        ndarray representing the vector of partial derivatives dL/dZ
    """
    dZ = np.array(dA, copy=True)
    dZ[z <= 0] = 0
    return dZ


def softmax(x):
    """
    Computes the numerically stable softmax, knowing thad softmax(x) = softmax(x + C)

    :param x:
        the input vector of the softmax function
    :return:
        the output vector of the softmax function
    """
    shifted_x = x - np.max(x)
    exps = np.exp(shifted_x)
    exps_sum = np.sum(exps)
    return exps/exps_sum


def softmax_backward(dA, z, action):
    """
    Calculates dL/dZ
    :param dA:
        ndarray representing the vector of partial derivatives dL/dA
    :param z:
        the values of z that were passed to the softmax on the forwards pass
    :return:
        ndarray representing the vector of partial derivatives dL/dZ
    """
    z = z.reshape(1, -1).squeeze()
    S = softmax(z)
    action = action.squeeze()
    dZ = np.zeros(len(z))
    for j in range(len(z)):
        if j == action:
            dZ[j] = dA * S[action] * (1 - S[action])
        else:
            dZ[j] = - dA * S[action] * S[j]
    return dZ.reshape((-1, 1))


def linear(x):
    """
    Applies a linear fuction y = x

    :param x:
        input ndarray
    :return:
        output ndarray
    """
    return x


def linear_backward(dA, z):
    """
    Derivative for the linear funciont y = x

    :param dA:
        the derivatives of the output with respect to the output of the linear funciton, i.e. dL/dA
    :param z:
        the values of z that were passed to the relu on the forwards pass
    :return:
        ndarray representing the vector of partial derivatives dL/dZ
    """
    return dA

