import numpy as np


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'valid')


def movingaverage2(interval, window_size):
    length = len(interval) - window_size
    means = np.zeros(shape=(length,))
    stds = np.zeros(shape=(length,))
    for i in range(length):
        window = interval[i:i+window_size]
        means[i] = np.mean(window)
        stds[i] = np.std(window)

    return means, stds
