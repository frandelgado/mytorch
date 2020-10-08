import pickle

import numpy as np


class ReluSaturationObserver:

    def __init__(self, window_size=100):
        self.iteration = 0
        self.window_size = window_size
        self.net_outputs = np.zeros(shape=(window_size,), dtype=object)
        self.saturation_coefficients = []

    def observe(self, memory):
        self.net_outputs[self.iteration % self.window_size] = memory["Z1"]
        self.iteration += 1
        self._calculate_saturation_coefficient()
        self._report()

    def _calculate_saturation_coefficient(self):
        total = 0
        zeros = 0
        for relu_outputs in self.net_outputs:
            if isinstance(relu_outputs, int):
                continue
            for output in relu_outputs:
                if output == 0:
                    zeros += 1
                total += 1

        self.saturation_coefficients.append(zeros/total)

    def _report(self):
        if self.iteration % 1000 != 0:
            return
        with open("../pickles/saturation_observer.p", "wb") as file:
            pickle.dump(self.saturation_coefficients, file)
