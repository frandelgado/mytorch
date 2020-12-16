import matplotlib.pyplot as plt
import pickle
import argparse

import numpy as np

from graphs.utils import movingaverage

dims            = 8
window_size     = 100
pickle_file     = "../../pickles/ant/results.p"

results = pickle.load(open(pickle_file, "rb"))
entropy = results["entropy"]
x_av_av = []
for dim in range(dims):
    dim_entropy = entropy[dim][:30000]
    x_av = movingaverage(dim_entropy, window_size)
    x_av_av.append(x_av)

x_av_av = np.mean(x_av_av, axis=0)
plt.plot([i for i in range(0, len(x_av_av))], x_av_av)
# plt.plot([i for i in range(0, len(x_av_av), 500)], x_av_av[0:15000:500], 'o')

plt.xlabel("Episodios")
plt.ylabel("Entripía")
plt.show()
