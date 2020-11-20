import matplotlib.pyplot as plt
import pickle
import argparse

from graphs.utils import movingaverage

dims            = 8
window_size     = 50
pickle_file     = "../pickles/results.p"

results = pickle.load(open(pickle_file, "rb"))
entropy = results["entropy"]

for dim in range(dims):
    dim_entropy = entropy[dim]
    x_av = movingaverage(dim_entropy, window_size)
    plt.plot([i for i in range(0, len(x_av))], x_av, label=f"{dim}")


plt.xlabel("Episodios")
plt.ylabel("Entripía promedio")
plt.legend()
plt.show()
