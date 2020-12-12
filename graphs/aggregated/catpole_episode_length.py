import pickle

import matplotlib.pyplot as plt


aggregated = pickle.load(open("../../etl/store/etl_cartpole/aggregated_momentum.p", "rb"))
means = aggregated["means"][:2500]
stds = aggregated["stds"][:2500]

plt.errorbar([i for i in range(0, len(means))], means, yerr=stds, ecolor="b", color="r")

plt.xlabel("Episodios")
plt.ylabel("Duración del episodio (pasos)")
plt.show()
