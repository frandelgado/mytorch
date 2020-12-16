import pickle

import matplotlib.pyplot as plt
import numpy as np

data = pickle.load(open("../../etl/store/etl_cartpole/ppo_adam2.p", "rb"))
aggregated = []
for datum in data:
    aggregated.append(datum["entropy"])

aggregated = np.vstack(aggregated)

means = np.mean(aggregated, axis=0)
stds = np.std(aggregated[:2500], axis=0)

plt.errorbar([i for i in range(0, len(means))], means, yerr=stds, ecolor="b", color="r")

plt.xlabel("Episodios")
plt.ylabel("Entropía")
plt.show()
