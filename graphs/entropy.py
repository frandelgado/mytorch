import matplotlib.pyplot as plt
import pickle

from graphs.utils import movingaverage

results = pickle.load(open("../pickles/results.p", "rb"))
entropy = results["entropy"]
plt.plot([i for i in range(0, len(entropy))], entropy)

x_av = movingaverage(entropy, 200)
plt.plot([i for i in range(0, len(x_av))], x_av)


plt.xlabel("Episodios")
plt.ylabel("Entripía promedio")
plt.show()
