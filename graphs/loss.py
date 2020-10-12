import matplotlib.pyplot as plt
import pickle

from graphs.utils import movingaverage

results = pickle.load(open("../pickles/results.p", "rb"))
loss = results["loss"]

plt.plot([i for i in range(0, len(loss))], loss)

x_av = movingaverage(loss, 200)
plt.plot([i for i in range(0, len(x_av))], x_av)

plt.xlabel("Episodios")
plt.ylabel("Pérdida promedio")

plt.show()
