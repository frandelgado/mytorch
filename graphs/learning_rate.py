import matplotlib.pyplot as plt
import pickle

from graphs.utils import movingaverage

results = pickle.load(open("../pickles/results.p", "rb"))
learning_rate = results["learning_rate"]

plt.plot([i for i in range(0, len(learning_rate))], learning_rate)

x_av = movingaverage(learning_rate, 200)
plt.plot([i for i in range(0, len(x_av))], x_av)


plt.xlabel("Episodios")
plt.ylabel("Learning rate")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
print(learning_rate[-10:])
plt.show()
