import matplotlib.pyplot as plt
import pickle

from graphs.utils import movingaverage

results = pickle.load(open("../pickles/ant/results.p", "rb"))
returns = results["returns"][0]


x_av = movingaverage(returns, 100)
plt.plot([i for i in range(0, len(x_av))], x_av)


plt.xlabel("Episodios")
plt.ylabel("Retorno por episodio")

plt.show()
