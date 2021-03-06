import matplotlib.pyplot as plt
import pickle

from graphs.utils import movingaverage

results = pickle.load(open("../pickles/ant_no_joints_cost/results.p", "rb"))
episode_length = results["episode_length"][0]

plt.plot([i for i in range(0, len(episode_length))], episode_length)

x_av = movingaverage(episode_length, 100)
plt.plot([i for i in range(0, len(x_av))], x_av)


plt.xlabel("Episodios")
plt.ylabel("Duración del episodio")

plt.show()
