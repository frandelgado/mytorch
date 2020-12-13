import matplotlib.pyplot as plt
import pickle

from graphs.utils import movingaverage

results = pickle.load(open("../pickles/ant_no_joints_cost/results.p", "rb"))
reward_sums = results["reward_sums"]

plt.plot([i for i in range(0, len(reward_sums))], reward_sums)

x_av = movingaverage(reward_sums, 50)
plt.plot([i for i in range(0, len(x_av))], x_av)


plt.xlabel("Episodios")
plt.ylabel("suma de recompensas sin descontar")

plt.show()
