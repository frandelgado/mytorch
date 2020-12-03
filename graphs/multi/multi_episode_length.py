import matplotlib.pyplot as plt
import pickle

from graphs.utils import movingaverage, movingaverage2

results_n = 1
ep_clip = 225

for i in range(results_n):
    results = pickle.load(open(f"../../pickles/q_agent_results/results{i}.p", "rb"))
    episode_length = results["episode_length"][:ep_clip]
    y_av, y_std = movingaverage2(episode_length, 30)
    plt.errorbar([i for i in range(0, len(y_av))], y_av, yerr=y_std, label=f"Agente {i}")


plt.xlabel("Episodios")
plt.ylabel("Duración del episodio")
plt.legend()
plt.show()
