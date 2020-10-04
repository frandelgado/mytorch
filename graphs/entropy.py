import matplotlib.pyplot as plt
import pickle

results = pickle.load(open("../pickles/results.p", "rb"))
episode_length = results["entropy"]
plt.plot([i for i in range(0, len(episode_length))], episode_length)
plt.xlabel("Iteraciones")
plt.ylabel("Entripía")
plt.show()
