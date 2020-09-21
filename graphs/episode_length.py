import matplotlib.pyplot as plt
import pickle

results = pickle.load(open("../pickles/results.p", "rb"))
episode_length = results["episode_length"]
plt.plot([i for i in range(0, len(episode_length))], episode_length)
plt.show()
