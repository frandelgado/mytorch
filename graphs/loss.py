import matplotlib.pyplot as plt
import pickle

results = pickle.load(open("../pickles/results.p", "rb"))
loss = results["loss"]
plt.plot([i for i in range(0, len(loss))], loss)
plt.show()
