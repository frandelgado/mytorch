import matplotlib.pyplot as plt
import pickle
import argparse

from graphs.utils import movingaverage

parser = argparse.ArgumentParser(
    description="Plots the entropy of the agent's action heads"
)

parser.add_argument(
    "metric",
    metavar="m",
    choices=[
        "entropy",
        "returns"
    ]
)

parser.add_argument(
    "-d", "--dims",
    metavar="d",
    type=int,
    help="dimensions of the action head "
         "(i.e. first dimension of the results pickle)",
    default=1,
)

parser.add_argument(
    "-w", "--window_size",
    metavar="w",
    type=int,
    help="Rolling window size. Size of 1 means no rolling window",
    default=1,

)

parser.add_argument(
    "-f", "--file",
    metavar="f",
    help="The pickle file from which the entropy records will be read",
    default="../pickles/results.p",
)



args = parser.parse_args()
dims = args.dims
window_size = args.window_size
pickle_file = args.file
metric = args.metric

results = pickle.load(open(pickle_file, "rb"))
entropy = results["entropy"]

for dim in range(dims):
    dim_entropy = entropy[dim]
    x_av = movingaverage(dim_entropy, window_size)
    plt.plot([i for i in range(0, len(x_av))], x_av)

plt.xlabel("Episodios")
plt.ylabel("Entripía promedio")
plt.show()
