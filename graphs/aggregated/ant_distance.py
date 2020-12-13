import pickle
import matplotlib.pyplot as plt

from configed_logging import log


def ant_distance(agent_types, agent_label):
    if len(agent_label) != len(agent_types):
        log.error("Labels and agents must be same size")
        return

    for i, agent_type in enumerate(agent_types):
        data = pickle.load(open(f"../../etl/store/etl_ant/{agent_type}_distance_covered.p", "rb"))
        x = data["x"]
        means = data["means"]
        stds = data["stds"]

        plt.errorbar(x, means, yerr=stds, label=agent_label[i], marker="o")

    plt.xlabel("Episodios")
    plt.ylabel("Distancia recorrida (m)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ant_distance(
        agent_types=[
            "ant",
            "ant_no_electricity_cost",
            "ant_no_joints_cost",
            "ant_no_electricity_no_joints",
        ],
        agent_label=[
            "control",
            "sin e",
            "sin j",
            "sin j ni e",
        ]
    )
