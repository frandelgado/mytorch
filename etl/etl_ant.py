import pickle

import numpy as np

from configed_logging import log
from tests.ant_test import ant_test


def ant_etl(runs=30, agent_type="ant"):
    x = []
    means = []
    stds = []
    for save in range(0, 30000, 500):
        x.append(save)
        distances = []
        log.info(f"starting tests for save {save}")
        for test in range(runs):
            distance = ant_test(enable_rendering=False, episode_length=1024, agent_save=save, agent_type=agent_type)
            log.info(f"finished test {test} for agent {agent_type}")
            distances.append(distance)
        means.append(np.mean(distances))
        stds.append(np.std(distances))

    with open(f"store/etl_ant/{agent_type}_distance_covered.p", "wb") as file:
        pickle.dump(
            {
                "x": x,
                "means": means,
                "stds": stds,
            },
            file,
        )


def multi_etl(agent_types):
    for agent_type in agent_types:
        ant_etl(agent_type=agent_type)


if __name__ == '__main__':
    multi_etl(
        agent_types=[
            "ant",
            # "ant_no_electricity_cost",
            # "ant_no_joints_cost2",
            # "ant_no_electricity_no_joints",
        ]
    )
