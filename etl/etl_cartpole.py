from configed_logging import log
import pickle

import numpy as np
import matplotlib.pyplot as plt

from envs.cartpole import cartpole


def etl_cartpole(runs=30, episodes_per_run=3000):
    log.info(f"Started etl_cartpole")
    aggregated_results = []

    for i in range(runs):
        results = cartpole(to_file=False, episodes=episodes_per_run)
        aggregated_results.append(np.array(results["episode_length"]))
        log.info(f"Finished run {i}")

    aggregated_results = np.vstack(aggregated_results)
    means = aggregated_results.mean(axis=0)
    stds = aggregated_results.std(axis=0)

    with open("store/etl_cartpole/aggregated.p", "wb") as file:
        pickle.dump(
            {"means": means, "stds": stds},
            file,
        )
    log.info(f"finished etl_cartpole")


if __name__ == '__main__':
    etl_cartpole()
