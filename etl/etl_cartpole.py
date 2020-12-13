from configed_logging import log
import pickle

import numpy as np
import matplotlib.pyplot as plt

from envs.cartpole import cartpole


def etl_cartpole(runs=30, episodes_per_run=2500):
    log.info(f"Started etl_cartpole")
    aggregated_results = []

    for i in range(runs):
        results = cartpole(to_file=False, episodes=episodes_per_run)
        aggregated_results.append(results)
        log.info(f"Finished run {i}")

    with open("store/etl_cartpole/ppo_adam2.p", "wb") as file:
        pickle.dump(
            aggregated_results,
            file,
        )
    log.info(f"finished etl_cartpole")


if __name__ == '__main__':
    etl_cartpole()
