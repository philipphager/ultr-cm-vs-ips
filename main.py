import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.data.preprocessing.util import random_split


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    print("Working directory : {}".format(os.getcwd()))

    data = instantiate(config.data)
    train = data.load("train")
    val = data.load("val")
    test = data.load("test")

    baseline_model = instantiate(config.baseline)
    baseline, train = random_split(train, frac=0.01)
    baseline_model.fit(baseline)

    val_split = len(val) / len(train)
    train_clicks = config.simulation.n_clicks
    val_clicks = int(train_clicks * val_split)
    print(f"Simulating {train_clicks} train clicks")
    print(f"Simulating {val_clicks} val clicks ({val_split:.4f}%) of train")

    simulator = instantiate(config.simulation.simulator, baseline_model=baseline_model)
    train = simulator(
        train, n_clicks=train_clicks, aggregate=config.simulation.aggregate_clicks
    )
    val = simulator(
        val, n_clicks=val_clicks, aggregate=config.simulation.aggregate_clicks
    )


if __name__ == "__main__":
    main()
