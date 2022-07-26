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
    baseline_model = instantiate(config.baseline)

    train = data.load("train")
    val = data.load("val")
    test = data.load("test")

    baseline, train = random_split(train, frac=0.01)
    baseline_model.fit(baseline)


if __name__ == "__main__":
    main()
