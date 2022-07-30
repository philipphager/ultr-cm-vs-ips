import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from src.data.preprocessing.util import random_split


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    print("Working directory : {}".format(os.getcwd()))

    seed_everything(config.random_state)

    data = instantiate(config.data)
    train = data.load("train")
    val = data.load("val")
    test = data.load("test")

    baseline_model = instantiate(config.baseline)
    baseline, train = random_split(train, n=20)
    baseline_model.fit(baseline)
    baseline_model.test(test)

    val_split = len(val) / len(train)
    train_sessions = config.simulation.n_sessions
    val_sessions = max(1, int(train_sessions * val_split))
    print(f"Simulating {train_sessions} train sessions")
    print(f"Simulating {val_sessions} val sessions ({val_split:.4f}%) of train")

    simulator = instantiate(config.simulation.simulator, baseline_model=baseline_model)
    train = simulator(
        train, n_sessions=train_sessions, aggregate=config.simulation.aggregate_clicks
    )
    val = simulator(
        val, n_sessions=val_sessions, aggregate=config.simulation.aggregate_clicks
    )

    model = instantiate(config.model)
    train_loader = instantiate(config.train_loader, dataset=train)
    val_loader = instantiate(config.val_test_loader, dataset=val)
    trainer = instantiate(config.trainer, auto_lr_find=True)

    lr_finder = trainer.tuner.lr_find(
        model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    print(f"{model.name}, {model.optimizer}, suggested LR:", lr_finder.suggestion())


if __name__ == "__main__":
    main()
