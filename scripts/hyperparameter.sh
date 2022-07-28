#!/bin/bash

python main.py -m \
    hydra/launcher=submitit_slurm \
    data=yahoo \
    experiment.name="hyperparameters" \
    "+experiment=ips_pointwise_unbiased,cm_unbiased,cm_estimated" \
    model.optimizer=sgd,sgd-momentum,adam,adagrad \
    model.learning_rate=0.1,0.05,0.01,0.005,0.001,0.0005,0.0001 \
    simulation.n_sessions=100_000_000 \
    simulation.aggregate_clicks=True \
    random_state=0,1,2,3,4
