#!/bin/bash

python main.py -m \
    hydra/launcher=submitit_slurm \
    data=yahoo \
    experiment.name="dataset_size" \
    "+experiment=ips_pointwise_unbiased,cm_unbiased,cm_estimated" \
    optimizer=sgd,adam,adagrad \
    learning_rate=0.1,0.05,0.01,0.005,0.001 \
    simulation.n_sessions=100_000_000 \
    simulation.aggregate_clicks=True \
    random_state=0
