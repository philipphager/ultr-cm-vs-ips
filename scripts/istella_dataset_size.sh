#!/bin/bash

python main.py -m \
    hydra/launcher=submitit_slurm \
    data=istella \
    experiment.name="dataset_size" \
    "+experiment=cm_unbiased" \
    simulation.n_sessions=100,1_000,10_000,100_000,1_000_000,10_000_000,100_000_000 \
    simulation.aggregate_clicks=True \
    random_state=0,1,2,3,4,5,6,7,8,9
