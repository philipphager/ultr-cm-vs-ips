#!/bin/bash

python main.py -m \
    hydra/launcher=submitit_slurm \
    data=synthetic \
    experiment.name="feature_collisions" \
    "+experiment=ips_pointwise_unbiased,cm_unbiased,cm_estimated" \
    data.perc_feature_collision=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
    simulation.n_sessions=100_000_000 \
    simulation.aggregate_clicks=True \
    random_state=0,1,2,3,4,5,6,7,8,9
