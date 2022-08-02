#!/bin/bash

python main.py -m \
    hydra/launcher=submitit_slurm \
    data=istella \
    experiment.name="position_bias_strength" \
    "+experiment=ips_pointwise_unbiased,cm_unbiased" \
    simulation.n_sessions=100_000_000 \
    simulation.aggregate_clicks=True \
    simulation.simulator.user_model.position_bias=0,0.5,1.0,1.5,2.0 \
    random_state=0
