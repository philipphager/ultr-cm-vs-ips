#!/bin/bash

python main.py -m \
    hydra/launcher=submitit_slurm \
    data=yahoo \
    experiment.name="dataset_size" \
    "+experiment=ips_pointwise_unbiased,ips_pointwise_biased,cm_unbiased,cm_biased,cm_estimated" \
    simulation.n_sessions=100,1_000,10_000,100_000,1_000_000,10_000_000,100_000_000 \
    simulation.aggregate_clicks=True \
    random_state=0,1,2,3,4,5,6,7,8,9
