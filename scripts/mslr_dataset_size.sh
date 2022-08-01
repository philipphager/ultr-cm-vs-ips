#!/bin/bash

python main.py -m \
    hydra/launcher=submitit_slurm \
    data=mslr30k \
    experiment.name="dataset_size" \
    "+experiment=cm_estimated" \
    simulation.n_sessions=100,1_000,10_000,100_000,1_000_000,10_000_000,100_000_000 \
    simulation.aggregate_clicks=True \
    random_state=0,1,2,3,4,5,6,7,8,9

#python main.py -m \
#    hydra/launcher=submitit_slurm \
#    data=mslr30k \
#    experiment.name="dataset_size" \
#    "+experiment=full_info" \
#    simulation.n_sessions=100_000_000 \
#    simulation.aggregate_clicks=True \
#    random_state=0,1,2,3,4,5,6,7,8,9
