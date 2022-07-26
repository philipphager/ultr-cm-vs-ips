#!/bin/bash

python hyperparameter.py -m \
    data=yahoo \
    model=lgbm \
    model.n_estimators=100,250,500,1000 \
    model.n_leaves=100,250,500,1000 \
    model.learning_rate=0.01,0.05,0.1,0.5 \
    model.random_state=0,1,2,3,4
