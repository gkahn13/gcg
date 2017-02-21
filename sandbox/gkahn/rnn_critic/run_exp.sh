#!/usr/bin/env bash

exp=$1
python examples/run_rnn_critic.py "examples/yamls/${exp}.yaml"
python scripts/analyze_experiment.py $exp