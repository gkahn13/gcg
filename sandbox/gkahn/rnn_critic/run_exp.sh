#!/usr/bin/env bash

exp=$1
python examples/run_rnn_critic.py "examples/yamls/${exp}.yaml"
python scripts/rnn_critic_analyze_experiment.py $exp