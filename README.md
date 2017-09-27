# Self-supervised Deep Reinforcement Learning with Generalized Computation Graphs for Robot Navigation

[Arxiv link](TODO)

<b>Abstract</b>: Enabling robots to autonomously navigate complex environments is essential for real-world deployment. Prior methods approach this problem by having the robot maintain an internal map of the world, and then use a localization and planning method to navigate through the internal map. However, these approaches often include a variety of assumptions, are computationally intensive, and do not learn from failures. In contrast, learning-based methods improve as the robot acts in the environment, but are difficult to deploy in the real-world due to their high sample complexity. To address the need to learn complex policies with few samples, we propose a generalized computation graph that subsumes value-based model-free methods and model-based methods, with specific instantiations interpolating between model-free and model-based. We then instantiate this graph to form a navigation model that learns from raw images and is sample efficient. Our simulated car experiments explore the design decisions of our navigation model, and show our approach outperforms single-step and N-step double Q-learning. We also evaluate our approach on a real-world RC car and show it can learn to navigate through a complex indoor environment with a few hours of fully autonomous, self-supervised training. 

Click below to view video

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/vgiW0HlQWVE/0.jpg)](https://www.youtube.com/watch?v=vgiW0HlQWVE)

---
# Code

This repository contains the code to run the simulation experiments. The main code is in [sandbox/gkahn/gcg](https://github.com/gkahn13/gcg/tree/gcg_release/sandbox/gkahn/gcg), while the rllab code was used for infrastructure purposes (e.g., running experiments on EC2).

---
### Installation

Clone the repository and add it to your PYTHONPATH

Install [Anaconda using the Python 2.7 installer](https://www.anaconda.com/download/).

We will always assume the current directory is [sandbox/gkahn/gcg](https://github.com/gkahn13/gcg/tree/gcg_release/sandbox/gkahn/gcg). Create a new Anaconda environment and activate it:
```bash
$ CONDA_SSL_VERIFY=false conda env create -f environment.yml
$ source activate gcg
```

Install Panda3D
```bash
$ pip install --pre --extra-index-url https://archive.panda3d.org/ panda3d
```

Increase the simulation speed by running
```bash
$ nvidia-settings
```
And disabling "Sync to VBLank" under "OpenGL Settings"

---
### Simulation environment

To drive in the simulation environment, run
```bash
$ python envs/rccar/square_cluttered_env.py
```

The commands are
- [ w ] forward
- [ x ] backward
- [ a ] left
- [ d ] right
- [ s ] stop
- [ r ] reset

---
### Yaml experiment configuration files

The [yamls](https://github.com/gkahn13/gcg/tree/gcg_release/sandbox/gkahn/gcg/yamls) folder contains experiment configuration files for [Double Q-learning](https://github.com/gkahn13/gcg/tree/gcg_release/sandbox/gkahn/gcg/yamls/dql.yaml) , [5-step Double Q-learning](https://github.com/gkahn13/gcg/tree/gcg_release/sandbox/gkahn/gcg/yamls/nstep_dql.yaml) , and [our approach](https://github.com/gkahn13/gcg/tree/gcg_release/sandbox/gkahn/gcg/yamls/ours.yaml).

These yaml files can be adapted to form alternative instantiations of the generalized computation graph. Please see the example yaml files for detailed descriptions.

---
### Running the code

To run our approach, execute
```bash
$ python run_exp.py --exps ours
```

The results will be stored in the gcg/data folder.

You can run other yaml files by replacing "ours" with the desired yaml file name (e.g., "dql" or "nstep_dql")

---
### References

```
TODO bibtex
```

