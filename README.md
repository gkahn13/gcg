# Self-supervised Deep Reinforcement Learning with Generalized Computation Graphs for Robot Navigation

Code implementation of our [paper](TODO). The main code is in [sandbox/gkahn/gcg](https://github.com/gkahn13/gcg/tree/gcg_release/sandbox/gkahn/gcg), while the rllab code was used for infrastructure purposes (e.g., running experiments on EC2).

---
### Installation

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

---
### Simulation environment

To drive in the simulation environment, run
```bash
$ python envs/rccar/square_cluttered_env.py
```

The commands are
* [w] forward
* [x] backward
* [a] left
* [d] right
* [s] stop
* [r] reset

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

You can run other yaml files by replacing "ours" with the desired yaml file name (e.g., "dql" or "nstep_dql")

---
### References

```
TODO bibtex
```

