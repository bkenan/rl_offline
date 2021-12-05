# AIPI 530 Final project 


Hello! This is a repository for building  a pipeline for offline RL. 
The starter code repo has been provided by [d3rlpy](https://github.com/takuseno/d3rlpy) 


## Brief blog about Reinforcement Learning to address the following questions:

1. What is reinforcement learning?

2. What are the pros and cons of reinforcement learning?

3. When should we consider applying reinforcement learning (and when should not)?

4. What's the difference between supervised learning and reinforcement learning?

5. What is offline reinforcement learning?

6. What are the pros and cons of offline reinforcement learning?

7. When should we consider applying offline reinforcement learning (and when should not)?

8. Have an example of offline reinforcement learning in the real-world?

The link: [Reinforcement Learning basics](https://medium.com/@kanan.bk/reinforcement-learning-basics-f90b1a2fd3c8) 


## Getting Started 

This project is customized to training CQL on a custom dataset in d3rlpy, and training OPE (FQE) to 
evaluate the trained policy. `cql.py` at the root of the project is the main script. The Default dataset is `hopper-bullet-mixed-v0`

### The steps for installation:

1. Install d3rlpy <br /> <br />
[![PyPI version](https://badge.fury.io/py/d3rlpy.svg)](https://badge.fury.io/py/d3rlpy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/d3rlpy)
```
$ pip install d3rlpy
```
2. Clone this repo: `git clone https://github.com/bkenan/rl_offline.git`
3. Install **pybullet** from source: `pip install git+https://github.com/takuseno/d4rl-pybullet`
4. Install requirements: `pip install Cython numpy` & `pip install -e`
5. Execute **`cql.py`** 
6. **The Logs:**
   * Average reward vs training steps: `d3rlpy_logs/CQL_hopper-bullet-mixed-v0_1/environment.csv`
   * True Q values vs training steps: `d3rlpy_logs/CQL_hopper-bullet-mixed-v0_1/true_q_value.csv`
   * Estimated Q values vs training steps: `d3rlpy_logs/CQL_hopper-bullet-mixed-v0_1/init_value.csv`
   
### My Colab notebook: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cof3Cyk7FTam3q4IkRhgHO2GpKYEXfzn?usp=sharing) 


### d3rlpy: An offline deep reinforcement learning library
d3rlpy is an offline deep reinforcement learning library for practitioners and researchers.

Documentation: https://d3rlpy.readthedocs.io

## d3rlpy examples

```py
import d3rlpy

dataset, env = d3rlpy.datasets.get_dataset("hopper-medium-v0")

# prepare algorithm
sac = d3rlpy.algos.SAC()

# train offline
sac.fit(dataset, n_steps=1000000)

# train online
sac.fit_online(env, n_steps=1000000)

# ready to control
actions = sac.predict(x)
```

### MuJoCo
```py
import d3rlpy

# prepare dataset
dataset, env = d3rlpy.datasets.get_d4rl('hopper-medium-v0')

# prepare algorithm
cql = d3rlpy.algos.CQL(use_gpu=True)

# train
cql.fit(dataset,
        eval_episodes=dataset,
        n_epochs=100,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'td_error': d3rlpy.metrics.td_error_scorer
        })
```
See more datasets at [d4rl](https://github.com/rail-berkeley/d4rl).

### Atari 2600
```py
import d3rlpy
from sklearn.model_selection import train_test_split

# prepare dataset
dataset, env = d3rlpy.datasets.get_atari('breakout-expert-v0')

# split dataset
train_episodes, test_episodes = train_test_split(dataset, test_size=0.1)

# prepare algorithm
cql = d3rlpy.algos.DiscreteCQL(n_frames=4, q_func_factory='qr', scaler='pixel', use_gpu=True)

# start training
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=100,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'td_error': d3rlpy.metrics.td_error_scorer
        })
```
See more Atari datasets at [d4rl-atari](https://github.com/takuseno/d4rl-atari).

### PyBullet

```py
import d3rlpy

# prepare dataset
dataset, env = d3rlpy.datasets.get_pybullet('hopper-bullet-mixed-v0')

# prepare algorithm
cql = d3rlpy.algos.CQL(use_gpu=True)

# start training
cql.fit(dataset,
        eval_episodes=dataset,
        n_epochs=100,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'td_error': d3rlpy.metrics.td_error_scorer
        })
```
See more PyBullet datasets at [d4rl-pybullet](https://github.com/takuseno/d4rl-pybullet).

## Tutorials 
Try a cartpole example on Google Colaboratory:
 * official offline RL tutorial: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takuseno/d3rlpy/blob/master/tutorials/cartpole.ipynb)

## Citation
The paper is available [here](https://arxiv.org/abs/2111.03788).
```
@InProceedings{seno2021d3rlpy,
  author = {Takuma Seno, Michita Imai},
  title = {d3rlpy: An Offline Deep Reinforcement Library},
  booktitle = {NeurIPS 2021 Offline Reinforcement Learning Workshop},
  month = {December},
  year = {2021}
}
```
