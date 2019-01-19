# Benchmark-ChainerRL-library-in-Gym-Environments
Benchmark ChainerRL library in [OpenAI Gym](https://gym.openai.com/) Environments

## Objectives
- **Benchmarking RL algorithms:** Deterministic Policy Gradient [DDPG](https://arxiv.org/abs/1509.02971), Trust Region Policy Optimization [TRPO](http://proceedings.mlr.press/v37/schulman15.pdf) and Proximal Policy Optimization [PPO](https://arxiv.org/abs/1707.06347) algorithms.

## OpenAI Gym Enviroment
- [OpenAI Gym](https://gym.openai.com/) Open source interface to reinforcement learning tasks. The gym library provides an easy-to-use suite of reinforcement learning tasks.

- Open AI Gym has several environments, We Use classical control environments [Pendulum](https://github.com/openai/gym/wiki/Pendulum-v0) and [Bipedal Walker2D](https://github.com/openai/gym/wiki/BipedalWalker-v2) environmens.

![OpenAI_Gym](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Results/OpenAI.png)

### Observations
the [observations](http://osim-rl.stanford.edu/docs/nips2018/observation/) can be divided into five components:

- **Body parts:** the agent observes its position, velocity, acceleration, rotation, rotational velocity, and rotational acceleration.

- **Joints:** the agent observes its position, velocity and acceleration.

- **Muscles:** the agent observes its activation, fiber force, fiber length and fiber velocity.

- **Forces:** describes the forces acting on body parts.

- **Center of mass:** the agent observes the position, velocity, and acceleration.

### Actions

- Muscles activation, lenght and velocity

- Joints angels.

- Tendons.

### Reward

**<img src="https://latex.codecogs.com/gif.latex?R_{t}=9-(3-V_{t})^2" />**


Where the <img src="https://latex.codecogs.com/gif.latex?V_{t}"/> is the horizontal velocity vector of the pelvi which is function of all state variables.

The termination condition for the episode is filling 300 steps or the height of the pelvis falling below 0.6 meters
## Algorithms and Hyperparameters

- **[DDPG](https://arxiv.org/abs/1509.02971)** is a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces.DDPG is based on the deterministic policy gradient (DPG) algorithm. it combines the actor-critic approach with insights from the recent success of Deep Q Network (DQN).

- **[PPO](https://arxiv.org/abs/1707.06347)** is a policy optimization method that use multiple epochs of stochastic gradient ascent to perform each policy update.

- **[TRPO](http://proceedings.mlr.press/v37/schulman15.pdf)** is a model free, on-policy optimization method that effective for optimizing large nonlinear policies such as neural networks.

## Results

- **Pendelum**

![Pendelum_result](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Results/pendulum_mean.png)

- **Bipedal Walker2D**

![Bipedal_results](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Results/walker_mean.png)

## Demo
- **Random Actions**

![Random](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Demo/Random_walker2d.gif)

- **[TRPO](http://proceedings.mlr.press/v37/schulman15.pdf)**

![TRPO](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Demo/TRPO_walker2d.gif)

- **[PPO](https://arxiv.org/abs/1707.06347)**

![PPO](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Demo/PPO_walker2d.gif)

- **[DDPG](https://arxiv.org/abs/1509.02971)**

![DDPG](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Demo/DDPG_walker2d.gif)
## Discussion

- OpenSim [ProstheticsEnv](http://osim-rl.stanford.edu) is a very **complex environment**, it contains more than 158 continuous state variables and 19 continuous action variables.

- RL algorithms take a **long time** to build a complex policy which has the ability to compute all state variables and select action variables which will maximize the reward.

- **[DDPG](https://arxiv.org/abs/1509.02971) algorithm achieves good** reward because it designed for high dimensions continuous space environments and it uses the replay buffer.

- **[PPO](https://arxiv.org/abs/1707.06347) the least training time** comparing to [DDPG](https://arxiv.org/abs/1509.02971) and [TRPO](http://proceedings.mlr.press/v37/schulman15.pdf) because [PPO](https://arxiv.org/abs/1707.06347) uses gradient algorithm approximation instance of the conjugate gradient algorithm.

- **[TRPO](http://proceedings.mlr.press/v37/schulman15.pdf) algorithm achieved the maximum Reward** because it takes time to reach the “trusted” region so it slower than [DDPG](https://arxiv.org/abs/1509.02971) and [PPO](https://arxiv.org/abs/1707.06347) .

# Installation
.. code:: shell

.. code:: shell
