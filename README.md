# Benchmark-ChainerRL-library-in-Gym-Environments
Benchmark ChainerRL library in [OpenAI Gym](https://gym.openai.com/) Environments

## Objectives
- **Benchmarking RL algorithms:** Deterministic Policy Gradient [DDPG](https://arxiv.org/abs/1509.02971), Trust Region Policy Optimization [TRPO](http://proceedings.mlr.press/v37/schulman15.pdf) and Proximal Policy Optimization [PPO](https://arxiv.org/abs/1707.06347) algorithms.

## OpenAI Gym Enviroment
- [OpenAI Gym](https://gym.openai.com/) Open source interface to reinforcement learning tasks. The gym library provides an easy-to-use suite of reinforcement learning tasks.

- Open AI Gym has several environments, We Use classical control environments [Pendulum](https://github.com/openai/gym/wiki/Pendulum-v0) and [Bipedal Walker2D](https://github.com/openai/gym/wiki/BipedalWalker-v2) environmens.

![OpenAI_Gym](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Results/OpenAI.png)
## Codes:

- [TRPO Pendelum](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Pendulum/TRPO_Pendulum.ipynb

- [PPO Pendelum](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Pendulum/PPO_Pendulum.ipynb)

- [DDPG Pendelum](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Pendulum/DDPG_Pendulum.ipynb)

- [TRPO BipedalWalker](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/BipedalWalder2d/DDPG_BiPedalWalker.ipynb)

- [PPO BipedalWalker](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/BipedalWalder2d/PPO_walker2d.ipynb)

- [DDPG BipedalWalker](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/BipedalWalder2d/DDPG_BiPedalWalker.ipynb)

### Observations

#### Pendelum

- States: cosine and sine of angle between center and pendelum.

#### Bipedal Walker2D

- 14 Observations: hull angle, hull angular velocity, hip joint angle, hip joint speed, knee joint angle, knee joint speed, etc

### Actions

#### Pendelum

- Joint effort

#### Bipedal Walker2D

- 4 Actions: Hip_1 (Torque / Velocity), Hip_2 (Torque / Velocity), Knee_1 (Torque / Velocity) and Knee_2 (Torque / Velocity)

### Reward


#### Pendelum

![reward_fun](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Results/reward_fun.png)

#### Bipedal Walker2D

- 300+ points up to the far end. If the robot falls, it gets -100 

## Algorithms and Hyperparameters

- **[DDPG](https://arxiv.org/abs/1509.02971)** is a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces.DDPG is based on the deterministic policy gradient (DPG) algorithm. it combines the actor-critic approach with insights from the recent success of Deep Q Network (DQN).

- **[PPO](https://arxiv.org/abs/1707.06347)** is a policy optimization method that use multiple epochs of stochastic gradient ascent to perform each policy update.

- **[TRPO](http://proceedings.mlr.press/v37/schulman15.pdf)** is a model free, on-policy optimization method that effective for optimizing large nonlinear policies such as neural networks.

## Results

- **Pendelum**

|  |  **TRPO** | **PPO** | **DDPG** | 
| :---:         |     :---:      |   :---: |   :---: | 
|**Mean Reward** | -1216| -1252 | **-594** | 
|**Maximum Reward** | -986| -489  | -**371** |

![Pendelum_result](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Results/pendulum_mean.png)

- **Bipedal Walker2D**

|  |  **TRPO** | **PPO** | **DDPG** | 
| :---:         |     :---:      |   :---: |   :---: | 
|**Mean Reward** | 120| **163** | -96 | 
|**Maximum Reward** | 183| **262**  | -25 |


![Bipedal_results](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Results/walker_mean.png)


## Demo
- **Random Actions**

![Random](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Demo/random.gif)

- **[TRPO](http://proceedings.mlr.press/v37/schulman15.pdf)**

![TRPO](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Demo/trpo.gif)

- **[PPO](https://arxiv.org/abs/1707.06347)**

![PPO](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Demo/ppo.gif)

- **[DDPG](https://arxiv.org/abs/1509.02971)**

![DDPG](https://github.com/montaserFath/Benchmark-ChainerRL-library-in-Gym-Environments/blob/master/Demo/ddpg.gif)
## Discussion

- **[DDPG](https://arxiv.org/abs/1509.02971) algorithm achieves the best reward in Pendelum** because it designed for high dimensions continuous space environments and it uses the replay buffer.

- **[PPO](https://arxiv.org/abs/1707.06347) and [TRPO](http://proceedings.mlr.press/v37/schulman15.pdf) algorithms achieve the best reward in Bipedal Walker2D**.

- **[PPO](https://arxiv.org/abs/1707.06347) Reachs the best reward faster** than uses [TRPO](http://proceedings.mlr.press/v37/schulman15.pdf) because it use gradient algorithm approximation instance of the conjugate. gradient algorithm.

### Installing
Install OpenAI Gym Envirnment 
```
pip install gym
```
Install ChainerRL libary
```
pip install chainerrl
```
