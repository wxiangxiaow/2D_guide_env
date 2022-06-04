# 2D_guide_env
An easy 2D guide environment for RL.

## Background
The 2D coordinate of agent is (x, y), and θ=tan(y/x) shows the included angle between travel direction and positive x-axis of agent. 

Target : Guide the agent to a designated point.

Action : { velocity | palstance }

State : { agent position | target postion | relative position }

## Algorithm
Continuous-PPO with tricks shown in [this work](https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/5.PPO-continuous).

#### Tricks List
Trick 1—Advantage Normalization.

Trick 2—State Normalization.

Trick 4—— Reward Reward Scaling.

Trick 5—Policy Entropy.

Trick 6—Learning Rate Decay.

Trick 7—Gradient clip.

Trick 8—Orthogonal Initialization.

Trick 9—Adam Optimizer Epsilon Parameter.

Trick10—Tanh Activation Function.

#### Tips
An beta distribution must be used instead of Gaussian one for avoiding agent **sample to much at the edge of the action space**.

An action distribution sample on Beta distribution be like:

![](https://github.com/wxiangxiaow/2D_guide_env/blob/main/imgs/Action_1.png)

## Effect of Reward Function
Two Rewards are used in this session: R = T + α * F

1) F = -0.01 * (preious distance - now distance)
2) T = 100(Terminal); -2(Out of Space); -1(Max Episode)

**Terminal Reward effects the agents**

10 times of evaluation have done for each 5e3 steps, shows the differences:

Terminal Reward = 50:

![](https://github.com/wxiangxiaow/2D_guide_env/blob/main/imgs/evaluate_reward_50.png)

Terminal Reward = 80:

![](https://github.com/wxiangxiaow/2D_guide_env/blob/main/imgs/Evaluate_Rewards_80.png)

Terminal Reward = 90:

![](https://github.com/wxiangxiaow/2D_guide_env/blob/main/imgs/Evaluate_Reward_90.png)

Termiinal Reward = 100:

![](https://github.com/wxiangxiaow/2D_guide_env/blob/main/imgs/evaluate_reward.png)


## Test
A set of 10 times evaluation had been done. The result is shown in /test_img

For example:

![](https://github.com/wxiangxiaow/2D_guide_env/blob/main/test_img/Evaluate_No_4.png)

#### Tips
Beside the args of PPO, the args of normalization(mean and std) must be used.

## Requirments
numpy

matplotlib

math

gym
