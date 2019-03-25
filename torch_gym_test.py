import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make('CartPole-v0')


state = env.reset()


for i in range(100):

    env.render()
    action = env.action_space.sample()
    s_next, r, done, _ = env.step(action)
    if done:
        break

env.close()

torch.manual_seed(1)
lin = nn.Linear(5, 3)
data = torch.randn(2, 5)
print(lin(data))
