import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Class creating the Actor
# https://pytorch.org/docs/stable/nn.html
# Models should inherit from nn.module
EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, scale, hp):

        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scale = scale

        self.l1 = nn.Linear(state_dim,
                            hp['num_hidden_actor_l1'])
        self.l1_bn = nn.LayerNorm(hp['num_hidden_actor_l1'])
        self.l1.weight.data = fanin_init(self.l1.weight.data.size())


        self.l2 = nn.Linear(hp['num_hidden_actor_l1'],
                            hp['num_hidden_actor_l2'])
        self.l2_bn = nn.LayerNorm(hp['num_hidden_actor_l2'])
        self.l2.weight.data = fanin_init(self.l2.weight.data.size())

        self.l3 = nn.Linear(hp['num_hidden_actor_l2'],action_dim)
        self.l3.weight.data.uniform_(-EPS,EPS)
        self.optimizer = torch.optim.Adam(self.parameters(),hp['lr_actor'])

    def predict(self, state):

        out = F.relu(self.l1_bn(self.l1(state)))
        out = F.relu(self.l2_bn(self.l2(out)))
        action = torch.tanh(self.l3(out))
        return action * self.scale

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
