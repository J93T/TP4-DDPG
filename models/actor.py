import torch
import torch.nn as nn
import torch.nn.functional as F


# Class creating the Actor
# https://pytorch.org/docs/stable/nn.html
# Models should inherit from nn.module

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, scale, hp):

        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scale = scale

        self.l1 = nn.Linear(state_dim,
                            hp['num_hidden_actor_l1'])

        self.l2 = nn.Linear(hp['num_hidden_actor_l1'],
                            hp['num_hidden_actor_l2'])

        self.l3 = nn.Linear(hp['num_hidden_actor_l2'],action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(),hp['lr_actor'])

    def predict(self, state):

        out = F.relu(self.l1(state))
        out = F.relu(self.l2(out))
        action = torch.tanh(self.l3(out))
        return action * self.scale

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
