import torch
import torch.nn as nn
import random
from torch.autograd import Variable
import numpy as np

# Class creating the Actor
# https://pytorch.org/docs/stable/nn.html
# Models should inherit from nn.module

class Actor(nn.Module):

    def __init__(self, input_dim, output_dim, num_hidden):

        super(Actor, self).__init__()

        self.model = nn.Sequential(nn.Linear(input_dim, num_hidden),
                                   nn.ReLU(),
                                   nn.Linear(num_hidden, output_dim),
                                   nn.Sigmoid())

        self.target_model = nn.Sequential(nn.Linear(input_dim, num_hidden),
                                   nn.ReLU(),
                                   nn.Linear(num_hidden, output_dim),
                                   nn.Sigmoid())

        self.optimizer = torch.optim.Adam(self.model.parameters(),0.001)

    def predict(self, state, target=False):

        state = Variable(torch.from_numpy(state)).float()

        if target:
            return self.target_model(state).detach().numpy()
        else:
            return self.model(state).detach().numpy()

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
