import torch
import torch.nn as nn
import random

# Class creating the Actor
# https://pytorch.org/docs/stable/nn.html
# Models should inherit from nn.module

class Actor(nn.Module):

    def __init__(self, num_act, num_state, num_hidden):

        super(Actor, self).__init__()

        self.model = nn.Sequential(nn.Linear(num_act + num_state, num_hidden),
                                   nn.ReLU(),
                                   nn.Linear(num_hidden, 1),
                                   nn.Sigmoid())

        self.target_model = nn.Sequential(nn.Linear(num_act + num_state, num_hidden),
                                   nn.ReLU(),
                                   nn.Linear(num_hidden, 1),
                                   nn.Sigmoid())

    def predict(self, state, target=False):
        action = random.uniform(-2, 2)
        return action
