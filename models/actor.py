import torch
import torch.nn as nn
import random

# Class creating the Actor

class Actor():

    def __init__(self, num_act, num_state, num_hidden):

        self.model = nn.Sequential(nn.Linear(num_act + num_state, num_hidden),
                                   nn.ReLU(),
                                   nn.Linear(num_hidden, 1),
                                   nn.Sigmoid())

    def predict(self, state):
        # Full exploration
        action = random.uniform(-2, 2)
        # needs to be list or array?

        return [action]
