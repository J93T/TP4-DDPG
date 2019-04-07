import torch
import torch.nn as nn

# Class creating the critic

class Critic(nn.Module):

    def __init__(self, num_act, num_state, num_hidden):

        super(Critic, self).__init__()

        self.model = nn.Sequential(nn.Linear(num_act + num_state, num_hidden),
                                   nn.ReLU(),
                                   nn.Linear(num_hidden, 1),
                                   nn.Sigmoid())
