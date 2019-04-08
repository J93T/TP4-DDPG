import torch
import torch.nn as nn

# Class creating the critic

class Critic(nn.Module):

    def __init__(self, input_dim, output_dim, num_hidden):

        super(Critic, self).__init__()

        self.model = nn.Sequential(nn.Linear(input_dim, num_hidden),
                                   nn.ReLU(),
                                   nn.Linear(num_hidden, output_dim),
                                   nn.Sigmoid())

        self.target_model = nn.Sequential(nn.Linear(input_dim, num_hidden),
                                   nn.ReLU(),
                                   nn.Linear(num_hidden, output_dim),
                                   nn.Sigmoid())



    def train(self, target):
        pass
