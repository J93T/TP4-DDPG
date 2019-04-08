import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


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

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),0.001)


    def predict(self, state, action, target=False):

        state = Variable(torch.from_numpy(state)).float()
        actions = Variable(torch.from_numpy(action)).float()

        sa = torch.cat((state, actions), 1)

        if target:
            return self.target_model(sa)#.detach().numpy()
        else:
            return self.model(sa)#.detach().numpy()

    def train(self, y_pred, y_target):

        loss = self.criterion(y_pred, y_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
