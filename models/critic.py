import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, hp):

        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.l1s = nn.Linear(state_dim,
                            hp['num_hidden_critic_l1'])
        self.l1s_bn = nn.BatchNorm1d(hp['num_hidden_critic_l1'])
        self.l1s.weight.data = fanin_init(self.l1s.weight.data.size())


        self.l1a = nn.Linear(action_dim,
                              hp['num_hidden_critic_l1'])
        self.l1a.weight.data = fanin_init(self.l1a.weight.data.size())


        #self.l4 = nn.Linear(2 * hp['num_hidden_critic_l1'],
        #                    hp['num_hidden_critic_l2'])

        self.l4 = nn.Linear(hp['num_hidden_critic_l1'] + action_dim,
                            hp['num_hidden_critic_l2'])
        self.l4.weight.data = fanin_init(self.l4.weight.data.size())


        self.l5 = nn.Linear(hp['num_hidden_critic_l2'],1)
        self.l5.weight.data.uniform_(-EPS,EPS)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),hp['lr_critic'],weight_decay=0.01)

    def predict(self, state, action):

        # out_s = F.relu(self.l1s_bn(self.l1s(state)))
        # out_a = F.relu(self.l1a(action))
        # out = torch.cat((out_s,out_a),dim=1)
        # out = F.relu(self.l4(out))
        # return self.l5(out)

        out_s = F.relu(self.l1s_bn(self.l1s(state)))
        # #out_a = F.relu(self.l1a(action))
        out = torch.cat((out_s,action),dim=1)
        out = F.relu(self.l4(out))
        return self.l5(out)

    def train(self, y_pred, y_target):

        loss  = self.criterion(y_pred, y_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
