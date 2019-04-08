import numpy as np
import torch
from torch.autograd import Variable
from models.critic import Critic
from models.actor import Actor
from models.replay_buffer import ReplayBuffer
from random_process import OrnsteinUhlenbeckProcess

class DDPGAgent():

    def __init__(self, env, hp):

        #maybe a cleaner way to do this?
        self.hp = hp
        # critic input dim = action_dim+observation_dim
        self.critic = Critic(env.action_space.shape[0]
                             + env.observation_space.shape[0],
                             1,
                             hp['num_hidden_critic'])

        self.actor = Actor(env.observation_space.shape[0],
                           env.action_space.shape[0],
                           hp['num_hidden_actor'])
        self.dataset = ReplayBuffer(self.hp['batch_size'],
                                    self.hp['max_buffer_size'])

        self.noise = OrnsteinUhlenbeckProcess(1)
        self.noise.reset_states()


    def take_action(self, state):
        # TODO Add noise according to OU-Process
        action = self.actor.predict(state, False)
        action += self.noise.sample()
        return action

    def buffer_update(self, sample):

        self.dataset.add_sample(sample)

    def _critic_update(self, batch):


        s = np.asarray([item[0] for item in batch])
        a = np.asarray([item[1] for item in batch])
        r = np.expand_dims(np.asarray([item[2] for item in batch]),1)
        r = Variable(
            torch.from_numpy(
                np.asarray([item[2] for item in batch]))).float().unsqueeze(1)
        s_next = np.asarray([item[3] for item in batch])
        target_actions = self.actor.predict(s_next, True)
        Q_val = self.critic.predict(s_next,target_actions, True)
        y_target = r + self.hp['gamma'] * Q_val
        y_pred = self.critic.predict(s, a, False)
        self.critic.train(y_pred, y_target)

    def _actor_update(self, batch):

        s = np.asarray([item[0] for item in batch])
        pred_a1 = self.actor.predict(s, False)
        loss = -1*torch.sum(self.critic.predict(s, pred_a1))
        self.actor.train(loss)


    def update(self):

        if self.dataset.length < self.hp['batch_size']:
            return
        batch = self.dataset.get_batch()

        self._critic_update(batch)
        self._actor_update(batch)
        self._target_update(self.hp['tau'],
                            self.critic.target_model,
                            self.critic.model)
        self._target_update(self.hp['tau'],
                            self.actor.target_model,
                            self.actor.model)

    def _target_update(self, tau, target_network, network):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau*param.data + target_param.data*(1.0 - tau))
