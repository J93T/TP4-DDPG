import torch
from torch.autograd import Variable
from models.critic import Critic
from models.actor import Actor
from models.replay_buffer import ReplayBuffer
from random_process import OrnsteinUhlenbeckProcess


class DDPGAgent():

    def __init__(self, env, hp):

        #maybe a cleaner way to do this?
        self.env = env
        self.hp = hp
        # critic input dim = action_dim+observation_dim
        self.critic = Critic(env.observation_space.shape[0] ,
                             env.action_space.shape[0],hp)

        self.critict = Critic(env.observation_space.shape[0],
                             env.action_space.shape[0],hp)

        self.actor = Actor(env.observation_space.shape[0],
                           env.action_space.shape[0],
                           env.action_space.high[0],hp)


        self.actort = Actor(env.observation_space.shape[0],
                           env.action_space.shape[0],
                           env.action_space.high[0],hp)


        self.dataset = ReplayBuffer(self.hp['batch_size'],
                                    self.hp['max_buffer_size'])

        self.noise = OrnsteinUhlenbeckProcess(1)
        self.noise.reset_states()


    def take_action(self,state):

        state = Variable(torch.from_numpy(state)).float()
        action = self.actor.predict(state)
        return action.detach().numpy() \
            + (self.noise.sample() * self.env.action_space.high[0])

    def buffer_update(self, sample):

        self.dataset.add_sample(sample)

    def _critic_update(self, batch):

        s = batch[0]
        a = batch[1]
        r = batch[2]
        s_next = batch[3]
        target_actions = self.actort.predict(s_next)
        Q_val = self.critict.predict(s_next,target_actions)
        y_target = r + self.hp['gamma'] * Q_val
        y_pred = self.critic.predict(s,a)
        self.critic.train(y_pred, y_target)

    def _actor_update(self, batch):

        s = batch[0]
        pred_a = self.actor.predict(s)
        loss = torch.mean(-self.critic.predict(s, pred_a))
        self.actor.train(loss)


    def update(self):

        if self.dataset.length < self.hp['batch_size']:
            return
        batch = self.dataset.get_batch()

        self._critic_update(batch)
        self._actor_update(batch)
        self._target_update(self.hp['tau'],
                            self.critict,
                            self.critic)
        self._target_update(self.hp['tau'],
                            self.actort,
                            self.actort)

    def _target_update(self, tau, target_network, network):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau*param.data + target_param.data*(1.0 - tau))
