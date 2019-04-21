import torch
from torch.autograd import Variable
from models.critic import Critic
from models.actor import Actor
from models.replay_buffer import ReplayBuffer
from random_process import OrnsteinUhlenbeckProcess


class DDPGAgent():

    def __init__(self, env, hp):

        self.env = env
        self.hp = hp
        self.critic = Critic(env.observation_space.shape[0] ,
                             env.action_space.shape[0],hp)

        self.target_critic = Critic(env.observation_space.shape[0],
                             env.action_space.shape[0],hp)

        self.actor = Actor(env.observation_space.shape[0],
                           env.action_space.shape[0],
                           env.action_space.high[0],hp)


        self.target_actor = Actor(env.observation_space.shape[0],
                           env.action_space.shape[0],
                           env.action_space.high[0],hp)


        self.dataset = ReplayBuffer(self.hp['batch_size'],
                                    self.hp['max_buffer_size'])

        self.noise = OrnsteinUhlenbeckProcess(env.action_space.shape[0])
        self.noise.reset_states()


    def take_action(self,state, greedy = False):

        state = Variable(torch.from_numpy(state)).float()
        action = self.actor.predict(state)
        if greedy:
            return action.detach().numpy()
        sample = self.noise.sample()
        return action.detach().numpy() \
            + (self.noise.sample() * self.env.action_space.high[0])

    def buffer_update(self, sample):

        self.dataset.add_sample(sample)

    def _critic_update(self, batch):

        s = batch[0]
        a = batch[1]
        r = batch[2]
        s_next = batch[3]
        done = batch[4]
        target_actions = self.target_actor.predict(s_next)
        Q_val = self.target_critic.predict(s_next,target_actions)
        y_target = r + done * (self.hp['gamma'] * Q_val)
        #y_target2 = r + self.hp['gamma'] * Q_val
        #print(y_target!=y_target2,done)
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
                            self.target_critic,
                            self.critic)
        self._target_update(self.hp['tau'],
                            self.target_actor,
                            self.actor)

    def _target_update(self, tau, target_network, network):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau*param.data + target_param.data*(1.0 - tau))
