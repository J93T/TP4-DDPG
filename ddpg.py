import numpy as np
from models.critic import Critic
from models.actor import Actor
from models.replay_buffer import ReplayBuffer

class DDPGAgent():

    def __init__(self, env, hp):

        #maybe a cleaner way to do this?
        self.hp = hp
        # critic input dim = action_dim+observation_dim
        self.critic = Critic(env.action_space.shape[0]
                             + env.observation_space.shape[0],
                             env.observation_space.shape[0],
                             hp['num_hidden_critic'])

        self.actor = Actor(env.observation_space.shape[0],
                           env.action_space.shape[0],
                           hp['num_hidden_actor'])
        self.dataset = ReplayBuffer(self.hp['batch_size'],
                                    self.hp['max_buffer_size'])


    def take_action(self, state):
        # TODO Add noise according to OU-Process
        action = self.actor.predict(state, False)
        return [action]

    def buffer_update(self, sample):

        self.dataset.add_sample(sample)

    def target_predict(self, action):
        pass

    def _critic_update(self):
        #TODO Minimize TD- error
        pass

    def _actor_update(self):
        #TODO Use policy Gradient
        pass

    def update(self):
        self._critic_update()
        self._actor_update()
        self._target_update(self.hp['tau'],
                            self.critic.target_model,
                            self.actor.model)
        self._target_update(self.hp['tau'],
                            self.critic.target_model,
                            self.actor.model)

    def _target_update(self, tau, target_network, network):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(tau*param.data + target_param.data*(1.0 - tau))
