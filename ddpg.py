import numpy as np
from models.critic import Critic
from models.actor import Actor

class DDPGAgent():

    def __init__(self, dataset):

        self.critic = Critic(1,1,1)
        self.actor = Actor(1,1,1)
        self.dataset = dataset


    def take_action(self, state):
        # TODO Add noise according to OU-Process
        action = self.actor.predict(state, False)
        return [action]

    def buffer_update(self, sample):
        # TODO add sample to dataset
        pass

    def target_predict(self, action):
        pass

    def _critic_update(self):
        #TODO Minimize TD- error
        pass

    def _actor_update(self):
        #TODO Use policy Gradient
        pass

    def update(self):
        #TODO update critic, actor and target networks
        pass
