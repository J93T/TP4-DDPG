import numpy as np
import random


class ReplayBuffer(object):
    """
    Replay Buffer for storing samples of shape <s,a,r,s'> where
    s = current state
    a = action used
    r = reward
    s_next = next state
    t = terminal boolean
    """
    def __init__(self, batch_size = 32, max_size = 10000):

        self.elements = []
        self.length = 0
        self.batch_size = batch_size
        self.max_size = max_size
        self.batch_idx = [1,2,3,4,5]

    def add_sample(self, sample):

        if self.length <= self.max_size:
            # TODO add sample
            self.elements.append(sample)
            pass
        else:
            #TODO delete oldest sample and add sample
            pass
        print(random.choices(self.elements, k=self.batch_size))

    def get_batch(self):

        # TODO shuffle batch_idx and return element[batch_index]
        pass
