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
    def __init__(self, batch_size = 32, max_size = 10):

        self.elements = []
        self.length = 0
        self.batch_size = batch_size
        self.max_size = max_size
        self.batch_idx = []

    def add_sample(self, sample):

        if self.length < self.max_size:
            self.elements.append(sample)
            self.length += 1
        else:
            self.elements.pop(0)
            self.elements.append(sample)

    def get_batch(self):
        self.batch_idx = self._random_sample(self.batch_size, 0, self.length)
        batch = [self.elements[i] for i in self.batch_idx]
        return batch


    def _random_sample(self,count, start, stop, step=1):
        def gen_random():
            while True:
                yield random.randrange(start, stop, step)

        def gen_n_unique(source, n):
            seen = set()
            seenadd = seen.add
            for i in (i for i in source() if i not in seen and not seenadd(i)):
                yield i
                if len(seen) == n:
                    break

        return [i for i in gen_n_unique(gen_random,
                        min(count, int(abs(stop - start) / abs(step))))]
