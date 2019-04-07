import gym
from models.critic import Critic
from models.actor import Actor
from models.replay_buffer import ReplayBuffer

env = gym.make('Pendulum-v0')
# Reproducability
env.seed(1)

critic = Critic(1,2,3)
actor = Actor(1,2,3)
buffer = ReplayBuffer()

#------------------------------#
#------Hyperparameters---------#
#------------------------------#

num_episodes = 3
num_steps = 200

#------------------------------#
#------Hyperparameters---------#
#------------------------------#

s = env.reset()

for e in range(num_episodes):
    for step in range(num_steps):

        env.render()

        # Get action from Actor
        a = actor.predict(s)

        # Execute action, receive transition
        s_next, r, done, _ = env.step(a)

        # Save transition
        buffer.add_sample([s, a, r, s_next, done])

        # Train critic and actor?

        s = s_next

env.close()
