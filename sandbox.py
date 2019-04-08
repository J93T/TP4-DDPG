import gym
from models.critic import Critic
from models.actor import Actor
from models.replay_buffer import ReplayBuffer
from ddpg import DDPGAgent
import numpy as np

env = gym.make('Pendulum-v0')
#env = gym.make('CarRacing-v0')
#env = gym.make('HalfCheetah-v2')
#env = gym.make('LunarLanderContinuous-v2')
# Reproducability
env.seed(1)

#------------------------------#
#------Hyperparameters---------#
#------------------------------#
hyperparameter = {
    'num_hidden_critic': 128,
    'num_hidden_actor': 128,
    'gamma': 0.99,
    'batch_size': 5,
    'max_buffer_size': 1e5,
    'tau': 0.001,
}
num_episodes = 100
num_steps = 500
#------------------------------#
#------Hyperparameters---------#
#------------------------------#

agent = DDPGAgent(env, hyperparameter)



for e in range(num_episodes):
    print("starting episode ", e)
    ret = 0
    s = env.reset()
    for step in range(num_steps):

        env.render()

        # Get action from Actor
        a = agent.take_action(s)

        # Execute action, receive reward
        s_next, r, done, _ = env.step(a)
        ret += r

        # Save transition
        agent.buffer_update([s, a, r, s_next, done])

        if done:
            break

        # Train critic and actor?
        agent.update()

        s = s_next
    print(ret)

env.close()
