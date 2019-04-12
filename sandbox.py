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
#env.seed(1)

#------------------------------#
#------Hyperparameters---------#
#------------------------------#

hyperparameter = {
    'num_hidden_critic_l1': 200,
    'num_hidden_critic_l2': 100,
    'num_hidden_actor_l1': 200,
    'num_hidden_actor_l2': 100,
    'lr_actor': 0.0001,
    'lr_critic': 0.001,
    'gamma': 0.99,
    'batch_size': 64,
    'max_buffer_size': 1e6,
    'tau': 0.001,

}
num_episodes = 100
num_steps = 1000

#------------------------------#
#------Hyperparameters---------#
#------------------------------#

agent = DDPGAgent(env, hyperparameter)


for e in range(num_episodes):
    ret = 0
    s = env.reset()
    for step in range(num_steps):
        if e > 60:
            env.render()
        a = agent.take_action(s)
        s_next, r, done, _ = env.step(a)
        ret += r
        agent.buffer_update([s, a, r, s_next, done])
        if done:
            break
        agent.update()
        s = s_next
    print("episode: {}/{}, reward: {}".format(e,num_episodes,ret))

env.close()
