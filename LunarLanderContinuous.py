import gym
from models.critic import Critic
from models.actor import Actor
from models.replay_buffer import ReplayBuffer
from ddpg import DDPGAgent
from rendering import rendering
import numpy as np


env = gym.make('LunarLanderContinuous-v2')


# Reproducability
#env.seed(1)
#print(env.action_space)

#------------------------------#
#------Hyperparameters---------#
#------------------------------#

hyperparameter = {
    'num_hidden_critic_l1': 256,
    'num_hidden_critic_l2': 128,
    'num_hidden_actor_l1': 256,
    'num_hidden_actor_l2': 128,
    'lr_actor': 0.0001,
    'lr_critic': 0.001,
    'gamma': 0.995,
    'batch_size': 128,
    'max_buffer_size': 1e6,
    'tau': 0.001,

}
num_episodes = 300
num_steps = 1000

#------------------------------#
#------Hyperparameters---------#
#------------------------------#

agent = DDPGAgent(env, hyperparameter)

rets =[]
render = False
for e in range(num_episodes):
    ret = 0
    s = env.reset()
    for step in range(num_steps):

        a = agent.take_action(s, greedy=False)
        #a = env.action_space.sample()
        s_next, r, done, _ = env.step(a)

        # Press Enter in the console to activate/deactivate Rendering
        render = rendering(env, render, r)
        ret += r
        agent.buffer_update([s, a, r/10, s_next, 1 - done])
        if done:
            break
        agent.update()
        s = s_next
    rets.append(ret)
    print("episode: {}/{}, reward: {}".format(e, num_episodes, ret))

np.save('results/data/pend_rets3.npy', rets)

env.close()