import gym
import sys
from models.critic import Critic
from models.actor import Actor
from models.replay_buffer import ReplayBuffer
from ddpg import DDPGAgent
from util.rendering import rendering
import numpy as np

if sys.argv[1] == 'll':
    model_path = 'trained_models/Lunar_Lander_200episodes/'
    env = gym.make('LunarLanderContinuous-v2')
    hyperparameter = {
    'num_hidden_critic_l1': 300,
    'num_hidden_critic_l2': 200,
    'num_hidden_actor_l1': 300,
    'num_hidden_actor_l2': 200,
    'lr_actor': 0.0001,
    'lr_critic': 0.001,
    'gamma': 0.99,
    'batch_size': 128,
    'max_buffer_size': 1e6,
    'tau': 0.01,
    'noise_sigma': 0.2}
elif sys.argv[1] == 'mc':
    model_path = 'trained_models/Mountain_Car_150episodes/'
    env = gym.make('MountainCarContinuous-v0')
    hyperparameter = {
    'num_hidden_critic_l1': 128,
    'num_hidden_critic_l2': 64,
    'num_hidden_actor_l1': 128,
    'num_hidden_actor_l2': 64,
    'lr_actor': 0.0001,
    'lr_critic': 0.001,
    'gamma': 0.99,
    'batch_size': 128,
    'max_buffer_size': 1e6,
    'tau': 0.005,
    'noise_sigma': 0.2,
}
elif sys.argv[1] == 'p':
    model_path = 'trained_models/Pendulum_125episodes/'
    env = gym.make('Pendulum-v0')
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
    'tau': 0.01,
    'noise_sigma': 0.2}
else:
    print('Environment unknown!')
    exit()

num_episodes = 200
num_steps = 250

#------------------------------#
#------Hyperparameters---------#
#------------------------------#

agent = DDPGAgent(env, hyperparameter)
agent.load_models(model_path)

rets =[]
render = False
for e in range(num_episodes):
    ret = 0
    s = env.reset()
    for step in range(num_steps):
        a = agent.take_action(s, greedy=True)
        s_next, r, done, _ = env.step(a)
        # Press Enter in the console to activate/deactivate Rendering
        render = rendering(env, render, r)
        ret += r
        if done:
            break
        s = s_next
    rets.append(ret)
    print("episode: {}/{}, score {}, Running Avg {}".format(e,num_episodes,ret, np.mean(rets)))

env.close()
