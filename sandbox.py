import gym
from models.critic import Critic
from models.actor import Actor
from models.replay_buffer import ReplayBuffer
from ddpg import DDPGAgent
from rendering import rendering
import numpy as np

#env = gym.make('Pendulum-v0')
env = gym.make('MountainCarContinuous-v0')
#env = gym.make('LunarLanderContinuous-v2')
# Reproducability
#env.seed(1)
#print(env.action_space)

#------------------------------#
#------Hyperparameters---------#
#------------------------------#


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

num_episodes = 200
num_steps = 600
warmup_steps = 50000

#------------------------------#
#------Hyperparameters---------#
#------------------------------#

for i in range(10):
    agent = DDPGAgent(env, hyperparameter)
    #agent.warmup(warmup_steps,num_steps)
    #agent.load_models(95)
    total_steps = 0
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
            agent.buffer_update([s, a, r, s_next, 1 - done])
            if done:
                break
            agent.update()
            total_steps += 1
            s = s_next
        score = 0
        if e%1 == 0 and e>0:

            agent.save_models(e)
            score = agent.collect(1,num_steps)

        rets.append(ret)
        print("episode: {}/{}, reward: {}, score {}".format(e,num_episodes,ret, score))

    np.save('results/data/car_rets_'+str(i)+'.npy',rets)

env.close()
