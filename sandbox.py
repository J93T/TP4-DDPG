import gym
from models.critic import Critic
from models.actor import Actor
from models.replay_buffer import ReplayBuffer
from ddpg import DDPGAgent
from util.rendering import rendering
import numpy as np

env = gym.make('Pendulum-v0')
#env = gym.make('MountainCarContinuous-v0')
#env = gym.make('LunarLanderContinuous-v2')

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
    'gamma': 0.95,
    'batch_size': 64,
    'max_buffer_size': 1e6,
    'tau': 0.01,
    'noise_sigma': 0.2,
}

#------------------------------#
#------Hyperparameters---------#
#------------------------------#

num_episodes = 400
num_steps = 1000

for i in range(10):
    agent = DDPGAgent(env, hyperparameter)
    rets =[]
    render = False
    for e in range(num_episodes):
        ret = 0
        s = env.reset()
        for step in range(num_steps):

            a = agent.take_action(s, greedy=False)
            s_next, r, done, _ = env.step(a)

            # Press Enter in the console to activate/deactivate Rendering
            render = rendering(env, render, r)
            ret += r
            agent.buffer_update([s, a, r/10, s_next, 1 - done])
            if done:
                break
            agent.update()
            s = s_next
        if e % 10 == 0 and e > 0:
            agent.save_models(e)
        score = agent.collect(5,num_steps)
        rets.append(ret)
        print("episode: {}/{}, reward: {}, avg greedy score {}".format(e,num_episodes, ret, score))

    np.save('pend_rets_'+str(i)+'.npy',rets)

env.close()
