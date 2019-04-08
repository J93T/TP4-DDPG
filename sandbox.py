import gym
from models.critic import Critic
from models.actor import Actor
from models.replay_buffer import ReplayBuffer
from ddpg import DDPGAgent

env = gym.make('Pendulum-v0')
#env = gym.make('CarRacing-v0')
#env = gym.make('HalfCheetah-v2')
#env = gym.make('LunarLanderContinuous-v2')
# Reproducability
env.seed(1)

print(env.action_space)
exit()


#------------------------------#
#------Hyperparameters---------#
#------------------------------#
hyperparameter = {
    'num_hidden_critic': 2,
    'num_hidden_actor': 2,
    'batch_size': 64,
    'max_buffer_size': 1e5,
    'tau': 0.001,
}
num_episodes = 100
num_steps = 500
#------------------------------#
#------Hyperparameters---------#
#------------------------------#

agent = DDPGAgent(env, hyperparameter)

s = env.reset()

for e in range(num_episodes):
    for step in range(num_steps):

        #env.render()

        # Get action from Actor
        a = agent.take_action(s)
        a = env.action_space.sample()

        # Execute action, receive transition
        s_next, r, done, _ = env.step(a)

        # Save transition
        agent.buffer_update([s, a, r, s_next, done])

        # Train critic and actor?
        agent.update()

        s = s_next

env.close()
