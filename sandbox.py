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

critic = Critic(1,2,3)
actor = Actor(1,2,3)
buffer = ReplayBuffer()

#------------------------------#
#------Hyperparameters---------#
#------------------------------#
num_episodes = 100
num_steps = 500
batch_size = 32
max_size = 10000
#------------------------------#
#------Hyperparameters---------#
#------------------------------#

buffer = ReplayBuffer(batch_size, max_size)
agent = DDPGAgent(buffer)

s = env.reset()

for e in range(num_episodes):
    print(e)
    for step in range(num_steps):

        #env.render()

        # Get action from Actor
        a = agent.take_action(s)
        a = env.action_space.sample()

        # Execute action, receive transition
        s_next, r, done, _ = env.step(a)

        # Save transition
        buffer.add_sample([s, a, r, s_next, done])

        batch = buffer.get_batch()


        # Train critic and actor?
        agent.update()

        s = s_next

env.close()
