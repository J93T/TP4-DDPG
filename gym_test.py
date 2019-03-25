import gym


env = gym.make('CartPole-v0')


state = env.reset()


for i in range(10000):

    env.render()

    action = env.action_space.sample()

    s_next, r, done, _ = env.step(action)

    if done:
        break
