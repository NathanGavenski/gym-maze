import gym
import gym_maze
import numpy as np

previous = None

for _ in range(10):
    env = gym.make("maze-random-10x10-v0")
    state = env.reset()
    env.set_random(True)

    count = 0
    while count < 200:

        image = env.render('rgb_array')
        if np.array_equal(np.array(previous), image):
            pass
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        previous = image
        count += 1
    
        if done:
            env.reset()

    env.close()
    del env
