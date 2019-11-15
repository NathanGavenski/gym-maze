import gym
import gym_maze
import numpy as np
import time
from copy import deepcopy
import os

from PIL import Image
from progress.bar import Bar


states = np.ndarray((0, 2, 640, 640, 3))
transformed_label = np.ndarray((0, 1))
true_label = np.ndarray((0, 1))
max_iter = 200
image_amount = 10000
epochs = image_amount // max_iter
previous_count = 0


if os.path.exists('./dataset/') is False:
    os.mkdir('./dataset/')

path = f'./dataset/maze10/'
if os.path.exists(path) is False:
    os.mkdir(path)
    os.mkdir(path + 'images/')

with open(f'{path}random.txt', 'w') as f:
    for z in range(100):
        for i in range(10):
            env = gym.make("maze-sample-10x10-v1", version=z + 1)
            done = False
            env.reset()

            total_reward = 0
            while done is False:
                # Perform a random action
                action = env.action_space.sample()
                position, reward, done, info = env.step(action)
                total_reward += reward

            print(f'{z}/100\t{i}/10\t{total_reward}')
            f.write(f'{z};{i};{total_reward}\n')
            f.flush()
            env.close()
            del env
