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

for i in [3, 5, 10]:
    path = f'./dataset/maze{i}/'
    if os.path.exists(path) is False:
        os.mkdir(path)
        os.mkdir(path + 'images/')

    with open(f'{path}maze.txt', 'w') as f:
        count_all = 0
        count_dict = np.zeros(5, dtype=np.int32)
        bar = Bar(f'Creating {i}', max=(image_amount // count_dict.shape[0]), suffix='%(percent).1f%% - %(eta)ds')
        while count_dict.min() < (image_amount // count_dict.shape[0]):
            version = 1
            env = gym.make(f"maze-sample-{i}x{i}-v1", version=version)

            previous = env.reset()
            if np.random.random() < 0.75:
                previous = deepcopy(env.set_random(True))

            if np.random.random() < 0:
                env.turn_augmentation_on()

            count = 0
            while count < max_iter:

                # Render current state (640, 640, 3)
                state = env.render('rgb_array')

                # Perform a random action 1
                action = env.action_space.sample()
                position, reward, done, info = env.step(action)

                # Render next state (640, 640, 3)
                next_state = env.render('rgb_array')

                # Save all data
                Image.fromarray(state).convert('RGB').save(f'{path}images/previous_{str(count_all)}.png')
                Image.fromarray(next_state).convert('RGB').save(f'{path}images/next_{str(count_all)}.png')

                temp_transformed_action = -1 if (previous == position).all() else action
                count_dict[temp_transformed_action] += 1

                f.write(f'{str(count_all)}.png;{str(count_all + 1)}.png;{previous};{position};{str(temp_transformed_action)};{str(action)}\n')

                # Save previous state for compartion
                previous = deepcopy(position)

                # Ensure the max number of iterations
                if done:
                    previous = env.reset()

                # Max iterations per maze
                count += 1
                count_all += 1

                if count_dict.min() > previous_count:
                    bar.next()
                    previous_count = count_dict.min()

            env.close()
            del env
            version = 0 if version == 100 else version + 1
