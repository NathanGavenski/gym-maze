import gym
import gym_maze
import numpy as np

from PIL import Image
from progress.bar import Bar


states = np.ndarray((0, 2, 640, 640, 3))
transformed_label = np.ndarray((0, 1))
true_label = np.ndarray((0, 1))

max_iter = 200
image_amount = 10000
epochs = image_amount // max_iter

bar = Bar('Creating', max=image_amount, suffix='%(percent).1f%% - %(eta)ds')

count_dict = np.zeros(5, dtype=np.int32)

with open('./maze.txt', 'w') as f:
    count_all = 0
    while count_dict.min() < (image_amount // count_dict.shape[0]):
        env = gym.make("maze-random-3x3-v0")
        state = env.reset()
        previous = env.set_random(True)

        count = 0
        while count < max_iter:

            # Render current state (640, 640, 3)
            state = env.render('rgb_array')

            # Perform a random action 1
            action = env.action_space.sample()
            position, reward, done, info = env.step(action)

            # Render next state (640, 640,´[[""]] 3)
            next_state = env.render('rgb_array')

            # Save all data
            Image.fromarray(state).convert('RGB').save(f'./images/{str(count_all)}.png')
            Image.fromarray(next_state).convert('RGB').save(f'./images/{str(count_all + 1)}.png')

            temp_transformed_action = -1 if (previous == position).all() else action
            count_dict[temp_transformed_action] += 1

            f.write(f'{str(count_all)}.png;{str(count_all + 1)}.png;{str(temp_transformed_action)};{str(action)}\n')

            # Save previous state for compartion
            previous = position

            # Ensure the max number of iterations
            if done:
                previous = env.reset()

            # Max iterations per maze
            count += 1
            count_all += 1
            bar.next()

        env.close()
        del env
