import gym
import gym_maze
import numpy as np

from progress.bar import Bar

previous = None

states = np.ndarray((0, 2, 640, 640, 3))
transformed_label = np.ndarray((0, 1))
true_label = np.ndarray((0, 1))

max_iter = 200
image_amount = 10000
epochs = image_amount // max_iter

bar = Bar('Creating', max=image_amount, suffix='%(percent).1f%% - %(eta)ds')

for _ in range(epochs):
    env = gym.make("maze-random-10x10-v0")
    state = env.reset()
    env.set_random(True)

    count = 0
    while count < max_iter:

        # Render current state (640, 640, 3)
        state = env.render('rgb_array')

        # Perform a random action 1
        action = env.action_space.sample()
        _, reward, done, info = env.step(action)
        previous = state

        # Render next state (640, 640, 3)
        next_state = env.render('rgb_array')

        # Save all data
        temp_state = state.reshape((1, *state.shape))
        temp_next_state = next_state.reshape((1, *next_state.shape))
        images = np.append(temp_state, temp_next_state, axis=0)
        images = images.reshape((1, *images.shape))
        states = np.append(states, images, axis=0)

        if np.array_equal(np.array(previous), state):
            temp_transformed_action = -1
            temp_transformed_action = np.array(temp_transformed_action).reshape((1, 1))
        else:
            temp_transformed_action = np.array(action).reshape((1, 1))
        transformed_label = np.append(transformed_label, temp_transformed_action, axis=0)

        temp_action = np.array(action).reshape((1, 1))
        true_label = np.append(true_label, temp_action, axis=0)

        # Ensure the max number of iterations
        if done:
            env.reset()

        # Max iterations per maze
        count += 1
        bar.next()

    env.close()
    del env

np.savez_compressed('pickle_10x10', images=states, true_label=true_label, transformed_label=transformed_label)
