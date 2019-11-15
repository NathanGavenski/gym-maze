import sys
import numpy as np
import math
import random
import time

import gym
import gym_maze
import os

from PIL import Image


def simulate():

    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99

    num_streaks = 0

    # Render tha maze
    env.render()

    for episode in range(NUM_EPISODES):

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)
        total_reward = 0

        for t in range(MAX_T):

            # Select an action
            action = select_action(state_0, explore_rate)

            # execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if DEBUG_MODE == 2:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("")

            elif DEBUG_MODE == 1:
                if done or t >= MAX_T - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("")

            # Render tha maze
            if RENDER_MAZE:
                env.render()

            if env.is_game_over():
                sys.exit()

            if done:
                # print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                    #   % (episode, t, total_reward, num_streaks))

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                pass
                # print("Episode %d timed out at %d with total reward = %f."
                    #   % (episode, t, total_reward))

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(q_table[state]))
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


def write_status(z, i, reward, initial_count, count):
    with open('expert.txt', 'a') as f:
        f.write(f'{z};{i};{reward};{initial_count};{count}\n')


if __name__ == "__main__":

    count = 0
    if os.path.exists('./policy_dataset/maze10/') is False:
        os.mkdir('./policy_dataset/')
        os.mkdir('./policy_dataset/maze10/')

    with open('./policy_dataset/maze10/maze.txt', 'w') as f:
        for z in range(100):
            # Initialize the "maze" environment
            env = gym.make("maze-sample-10x10-v1", version=z + 1)
            for i in range(10):
                '''
                Defining the environment related constants
                '''
                # Number of discrete states (bucket) per state dimension
                MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
                NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

                # Number of discrete actions
                NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
                # Bounds for each discrete state
                STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

                '''
                Learning related constants
                '''
                MIN_EXPLORE_RATE = 0.001
                MIN_LEARNING_RATE = 0.2
                DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

                '''
                Defining the simulation related constants
                '''
                NUM_EPISODES = 50000
                MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
                STREAK_TO_END = 100
                SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
                DEBUG_MODE = 0
                RENDER_MAZE = True
                ENABLE_RECORDING = True

                '''
                Creating a Q-Table for each state-action pair
                '''
                q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

                '''
                Begin simulation
                '''
                recording_folder = "/recordings/"

                simulate()

                obv = env.reset()
                done = False
                explore_rate = get_explore_rate(127)

                total_reward = 0
                initial_count = count
                while not done:
                    state = env.render('rgb_array')
                    state_0 = state_to_bucket(obv)

                    action = select_action(state_0, explore_rate)
                    obv, reward, done, _ = env.step(action)
                    nState = env.render('rgb_array')

                    Image.fromarray(state).convert('RGB').save(f'./policy_dataset/maze10/prev_{count}.png')
                    Image.fromarray(nState).convert('RGB').save(f'./policy_dataset/maze10/next_{count}.png')
                    f.write(f'prev_{count}.png;next_{count}.png;{action}\n')
                    total_reward += reward
                    count += 1

                write_status(z + 1, i, total_reward, initial_count, count-1)
                print(f'{str(z + 1)}/10 \t{str(i + 1)}/10')
            env.close()
            del env
