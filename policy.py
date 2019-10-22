import gym
import gym_maze
import numpy as np
import time
from copy import deepcopy

from PIL import Image
from progress.bar import Bar

for i in range(1, 11):
    np.random.seed(100)
    env = gym.make(f"maze-sample-10x10-v1", version=1)
    env.reset()
    env.render() 
    time.sleep(5)
    env.close()
    del env
