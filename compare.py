from PIL import Image
import numpy as np

with open('./dataset/maze10/maze.txt') as f:
    for line in f:
        words = line.replace('\n', '').split(';')
        if int(words[2]) == -1:
            s = Image.open(f'./dataset/maze10/images/{words[0]}').convert('RGB')
            nS = Image.open(f'./dataset/maze10/images/{words[1]}').convert('RGB')

            print(np.unique(np.array(s) - np.array(nS)))
