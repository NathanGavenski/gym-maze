from PIL import Image
import numpy as np

with open('./maze.txt') as f:
    for line in f:
        words = line.replace('\n', '').split(';')
        if int(words[2]) == -1:
            s = Image.open(f'./images/{words[0]}').convert('RGB')
            nS = Image.open(f'./images/{words[1]}').convert('RGB')

            if np.unique(np.array(s) - np.array(nS)).sum() > 0:
                print(words[0], words[1])
