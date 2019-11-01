for i in [3, 5, 10, 100]:
    count = {}
    count['-1'] = 0 
    count['0'] = 0
    count['1'] = 0
    count['2'] = 0
    count['3'] = 0

    with open(f'./dataset/maze{i}/maze.txt') as f:
        for line in f:
            words = line.split(';')
            count[words[2]] += 1
    
    print(i, count)
