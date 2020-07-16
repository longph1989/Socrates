from os import walk
from pytimeparse.timeparse import timeparse

paths = ['./prop1/', './prop2/', './prop3/', './prop4/', './prop5/', './prop6/', './prop7/', './prop8/', './prop9/', './prop10/']

for path in paths:
    print(path)

    H0 = 0
    H1 = 0
    time = 0
    samples = 0

    for (dirpath, dirnames, filenames) in walk(path):
        for filename in filenames:
            file = open(dirpath + filename, 'r')
            lines = file.readlines()
            for line in lines:
                if 'H0' in line:
                    H0 = H0 + 1
                    data = line.split(' ')
                    samples = samples + int(data[-2])
                elif 'H1' in line:
                    H1 = H1 + 1
                    data = line.split(' ')
                    samples = samples + int(data[-2])
                elif line.startswith('real'):
                    time = time + timeparse(line[5:])

    time = round(time)

    print('H0 = {}'.format(H0))
    print('H0 = {}'.format(H1))
    print('Time = {}s'.format(time))
    print('Samples: {}'.format(samples))
