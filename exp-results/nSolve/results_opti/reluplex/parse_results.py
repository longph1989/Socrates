from os import walk
from pytimeparse.timeparse import timeparse

paths = ['./prop1/', './prop2/', './prop3/', './prop4/', './prop5/', './prop6/', './prop7/', './prop8/', './prop9/', './prop10/']

for path in paths:
    print('################################\n')
    print(path)

    verified = 0
    falsified = 0
    time = 0

    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        for filename in filenames:
            file = open(dirpath + filename, 'r')
            lines = file.readlines()
            for line in lines:
                if ' satisfied' in line:
                    verified = verified + 1
                elif 'unsatisfied' in line:
                    falsified = falsified + 1
                elif line.startswith('real'):
                    time = time + timeparse(line[5:])

    time = round(time)

    print('Verified = {}'.format(verified))
    print('Falsified = {}'.format(falsified))
    print('Time = {}s'.format(time))

    print('\n################################')
