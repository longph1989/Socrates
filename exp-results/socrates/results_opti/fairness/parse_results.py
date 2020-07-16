from pytimeparse.timeparse import timeparse

folders = ['./']
file_names = ['log_bank.txt', 'log_census.txt', 'log_credit.txt']

end = 'analysis precision'

for folder in folders:
    print(folder)

    for name in file_names:
        print('#################################\n')

        print(name)
        path = folder + name

        file = open(path, 'r')
        lines = file.readlines()

        verified = 0
        falsified = 0
        time = 0

        for line in lines:
            if 'not robust' in line:
                falsified = falsified + 1
            elif 'probably robust' in line:
                verified = verified + 1
            elif line.startswith('real'):
                time = time + timeparse(line[5:])

        time = round(time)
        min = int(time / 60)
        sec = time - min * 60

        print('Verified: {}'.format(verified))
        print('Falsified: {}'.format(falsified))
        print('Time: {}m{}s'.format(min, sec))

        print('\n#################################')
