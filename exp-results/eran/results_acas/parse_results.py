import os

def main():
    verified = [0] * 11
    falsified = [0] * 11
    timeout = [0] * 11
    time = [0.0] * 11

    folder = './refinepoly/'

    for file in os.listdir(folder):
        i = int(file[4:file.index('_')])
        finished = False

        lines = open(folder + file, 'r')

        for line in lines:
            if 'Verified' in line:
                verified[i] = verified[i] + 1
            elif 'Failed' in line:
                falsified[i] = falsified[i] + 1
            elif 'Total time needed' in line:
                time[i] = time[i] + float(line[19:-8])
                finished = True

        if not finished:
            timeout[i] = timeout[i] + 1
            time[i] = time[i] + 60.0

    for i in range(1,11):
        print('Prop {}'.format(i))

        rtime = round(time[i])
        min = int(rtime / 60)
        sec = rtime - 60 * min

        print('Verified = {}'.format(verified[i]))
        print('Falsified = {}'.format(falsified[i]))
        print('Timeout = {}'.format(timeout[i]))
        print('Time = {}m{}s'.format(min, sec))

        print('\n########################')


if __name__ == '__main__':
    main()
