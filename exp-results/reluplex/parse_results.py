def main():
    for i in range(1,11):
        file = 'logs/property' + str(i) + '_summary.txt'
        lines = open(file, 'r')

        print('########################\n')
        print('Prop {}'.format(i))

        verified = 0
        falsified = 0
        timeout = 0
        time = 0

        for line in lines:
            data = line.split(', ')
            time = time + int(data[2])

            if data[1] == 'SAT':
                falsified = falsified + 1
            elif data[1] == 'UNSAT':
                verified = verified + 1
            elif data[1] == 'TIMEOUT':
                timeout = timeout + 1
            else:
                raise Error()

        time = round(time / 1000)
        min = int(time / 60)
        sec = time - 60 * min

        print('Verified = {}'.format(verified))
        print('Falsified = {}'.format(falsified))
        print('Timeout = {}'.format(timeout))
        print('Time = {}m{}s'.format(min, sec))

        print('\n########################')


if __name__ == '__main__':
    main()
