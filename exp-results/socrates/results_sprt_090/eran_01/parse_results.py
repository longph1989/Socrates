from pytimeparse.timeparse import timeparse

folders = ['./']
file_names = ['log_mnist_relu_4_1024.txt', 'log_mnist_relu_6_100.txt', 'log_mnist_relu_9_200.txt', 'log_mnist_sigmoid_6_500.txt' \
    , 'log_mnist_sigmoid_6_500_pgd_0.1.txt', 'log_mnist_sigmoid_6_500_pgd_0.3.txt', 'log_mnist_tanh_6_500.txt', 'log_mnist_tanh_6_500_pgd_0.1.txt' \
    , 'log_mnist_tanh_6_500_pgd_0.3.txt', 'log_mnist_conv_small_relu.txt', 'log_mnist_conv_small_relu_diffai.txt', 'log_mnist_conv_small_relu_pgd.txt' \
    , 'log_mnist_conv_big_relu_diffai.txt', 'log_mnist_conv_super_relu_diffai.txt', 'log_cifar_relu_6_100.txt', 'log_cifar_relu_7_1024.txt' \
    , 'log_cifar_relu_9_200.txt', 'log_cifar_conv_small_relu.txt', 'log_cifar_conv_small_relu_diffai.txt', 'log_cifar_conv_small_relu_pgd.txt' \
    , 'log_cifar_conv_big_relu_diffai.txt']

end = 'analysis precision'

for folder in folders:
    print(folder)

    for name in file_names:
        print('#################################\n')

        print(name)
        path = folder + name

        file = open(path, 'r')
        lines = file.readlines()

        H0 = 0
        H1 = 0
        time = 0
        samples = 0

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
        min = int(time / 60)
        sec = time - min * 60

        print('H0: {}'.format(H0))
        print('H1: {}'.format(H1))
        print('Time: {}m{}s'.format(min, sec))
        print('Samples: {}'.format(samples))

        print('\n#################################')
