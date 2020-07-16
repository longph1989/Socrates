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
