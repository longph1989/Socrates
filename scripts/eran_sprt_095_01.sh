mkdir results_sprt_095
mkdir results_sprt_095/eran_01

(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_relu_4_1024/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_fc) &> results_sprt_095/eran_01/log_mnist_relu_4_1024.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_relu_6_100/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_fc) &> results_sprt_095/eran_01/log_mnist_relu_6_100.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_relu_9_200/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_fc) &> results_sprt_095/eran_01/log_mnist_relu_9_200.txt

(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_sigmoid_6_500/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_conv) &> results_sprt_095/eran_01/log_mnist_sigmoid_6_500.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_sigmoid_6_500_pgd_0.1/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_conv) &> results_sprt_095/eran_01/log_mnist_sigmoid_6_500_pgd_0.1.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_sigmoid_6_500_pgd_0.3/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_conv) &> results_sprt_095/eran_01/log_mnist_sigmoid_6_500_pgd_0.3.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_tanh_6_500/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_conv) &> results_sprt_095/eran_01/log_mnist_tanh_6_500.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_tanh_6_500_pgd_0.1/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_conv) &> results_sprt_095/eran_01/log_mnist_tanh_6_500_pgd_0.1.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_tanh_6_500_pgd_0.3/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_conv) &> results_sprt_095/eran_01/log_mnist_tanh_6_500_pgd_0.3.txt

(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_conv_big_relu_diffai/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_conv) &> results_sprt_095/eran_01/log_mnist_conv_big_relu_diffai.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_conv_small_relu/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_conv) &> results_sprt_095/eran_01/log_mnist_conv_small_relu.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_conv_small_relu_diffai/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_conv) &> results_sprt_095/eran_01/log_mnist_conv_small_relu_diffai.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_conv_small_relu_pgd/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_conv) &> results_sprt_095/eran_01/log_mnist_conv_small_relu_pgd.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/mnist_conv_super_relu_diffai/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset mnist_conv) &> results_sprt_095/eran_01/log_mnist_conv_super_relu_diffai.txt

(time python -u source/run_eran.py --spec benchmark/eran/nnet/cifar_relu_6_100/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset cifar_fc) &> results_sprt_095/eran_01/log_cifar_relu_6_100.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/cifar_relu_7_1024/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset cifar_fc) &> results_sprt_095/eran_01/log_cifar_relu_7_1024.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/cifar_relu_9_200/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset cifar_fc) &> results_sprt_095/eran_01/log_cifar_relu_9_200.txt

(time python -u source/run_eran.py --spec benchmark/eran/nnet/cifar_conv_big_relu_diffai/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset cifar_conv) &> results_sprt_095/eran_01/log_cifar_conv_big_relu_diffai.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/cifar_conv_small_relu/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset cifar_conv) &> results_sprt_095/eran_01/log_cifar_conv_small_relu.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/cifar_conv_small_relu_diffai/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset cifar_conv) &> results_sprt_095/eran_01/log_cifar_conv_small_relu_diffai.txt
(time python -u source/run_eran.py --spec benchmark/eran/nnet/cifar_conv_small_relu_pgd/spec.json --algorithm sprt --threshold 0.95 --eps 0.1 --dataset cifar_conv) &> results_sprt_095/eran_01/log_cifar_conv_small_relu_pgd.txt
