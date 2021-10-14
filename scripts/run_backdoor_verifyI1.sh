mkdir results_backdoor_verifyI1

echo Running mnist_relu_3_10.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_3_10/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_3_10.tf.txt
echo Running mnist_relu_3_20.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_3_20/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_3_20.tf.txt
echo Running mnist_relu_3_30.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_3_30/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_3_30.tf.txt
echo Running mnist_relu_3_40.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_3_40/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_3_40.tf.txt
echo Running mnist_relu_3_50.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_3_50/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_3_50.tf.txt
echo Running mnist_relu_4_10.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_4_10/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_4_10.tf.txt
echo Running mnist_relu_4_20.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_4_20/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_4_20.tf.txt
echo Running mnist_relu_4_30.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_4_30/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_4_30.tf.txt
echo Running mnist_relu_4_40.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_4_40/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_4_40.tf.txt
echo Running mnist_relu_4_50.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_4_50/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_4_50.tf.txt
echo Running mnist_relu_5_10.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_5_10/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_5_10.tf.txt
echo Running mnist_relu_5_20.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_5_20/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_5_20.tf.txt
echo Running mnist_relu_5_30.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_5_30/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_5_30.tf.txt
echo Running mnist_relu_5_40.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_5_40/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_5_40.tf.txt
echo Running mnist_relu_5_50.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_5_50/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_5_50.tf.txt
echo Running mnist_sigmoid_3_10.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_3_10/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_3_10.tf.txt
echo Running mnist_sigmoid_3_20.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_3_20/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_3_20.tf.txt
echo Running mnist_sigmoid_3_30.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_3_30/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_3_30.tf.txt
echo Running mnist_sigmoid_3_40.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_3_40/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_3_40.tf.txt
echo Running mnist_sigmoid_3_50.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_3_50/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_3_50.tf.txt
echo Running mnist_sigmoid_4_10.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_4_10/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_4_10.tf.txt
echo Running mnist_sigmoid_4_20.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_4_20/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_4_20.tf.txt
echo Running mnist_sigmoid_4_30.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_4_30/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_4_30.tf.txt
echo Running mnist_sigmoid_4_40.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_4_40/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_4_40.tf.txt
echo Running mnist_sigmoid_4_50.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_4_50/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_4_50.tf.txt
echo Running mnist_sigmoid_5_10.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_5_10/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_5_10.tf.txt
echo Running mnist_sigmoid_5_20.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_5_20/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_5_20.tf.txt
echo Running mnist_sigmoid_5_30.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_5_30/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_5_30.tf.txt
echo Running mnist_sigmoid_5_40.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_5_40/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_5_40.tf.txt
echo Running mnist_sigmoid_5_50.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_5_50/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_sigmoid_5_50.tf.txt
echo Running mnist_tanh_3_10.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_3_10/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_3_10.tf.txt
echo Running mnist_tanh_3_20.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_3_20/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_3_20.tf.txt
echo Running mnist_tanh_3_30.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_3_30/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_3_30.tf.txt
echo Running mnist_tanh_3_40.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_3_40/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_3_40.tf.txt
echo Running mnist_tanh_3_50.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_3_50/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_3_50.tf.txt
echo Running mnist_tanh_4_10.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_4_10/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_4_10.tf.txt
echo Running mnist_tanh_4_20.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_4_20/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_4_20.tf.txt
echo Running mnist_tanh_4_30.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_4_30/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_4_30.tf.txt
echo Running mnist_tanh_4_40.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_4_40/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_4_40.tf.txt
echo Running mnist_tanh_4_50.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_4_50/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_4_50.tf.txt
echo Running mnist_tanh_5_10.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_5_10/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_5_10.tf.txt
echo Running mnist_tanh_5_20.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_5_20/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_5_20.tf.txt
echo Running mnist_tanh_5_30.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_5_30/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_5_30.tf.txt
echo Running mnist_tanh_5_40.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_5_40/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_5_40.tf.txt
echo Running mnist_tanh_5_50.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_5_50/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_tanh_5_50.tf.txt

echo Running mnist_relu_4_1024.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_4_1024/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc --num_procs 5) &> results_backdoor_verifyI1/log_mnist_relu_4_1024.tf.txt
echo Running mnist_relu_6_100.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_6_100/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_6_100.tf.txt
echo Running mnist_relu_9_200.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_9_200/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_fc) &> results_backdoor_verifyI1/log_mnist_relu_9_200.tf.txt

echo Running mnist_conv_small_relu.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_conv_small_relu/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_conv) &> results_backdoor_verifyI1/log_mnist_conv_small_relu.tf.txt
echo Running mnist_conv_small_relu_diffai.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_conv_small_relu_diffai/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_conv) &> results_backdoor_verifyI1/log_mnist_conv_small_relu_diffai.tf.txt
echo Running mnist_conv_small_relu_pgd.tf
(time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_conv_small_relu_pgd/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset mnist_conv) &> results_backdoor_verifyI1/log_mnist_conv_small_relu_pgd.tf.txt

# echo Running cifar_relu_6_100.tf
# (time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/cifar_relu_6_100/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset cifar_fc) &> results_backdoor_verifyI1/log_cifar_relu_6_100.tf.txt
# echo Running cifar_relu_9_200.tf
# (time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/cifar_relu_9_200/spec.json --algorithm backdoor --total_imgs 100 --rate 1.0 --dataset cifar_fc) &> results_backdoor_verifyI1/log_cifar_relu_9_200.tf.txt

echo Done! > results_backdoor_verifyI1/done.txt
