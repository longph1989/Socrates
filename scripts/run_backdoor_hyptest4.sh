mkdir results_backdoor_hyptest4

echo Running mnist_relu_3_10.tf
(timeout 120m time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_3_10/spec.json --algorithm backdoor --num_imgs 10 --rate 0.8 --dataset mnist_fc) &> results_backdoor_hyptest4/log_mnist_relu_3_10.tf.txt
echo Running mnist_relu_5_50.tf
(timeout 600m time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_relu_5_50/spec.json --algorithm backdoor --num_imgs 10 --rate 0.8 --dataset mnist_fc) &> results_backdoor_hyptest4/log_mnist_relu_5_50.tf.txt
echo Running mnist_sigmoid_3_10.tf
(timeout 120m time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_3_10/spec.json --algorithm backdoor --num_imgs 10 --rate 0.8 --dataset mnist_fc) &> results_backdoor_hyptest4/log_mnist_sigmoid_3_10.tf.txt
echo Running mnist_sigmoid_5_50.tf
(timeout 600m time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_sigmoid_5_50/spec.json --algorithm backdoor --num_imgs 10 --rate 0.8 --dataset mnist_fc) &> results_backdoor_hyptest4/log_mnist_sigmoid_5_50.tf.txt
echo Running mnist_tanh_3_10.tf
(timeout 120m time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_3_10/spec.json --algorithm backdoor --num_imgs 10 --rate 0.8 --dataset mnist_fc) &> results_backdoor_hyptest4/log_mnist_tanh_3_10.tf.txt
echo Running mnist_tanh_5_50.tf
(timeout 600m time python -u source/run_backdoor.py --spec benchmark/cegar/nnet/mnist_tanh_5_50/spec.json --algorithm backdoor --num_imgs 10 --rate 0.8 --dataset mnist_fc) &> results_backdoor_hyptest4/log_mnist_tanh_5_50.tf.txt

echo Done! > results_backdoor_hyptest4/done.txt
