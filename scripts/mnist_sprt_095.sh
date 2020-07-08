mkdir results_sprt_095
mkdir results_sprt_095/mnist

(time python -u source/run_mnist.py --spec benchmark/mnist_challenge/spec.json --algorithm sprt --threshold 0.95) &> results_sprt_095/mnist/log_mnist.txt
