mkdir results_sprt_099
mkdir results_sprt_099/mnist

(time python -u source/run_mnist.py --spec benchmark/mnist_challenge/spec.json --algorithm sprt --threshold 0.99) &> results_sprt_099/mnist/log_mnist.txt
