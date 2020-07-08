mkdir results_opti
mkdir results_opti/mnist

(time python -u source/run_mnist.py --spec benchmark/mnist_challenge/spec.json --algorithm optimize) &> results_opti/mnist/log_mnist.txt
