mkdir results_sprt_090
mkdir results_sprt_090/mnist

(time python -u source/run_mnist.py --spec benchmark/mnist_challenge/spec.json --algorithm sprt --threshold 0.90) &> results_sprt_090/mnist/log_mnist.txt
