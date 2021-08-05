mkdir results_backdoor_repair

echo Running mnist_relu_3_10.tf
(time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_3_10/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 3) &> results_backdoor_repair/log_bd_mnist_relu_3_10.tf.txt
# echo Running mnist_relu_3_20.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_3_20/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 2) &> results_backdoor_repair/log_bd_mnist_relu_3_20.tf.txt
# echo Running mnist_relu_3_30.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_3_30/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 6) &> results_backdoor_repair/log_bd_mnist_relu_3_30.tf.txt
# echo Running mnist_relu_3_40.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_3_40/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 9) &> results_backdoor_repair/log_bd_mnist_relu_3_40.tf.txt
# echo Running mnist_relu_3_50.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_3_50/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 0) &> results_backdoor_repair/log_bd_mnist_relu_3_50.tf.txt
# echo Running mnist_relu_4_10.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_4_10/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 2) &> results_backdoor_repair/log_bd_mnist_relu_4_10.tf.txt
# echo Running mnist_relu_4_20.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_4_20/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 7) &> results_backdoor_repair/log_bd_mnist_relu_4_20.tf.txt
# echo Running mnist_relu_4_30.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_4_30/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 0) &> results_backdoor_repair/log_bd_mnist_relu_4_30.tf.txt
# echo Running mnist_relu_4_40.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_4_40/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 9) &> results_backdoor_repair/log_bd_mnist_relu_4_40.tf.txt
# echo Running mnist_relu_4_50.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_4_50/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 0) &> results_backdoor_repair/log_bd_mnist_relu_4_50.tf.txt
# echo Running mnist_relu_5_10.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_5_10/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 9) &> results_backdoor_repair/log_bd_mnist_relu_5_10.tf.txt
# echo Running mnist_relu_5_20.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_5_20/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 3) &> results_backdoor_repair/log_bd_mnist_relu_5_20.tf.txt
# echo Running mnist_relu_5_30.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_5_30/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 9) &> results_backdoor_repair/log_bd_mnist_relu_5_30.tf.txt
# echo Running mnist_relu_5_40.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_5_40/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 6) &> results_backdoor_repair/log_bd_mnist_relu_5_40.tf.txt
# echo Running mnist_relu_5_50.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_relu_5_50/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 1) &> results_backdoor_repair/log_bd_mnist_relu_5_50.tf.txt
# echo Running mnist_sigmoid_3_10.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_3_10/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 0) &> results_backdoor_repair/log_bd_mnist_sigmoid_3_10.tf.txt
# echo Running mnist_sigmoid_3_20.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_3_20/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 0) &> results_backdoor_repair/log_bd_mnist_sigmoid_3_20.tf.txt
# echo Running mnist_sigmoid_3_30.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_3_30/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 9) &> results_backdoor_repair/log_bd_mnist_sigmoid_3_30.tf.txt
# echo Running mnist_sigmoid_3_40.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_3_40/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 0) &> results_backdoor_repair/log_bd_mnist_sigmoid_3_40.tf.txt
# echo Running mnist_sigmoid_3_50.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_3_50/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 5) &> results_backdoor_repair/log_bd_mnist_sigmoid_3_50.tf.txt
# echo Running mnist_sigmoid_4_10.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_4_10/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 1) &> results_backdoor_repair/log_bd_mnist_sigmoid_4_10.tf.txt
# echo Running mnist_sigmoid_4_20.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_4_20/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 9) &> results_backdoor_repair/log_bd_mnist_sigmoid_4_20.tf.txt
# echo Running mnist_sigmoid_4_30.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_4_30/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 0) &> results_backdoor_repair/log_bd_mnist_sigmoid_4_30.tf.txt
# echo Running mnist_sigmoid_4_40.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_4_40/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 6) &> results_backdoor_repair/log_bd_mnist_sigmoid_4_40.tf.txt
# echo Running mnist_sigmoid_4_50.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_4_50/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 9) &> results_backdoor_repair/log_bd_mnist_sigmoid_4_50.tf.txt
# echo Running mnist_sigmoid_5_10.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_5_10/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 5) &> results_backdoor_repair/log_bd_mnist_sigmoid_5_10.tf.txt
# echo Running mnist_sigmoid_5_20.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_5_20/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 7) &> results_backdoor_repair/log_bd_mnist_sigmoid_5_20.tf.txt
# echo Running mnist_sigmoid_5_30.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_5_30/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 2) &> results_backdoor_repair/log_bd_mnist_sigmoid_5_30.tf.txt
# echo Running mnist_sigmoid_5_40.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_5_40/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 6) &> results_backdoor_repair/log_bd_mnist_sigmoid_5_40.tf.txt
# echo Running mnist_sigmoid_5_50.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_sigmoid_5_50/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 0) &> results_backdoor_repair/log_bd_mnist_sigmoid_5_50.tf.txt
# echo Running mnist_tanh_3_10.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_3_10/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 0) &> results_backdoor_repair/log_bd_mnist_tanh_3_10.tf.txt
# echo Running mnist_tanh_3_20.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_3_20/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 0) &> results_backdoor_repair/log_bd_mnist_tanh_3_20.tf.txt
# echo Running mnist_tanh_3_30.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_3_30/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 4) &> results_backdoor_repair/log_bd_mnist_tanh_3_30.tf.txt
# echo Running mnist_tanh_3_40.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_3_40/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 4) &> results_backdoor_repair/log_bd_mnist_tanh_3_40.tf.txt
# echo Running mnist_tanh_3_50.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_3_50/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 4) &> results_backdoor_repair/log_bd_mnist_tanh_3_50.tf.txt
# echo Running mnist_tanh_4_10.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_4_10/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 7) &> results_backdoor_repair/log_bd_mnist_tanh_4_10.tf.txt
# echo Running mnist_tanh_4_20.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_4_20/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 7) &> results_backdoor_repair/log_bd_mnist_tanh_4_20.tf.txt
# echo Running mnist_tanh_4_30.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_4_30/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 3) &> results_backdoor_repair/log_bd_mnist_tanh_4_30.tf.txt
# echo Running mnist_tanh_4_40.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_4_40/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 2) &> results_backdoor_repair/log_bd_mnist_tanh_4_40.tf.txt
# echo Running mnist_tanh_4_50.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_4_50/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 0) &> results_backdoor_repair/log_bd_mnist_tanh_4_50.tf.txt
# echo Running mnist_tanh_5_10.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_5_10/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 8) &> results_backdoor_repair/log_bd_mnist_tanh_5_10.tf.txt
# echo Running mnist_tanh_5_20.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_5_20/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 8) &> results_backdoor_repair/log_bd_mnist_tanh_5_20.tf.txt
# echo Running mnist_tanh_5_30.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_5_30/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 7) &> results_backdoor_repair/log_bd_mnist_tanh_5_30.tf.txt
# echo Running mnist_tanh_5_40.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_5_40/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 8) &> results_backdoor_repair/log_bd_mnist_tanh_5_40.tf.txt
# echo Running mnist_tanh_5_50.tf
# (time python -u source/run_backdoor_repair.py --spec benchmark/backdoor/bd_nnet/bd_mnist_tanh_5_50/spec.json --algorithm backdoor_repair --dataset mnist_fc --target 3) &> results_backdoor_repair/log_bd_mnist_tanh_5_50.tf.txt

# echo Done! > results_backdoor_repair/done.txt