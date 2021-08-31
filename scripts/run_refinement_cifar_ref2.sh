mkdir results_refinement_cifar_ref2

echo Running cifar_relu_6_100.tf
(time python -u source/run_refinement_cifar.py --spec benchmark/cegar/nnet/cifar_relu_6_100/spec.json --algorithm refinement --has_ref --ref_typ 2 --max_ref 100 --dataset cifar_fc) &> results_refinement_cifar_ref2/log_cifar_relu_6_100.tf.txt

# echo Running cifar_relu_7_1024.tf
# (time python -u source/run_refinement_cifar.py --spec benchmark/cegar/nnet/cifar_relu_7_1024/spec.json --algorithm refinement --has_ref --ref_typ 2 --max_ref 100 --dataset cifar_fc --num_tests 10) &> results_refinement_cifar_ref2/log_cifar_relu_7_1024.tf.txt

echo Running cifar_relu_9_200.tf
(time python -u source/run_refinement_cifar.py --spec benchmark/cegar/nnet/cifar_relu_9_200/spec.json --algorithm refinement --has_ref --ref_typ 2 --max_ref 100 --dataset cifar_fc) &> results_refinement_cifar_ref2/log_cifar_relu_9_200.tf.txt

echo Done!
